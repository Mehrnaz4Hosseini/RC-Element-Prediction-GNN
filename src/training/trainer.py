"""
Training Pipeline for HGT on RC Element Prediction

Handles:
- Device management (CUDA / MPS / CPU)
- K-fold cross validation
- Training/validation loops
- Early stopping
- Model checkpointing
- Comprehensive metrics tracking
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
from sklearn.model_selection import KFold
import logging
import copy

logger = logging.getLogger("TRAINER")


class DeviceManager:
    """Manages device selection for different hardware (CUDA, MPS, CPU)."""

    @staticmethod
    def get_device() -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device


class MetricsTracker:
    """Tracks and aggregates metrics across training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def update(self, epoch_metrics: Dict[str, float], model_state=None):
        """Update metrics with epoch results."""
        for key, value in epoch_metrics.items():
            self.metrics[key].append(value)

        # Track best model
        val_loss = epoch_metrics.get("val_loss", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.metrics.get("val_loss", [])) - 1
            if model_state is not None:
                self.best_model_state = copy.deepcopy(model_state)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

        # Add final metrics
        for key, values in self.metrics.items():
            if values:
                summary[f"final_{key}"] = values[-1]
                summary[f"best_{key}"] = min(values) if "loss" in key else max(values)

        return summary


class StructuralElementLoss(nn.Module):
    """
    Custom loss function for structural element prediction.

    Features:
    - Weighted MSE (can emphasize width vs height differently)
    - Physical constraint penalty (e.g., beam width should be reasonable)
    - Per-type weighting (beams vs columns importance)
    """

    def __init__(
        self,
        width_weight: float = 1.0,
        height_weight: float = 1.0,
        beam_weight: float = 1.0,
        column_weight: float = 1.0,
        use_physical_constraints: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "target_weights", torch.tensor([width_weight, height_weight])
        )
        self.beam_weight = beam_weight
        self.column_weight = column_weight
        self.use_physical_constraints = use_physical_constraints

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted MSE loss with optional physical constraints.

        Args:
            predictions: {'beam': [N_b, 2], 'column': [N_c, 2]}
            targets: {'beam': [N_b, 2], 'column': [N_c, 2]}

        Returns:
            total_loss, loss_components_dict
        """
        total_loss = 0.0
        loss_components = {}

        for node_type in predictions.keys():
            if node_type in targets and targets[node_type].numel() > 0:
                pred = predictions[node_type]
                true = targets[node_type]

                # Weighted MSE (width vs height importance)
                diff = pred - true
                weighted_diff = diff * self.target_weights.to(diff.device)
                mse = torch.mean(weighted_diff**2)

                # Type-specific weighting
                type_weight = (
                    self.beam_weight if node_type == "beam" else self.column_weight
                )
                weighted_mse = type_weight * mse

                total_loss += weighted_mse
                loss_components[f"{node_type}_loss"] = weighted_mse.item()

                # Optional: Physical constraint penalties
                if self.use_physical_constraints:
                    # Beams typically have width > height
                    if node_type == "beam":
                        width, height = pred[:, 0], pred[:, 1]
                        constraint_violation = torch.relu(height - width)
                        constraint_loss = torch.mean(constraint_violation) * 0.1
                        total_loss += constraint_loss
                        loss_components[f"{node_type}_constraint"] = (
                            constraint_loss.item()
                        )

        return total_loss, loss_components


class HGTrainer:
    """
    Trainer for Heterogeneous Graph Transformer.

    Handles complete training workflow:
    1. K-fold cross validation
    2. Training with validation
    3. Early stopping
    4. Model checkpointing
    5. Comprehensive evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: HGT model instance
            config: Configuration dictionary (from hgt.yaml)
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.config = config

        # Device setup
        self.device = device or DeviceManager.get_device()
        self.model = self.model.to(self.device)

        # Training parameters
        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.patience = training_config.get("patience", 20)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)
        self.grad_clip = training_config.get("grad_clip", 1.0)
        self.scheduler_factor = training_config.get("scheduler_factor", 0.5)
        self.scheduler_patience = training_config.get("scheduler_patience", 10)

        # Loss function
        self.criterion = StructuralElementLoss(
            width_weight=training_config.get("width_weight", 1.0),
            height_weight=training_config.get("height_weight", 1.0),
            beam_weight=training_config.get("beam_weight", 1.0),
            column_weight=training_config.get("column_weight", 1.0),
        )

        # Checkpointing
        paths_config = config.get("paths", {})
        self.checkpoint_dir = Path(paths_config.get("checkpoints", "checkpoints/hgt"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # K-fold
        self.k_folds = training_config.get("k_folds", 5)

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(
            f"Training config: epochs={self.epochs}, lr={self.learning_rate}, folds={self.k_folds}"
        )

    def _prepare_graph(self, graph) -> Tuple:
        """Move graph data to device with MPS compatibility fixes."""
        try:
            # Clone graph to avoid modifying original
            if hasattr(graph, "clone"):
                graph_device = graph.clone()
            else:
                graph_device = graph

            # Move to device (handle MPS limitations)
            graph_device = graph_device.to(self.device)

            # Extract features and targets
            x_dict = {}
            y_dict = {}

            for node_type in (
                graph_device.node_types
                if hasattr(graph_device, "node_types")
                else ["beam", "column"]
            ):
                try:
                    node_data = graph_device[node_type]
                    if hasattr(node_data, "x") and node_data.x is not None:
                        x_dict[node_type] = (
                            node_data.x.float()
                        )  # Ensure float32 for MPS
                    if hasattr(node_data, "y") and node_data.y is not None:
                        y_dict[node_type] = node_data.y.float()
                except Exception as e:
                    logger.debug(f"Could not extract {node_type} data: {e}")
                    continue

            # Get edge index dict
            edge_index_dict = {}
            if hasattr(graph_device, "edge_index_dict"):
                for edge_type, edge_data in graph_device.edge_index_dict.items():
                    edge_index_dict[edge_type] = (
                        edge_data.long()
                    )  # Ensure long for indices

            return x_dict, edge_index_dict, y_dict, graph_device

        except Exception as e:
            logger.error(f"Error preparing graph: {e}")
            # Fallback to CPU if MPS fails
            if self.device.type == "mps":
                logger.warning("MPS failed, falling back to CPU for this graph")
                graph_cpu = graph.cpu() if hasattr(graph, "cpu") else graph

                x_dict = {}
                y_dict = {}
                for node_type in ["beam", "column"]:
                    try:
                        node_data = graph_cpu[node_type]
                        if hasattr(node_data, "x") and node_data.x is not None:
                            x_dict[node_type] = node_data.x.float()
                        if hasattr(node_data, "y") and node_data.y is not None:
                            y_dict[node_type] = node_data.y.float()
                    except:
                        continue

                edge_index_dict = {}
                if hasattr(graph_cpu, "edge_index_dict"):
                    for edge_type, edge_data in graph_cpu.edge_index_dict.items():
                        edge_index_dict[edge_type] = edge_data.long()

                return x_dict, edge_index_dict, y_dict, graph_cpu
            else:
                raise

    def train_epoch(self, train_graphs: List) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_graphs = len(train_graphs)

        for i, graph in enumerate(train_graphs):
            try:
                # Prepare data
                x_dict, edge_index_dict, y_dict, graph_device = self._prepare_graph(
                    graph
                )

                # Forward pass
                predictions = self.model(graph_device)

                # Compute loss
                loss, loss_components = self.criterion(predictions, y_dict)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

                # Accumulate metrics
                epoch_metrics["train_loss"] += loss.item()
                for key, value in loss_components.items():
                    epoch_metrics[f"train_{key}"] += value

                # Clear MPS cache periodically
                if self.device.type == "mps" and i % 10 == 0:
                    torch.mps.empty_cache()

            except Exception as e:
                logger.error(f"Error training on graph {i}: {e}")
                continue

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_graphs)

        return dict(epoch_metrics)

    @torch.no_grad()
    def validate_epoch(self, val_graphs: List) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_graphs: List of validation HeteroData graphs

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_graphs = len(val_graphs)

        for graph in val_graphs:
            # Prepare data
            x_dict, edge_index_dict, y_dict, graph_device = self._prepare_graph(graph)

            # Forward pass
            predictions = self.model(graph_device)

            # Compute loss
            loss, loss_components = self.criterion(predictions, y_dict)

            # Accumulate metrics
            epoch_metrics["val_loss"] += loss.item()
            for key, value in loss_components.items():
                epoch_metrics[f"val_{key}"] += value

            # Additional metrics per node type
            for node_type in predictions.keys():
                if node_type in y_dict and y_dict[node_type].numel() > 0:
                    pred = predictions[node_type]
                    true = y_dict[node_type]

                    # MAE
                    mae = torch.mean(torch.abs(pred - true))
                    epoch_metrics[f"val_{node_type}_mae"] += mae.item()

                    # Per-dimension MAE
                    for dim, name in enumerate(["width", "height"]):
                        dim_mae = torch.mean(torch.abs(pred[:, dim] - true[:, dim]))
                        epoch_metrics[f"val_{node_type}_{name}_mae"] += dim_mae.item()

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_graphs

        return dict(epoch_metrics)

    def train_fold(
        self,
        train_graphs: List,
        val_graphs: List,
        fold: int = 0,
    ) -> Dict[str, Any]:
        """
        Train model on a single fold.

        Args:
            train_graphs: Training graphs for this fold
            val_graphs: Validation graphs for this fold
            fold: Fold number (for logging)

        Returns:
            Training history for this fold
        """
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Fold {fold + 1}: Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs"
        )
        logger.info(f"{'='*60}")

        # Reset model for each fold (train from scratch)
        self.model.apply(self._reset_weights)

        # Initialize optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True,
        )

        # Early stopping variables
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        # Training loop
        fold_metrics = defaultdict(list)

        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch(train_graphs)

            # Validation
            val_metrics = self.validate_epoch(val_graphs)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch + 1
            epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

            # Store metrics
            for key, value in epoch_metrics.items():
                fold_metrics[key].append(value)

            # Learning rate scheduling
            val_loss = val_metrics.get("val_loss", float("inf"))
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save best model for this fold
                self._save_checkpoint(best_model_state, fold, val_loss)

                logger.info(f"📈 New best model! Val loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = (
                    f"Fold {fold+1} | Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

                # Add key metrics if available
                for metric in ["val_beam_mae", "val_column_mae"]:
                    if metric in val_metrics:
                        log_msg += f" | {metric}: {val_metrics[metric]:.4f}"

                logger.info(log_msg)

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Return fold results
        fold_results = {
            "fold": fold + 1,
            "best_val_loss": best_val_loss,
            "best_epoch": fold_metrics["val_loss"].index(min(fold_metrics["val_loss"]))
            + 1,
            "metrics": dict(fold_metrics),
        }

        return fold_results

    def train_with_kfold(
        self,
        graphs: List,
        n_folds: Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train with K-fold cross validation.

        Args:
            graphs: List of all HeteroData graphs
            n_folds: Number of folds (default from config)
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training results for all folds
        """
        n_folds = n_folds or self.k_folds
        n_graphs = len(graphs)

        if n_graphs < n_folds:
            logger.warning(
                f"Only {n_graphs} graphs available, reducing folds to {n_graphs}"
            )
            n_folds = n_graphs

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {n_folds}-Fold Cross Validation")
        logger.info(f"Total graphs: {n_graphs}")
        logger.info(f"{'='*60}")

        # K-Fold split
        kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

        all_fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs)):
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]

            # Train this fold
            fold_results = self.train_fold(train_graphs, val_graphs, fold)
            all_fold_results.append(fold_results)

        # Aggregate results across folds
        cv_results = self._aggregate_cv_results(all_fold_results)

        # Save overall results
        self._save_cv_results(cv_results)

        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-Validation Complete!")
        logger.info(
            f"Average Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}"
        )
        logger.info(f"{'='*60}")

        return cv_results

    def train_simple(
        self,
        train_graphs: List,
        val_graphs: List,
    ) -> Dict[str, Any]:
        """
        Simple train/validation split training (no k-fold).

        Args:
            train_graphs: Training graphs
            val_graphs: Validation graphs

        Returns:
            Training results
        """
        return self.train_fold(train_graphs, val_graphs, fold=0)

    def evaluate(self, test_graphs: List) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_graphs: List of test HeteroData graphs

        Returns:
            Dictionary with test metrics and predictions
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {len(test_graphs)} test graphs")
        logger.info(f"{'='*60}")

        self.model.eval()
        all_predictions = []
        test_metrics = defaultdict(list)

        for i, graph in enumerate(test_graphs):
            # Prepare data
            x_dict, edge_index_dict, y_dict, graph_device = self._prepare_graph(graph)

            # Predict
            predictions = self.model(graph_device)

            # Store predictions
            pred_dict = {}
            for node_type in predictions.keys():
                pred_dict[node_type] = predictions[node_type].cpu().numpy()
            all_predictions.append(pred_dict)

            # Compute metrics
            for node_type in predictions.keys():
                if node_type in y_dict and y_dict[node_type].numel() > 0:
                    pred = predictions[node_type]
                    true = y_dict[node_type]

                    # Overall MAE
                    mae = torch.mean(torch.abs(pred - true)).item()
                    test_metrics[f"{node_type}_mae"].append(mae)

                    # Per-dimension MAE
                    for dim, name in enumerate(["width", "height"]):
                        dim_mae = torch.mean(
                            torch.abs(pred[:, dim] - true[:, dim])
                        ).item()
                        test_metrics[f"{node_type}_{name}_mae"].append(dim_mae)

                    # RMSE
                    rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
                    test_metrics[f"{node_type}_rmse"].append(rmse)

                    # R² score
                    ss_res = torch.sum((true - pred) ** 2)
                    ss_tot = torch.sum((true - torch.mean(true, dim=0)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    test_metrics[f"{node_type}_r2"].append(r2.item())

        # Average metrics
        final_metrics = {
            key: {
                "mean": np.mean(values),
                "std": np.std(values),
            }
            for key, values in test_metrics.items()
        }

        # Print results
        logger.info(f"\nTest Results:")
        logger.info(f"{'='*60}")
        for metric_name, values in final_metrics.items():
            logger.info(
                f"  {metric_name:25s}: {values['mean']:.4f} ± {values['std']:.4f}"
            )
        logger.info(f"{'='*60}")

        return {
            "metrics": final_metrics,
            "predictions": all_predictions,
        }

    def _reset_weights(self, module):
        """Reset model weights for fresh training in each fold."""
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def _save_checkpoint(self, model_state: Dict, fold: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "fold": fold,
            "model_state_dict": model_state,
            "val_loss": val_loss,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        path = self.checkpoint_dir / f"fold_{fold}_best.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all folds."""
        # Collect metrics across folds
        cv_metrics = defaultdict(list)

        for fold_result in fold_results:
            cv_metrics["best_val_loss"].append(fold_result["best_val_loss"])
            cv_metrics["best_epoch"].append(fold_result["best_epoch"])

            # Get final epoch metrics
            fold_metrics = fold_result["metrics"]
            for key in fold_metrics:
                if key.startswith("val_"):
                    cv_metrics[f"final_{key}"].append(fold_metrics[key][-1])

        # Compute statistics
        cv_results = {
            "n_folds": len(fold_results),
            "fold_results": fold_results,
        }

        for key, values in cv_metrics.items():
            cv_results[f"mean_{key}"] = np.mean(values)
            cv_results[f"std_{key}"] = np.std(values)

        return cv_results

    def _save_cv_results(self, cv_results: Dict[str, Any]):
        """Save cross-validation results to JSON."""
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for key, value in cv_results.items():
            if key == "fold_results":
                serializable_results[key] = value  # Keep as-is
            elif isinstance(value, (np.floating, np.integer)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value

        path = self.checkpoint_dir / "cv_results.json"
        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"CV results saved to {path}")

    def load_best_model(self, fold: int = 0) -> nn.Module:
        """Load the best model from a specific fold."""
        path = self.checkpoint_dir / f"fold_{fold}_best.pt"

        if not path.exists():
            logger.error(f"No checkpoint found: {path}")
            return self.model

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded best model from fold {fold} (val_loss: {checkpoint['val_loss']:.4f})"
        )

        return self.model
