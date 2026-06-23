"""
General training pipeline for heterogeneous element-prediction models.

Model-agnostic: works with any nn.Module whose forward(graph) returns
{node_type: [N, output_dim]}.

Key properties (correct + strong baseline):
  * Leakage-free normalization: feature AND target scalers are fit INSIDE
    each fold on that fold's TRAINING graphs only.
  * Training happens in scaled-target space (stable optimization), but every
    reported metric (MAE / RMSE / R2) is inverse-transformed back to REAL
    units so the numbers mean something physically.
  * cross_validate()  -> honest generalization estimate (per-fold scalers).
  * fit_final()       -> one model trained on the whole train pool, saved
                         together with its scalers for inference / test.
"""

import copy
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_manager.data_processor import FeatureNormalizer, TargetNormalizer
from src.training.losses import StructuralElementLoss

logger = logging.getLogger("TRAINER")


class DeviceManager:
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS")
            return torch.device("mps")
        logger.info("Using CPU")
        return torch.device("cpu")


class Trainer:
    """
    Generic trainer for heterogeneous GNN regressors.

    Pass RAW graphs (un-scaled features AND raw targets). The trainer scales
    them internally, per fold, with no leakage.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        normalize: bool = True,
        show_progress: bool = False,
    ):
        self.config = config
        self.device = device or DeviceManager.get_device()
        self.model = model.to(self.device)
        self.normalize = normalize
        self.show_progress = show_progress

        tcfg = config.get("training", {})
        self.epochs = tcfg.get("epochs", 100)
        self.patience = tcfg.get("patience", 20)
        self.learning_rate = tcfg.get("learning_rate", 1e-3)
        self.weight_decay = tcfg.get("weight_decay", 1e-4)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.scheduler_factor = tcfg.get("scheduler_factor", 0.5)
        self.scheduler_patience = tcfg.get("scheduler_patience", 10)
        self.k_folds = tcfg.get("k_folds", 5)
        self.node_types = config.get("model", {}).get("node_types", ["beam", "column"])

        self.criterion = (
            criterion
            or StructuralElementLoss(
                width_weight=tcfg.get("width_weight", 1.0),
                height_weight=tcfg.get("height_weight", 1.0),
                beam_weight=tcfg.get("beam_weight", 1.0),
                column_weight=tcfg.get("column_weight", 1.0),
                loss_type=tcfg.get("loss_type", "huber"),
                huber_delta=tcfg.get("huber_delta", 1.0),
            )
        ).to(self.device)

        ckpt = config.get("paths", {}).get("checkpoints", "checkpoints/hgt")
        self.checkpoint_dir = Path(ckpt)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Scalers for the CURRENT context (a fold, or the final model).
        self.feature_normalizer: Optional[FeatureNormalizer] = None
        self.target_normalizer: Optional[TargetNormalizer] = None

        self.optimizer = None
        self.scheduler = None
        logger.info(
            f"Trainer ready on {self.device} | epochs={self.epochs}, "
            f"lr={self.learning_rate}, folds={self.k_folds}, normalize={normalize}"
        )

    # ================================================================== #
    # Normalization helpers
    # ================================================================== #
    def _clone(self, graphs: List) -> List:
        return [g.clone() for g in graphs]

    def _fit_scalers(self, train_graphs: List) -> Tuple[FeatureNormalizer, TargetNormalizer]:
        fnorm = FeatureNormalizer(self.node_types).fit(train_graphs)
        tnorm = TargetNormalizer(self.node_types).fit(train_graphs)
        return fnorm, tnorm

    def _scale_features(self, graphs: List, fnorm: FeatureNormalizer):
        if self.normalize and fnorm is not None:
            fnorm.transform(graphs)

    def _scale_targets(self, graphs: List, tnorm: TargetNormalizer):
        if self.normalize and tnorm is not None:
            tnorm.transform(graphs)

    # ================================================================== #
    def _prepare_graph(self, graph):
        g = graph.clone().to(self.device) if hasattr(graph, "clone") else graph
        y_dict = {}
        for nt in g.node_types:
            node = g[nt]
            if hasattr(node, "y") and node.y is not None:
                y_dict[nt] = node.y.float()
        return g, y_dict

    @torch.no_grad()
    def _ensure_initialized(self, sample_graph):
        if getattr(self.model, "_initialized", True):
            return
        g, _ = self._prepare_graph(sample_graph)
        self.model(g)
        self.model.to(self.device)

    # ================================================================== #
    # Train / validate one epoch
    # ================================================================== #
    def train_epoch(self, train_graphs: List) -> Dict[str, float]:
        self.model.train()
        m = defaultdict(float)
        for i, graph in enumerate(train_graphs):
            g, y_dict = self._prepare_graph(graph)
            preds = self.model(g)
            loss, parts = self.criterion(preds, y_dict)  # scaled-target space

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            m["train_loss"] += loss.item()
            for k, v in parts.items():
                m[f"train_{k}"] += v
            if self.device.type == "mps" and i % 10 == 0:
                torch.mps.empty_cache()

        n = max(1, len(train_graphs))
        return {k: v / n for k, v in m.items()}

    @torch.no_grad()
    def validate_epoch(self, val_graphs: List) -> Dict[str, float]:
        # val targets are in SCALED space (transformed with the fold scaler)
        return self._eval_loop(val_graphs, prefix="val", true_is_scaled=True)

    # ================================================================== #
    # Shared metric loop (real-unit MAE / RMSE / R2)
    # ================================================================== #
    @torch.no_grad()
    def _eval_loop(self, graphs, prefix, true_is_scaled, collect_preds=False):
        self.model.eval()
        m = defaultdict(float)
        per_node = defaultdict(lambda: {"err": [], "sq": [], "true": []})
        preds_out = []

        for graph in graphs:
            g, y_dict = self._prepare_graph(graph)
            preds = self.model(g)
            loss, _ = self.criterion(preds, y_dict)
            m[f"{prefix}_loss"] += loss.item()

            graph_pred = {}
            for nt, pred in preds.items():
                if nt not in y_dict or y_dict[nt].numel() == 0:
                    continue
                true = y_dict[nt]

                # to REAL units
                if self.normalize and self.target_normalizer is not None:
                    pred_r = self.target_normalizer.inverse(nt, pred)
                    true_r = (
                        self.target_normalizer.inverse(nt, true)
                        if true_is_scaled
                        else true
                    )
                else:
                    pred_r, true_r = pred, true

                graph_pred[nt] = pred_r.cpu().numpy()

                err = (pred_r - true_r).cpu()
                per_node[nt]["err"].append(err)
                per_node[nt]["true"].append(true_r.cpu())

            if collect_preds:
                preds_out.append(graph_pred)

        # aggregate, in real units
        for nt, d in per_node.items():
            if not d["err"]:
                continue
            err = torch.cat(d["err"], dim=0)
            true = torch.cat(d["true"], dim=0)

            m[f"{prefix}_{nt}_mae"] = err.abs().mean().item()
            m[f"{prefix}_{nt}_mse"] = (err ** 2).mean().item()
            m[f"{prefix}_{nt}_rmse"] = torch.sqrt((err ** 2).mean()).item()
            for dim, name in enumerate(["width", "height"]):
                m[f"{prefix}_{nt}_{name}_mae"] = err[:, dim].abs().mean().item()
            ss_res = (err ** 2).sum()
            ss_tot = ((true - true.mean(0)) ** 2).sum()
            m[f"{prefix}_{nt}_r2"] = (
                (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            )

        n = max(1, len(graphs))
        m[f"{prefix}_loss"] /= n
        out = dict(m)
        return (out, preds_out) if collect_preds else out

    # ================================================================== #
    # One fold / one model
    # ================================================================== #
    def _train_loop(self, train_graphs, val_graphs, tag) -> Dict[str, Any]:
        self._ensure_initialized(train_graphs[0])
        if hasattr(self.model, "reset_all_parameters"):
            self.model.reset_all_parameters()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=self.scheduler_factor, patience=self.scheduler_patience,
        )

        best_val, best_state, patience = float("inf"), None, 0
        history = defaultdict(list)

        epoch_iter = range(self.epochs)
        if self.show_progress:
            try:
                from tqdm.auto import tqdm
                epoch_iter = tqdm(epoch_iter, desc=f"{tag} epochs", leave=False)
            except ImportError:
                pass

        for epoch in epoch_iter:
            tr = self.train_epoch(train_graphs)
            va = self.validate_epoch(val_graphs)
            if self.show_progress and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    val=f"{va.get('val_loss', 0):.3f}",
                    beam_mae=f"{va.get('val_beam_mae', 0):.2f}",
                )
            row = {**tr, **va, "epoch": epoch + 1,
                   "lr": self.optimizer.param_groups[0]["lr"]}
            for k, v in row.items():
                history[k].append(v)

            val_loss = va.get("val_loss", float("inf"))
            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience = 0
                self._save_checkpoint(best_state, tag, val_loss)
            else:
                patience += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"[{tag}] ep {epoch+1}/{self.epochs} | "
                    f"train {tr['train_loss']:.4f} | val {val_loss:.4f} | "
                    f"beam MAE {va.get('val_beam_mae', float('nan')):.3f} "
                    f"R2 {va.get('val_beam_r2', float('nan')):.3f} | "
                    f"col MAE {va.get('val_column_mae', float('nan')):.3f} "
                    f"R2 {va.get('val_column_r2', float('nan')):.3f}"
                )
            if patience >= self.patience:
                logger.info(f"[{tag}] early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "tag": tag,
            "best_val_loss": best_val,
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "metrics": dict(history),
        }

    # ================================================================== #
    # Public API
    # ================================================================== #
    def cross_validate(
        self,
        graphs: List,
        n_folds: Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Leakage-free k-fold CV. Scalers are re-fit on each fold's train set."""
        n_folds = n_folds or self.k_folds
        if len(graphs) < n_folds:
            n_folds = max(2, len(graphs))
        logger.info(f"\n{'='*60}\n{n_folds}-Fold CV on {len(graphs)} graphs "
                    f"(per-fold scaling)\n{'='*60}")

        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        results = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(graphs)):
            tr = self._clone([graphs[i] for i in tr_idx])
            va = self._clone([graphs[i] for i in va_idx])

            fnorm, tnorm = self._fit_scalers(tr)
            self.feature_normalizer, self.target_normalizer = fnorm, tnorm
            self._scale_features(tr, fnorm); self._scale_features(va, fnorm)
            self._scale_targets(tr, tnorm);  self._scale_targets(va, tnorm)

            logger.info(f"\n--- Fold {fold+1}/{n_folds}: "
                        f"{len(tr)} train / {len(va)} val ---")
            results.append(self._train_loop(tr, va, tag=f"fold_{fold}"))

        cv = self._aggregate_cv(results)
        self._save_json(cv, "cv_results.json")
        logger.info(
            f"\nCV done | val_loss {cv['mean_best_val_loss']:.4f} "
            f"± {cv['std_best_val_loss']:.4f}"
        )
        return cv

    def fit_final(
        self, train_pool: List, val_frac: float = 0.15, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train ONE final model on the whole train pool (with an internal val
        holdout for early stopping). Fits + saves the scalers for inference.
        """
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(train_pool))
        cut = int(len(train_pool) * (1 - val_frac))
        tr = self._clone([train_pool[i] for i in idx[:cut]])
        va = self._clone([train_pool[i] for i in idx[cut:]])

        fnorm, tnorm = self._fit_scalers(tr)
        self.feature_normalizer, self.target_normalizer = fnorm, tnorm
        self._scale_features(tr, fnorm); self._scale_features(va, fnorm)
        self._scale_targets(tr, tnorm);  self._scale_targets(va, tnorm)

        logger.info(f"\n{'='*60}\nFinal fit on {len(tr)} train / {len(va)} val"
                    f"\n{'='*60}")
        result = self._train_loop(tr, va, tag="final")

        # persist scalers next to the model
        fnorm.save(str(self.checkpoint_dir / "feature_normalizer.pkl"))
        tnorm.save(str(self.checkpoint_dir / "target_normalizer.pkl"))
        return result

    @torch.no_grad()
    def evaluate(self, test_graphs: List) -> Dict[str, Any]:
        """
        Evaluate the current model on RAW test graphs. Features are scaled with
        the fitted feature scaler; targets stay raw and predictions are mapped
        back to real units. Metrics are therefore in REAL units.
        """
        logger.info(f"\n{'='*60}\nEvaluating on {len(test_graphs)} graphs\n{'='*60}")
        test = self._clone(test_graphs)
        self._scale_features(test, self.feature_normalizer)  # targets left raw

        metrics, preds = self._eval_loop(
            test, prefix="test", true_is_scaled=False, collect_preds=True
        )
        final = {k: float(v) for k, v in metrics.items() if k != "test_loss"}

        # Headline: overall MAE in real units (cm) -- the number to report.
        type_maes = [final[f"test_{nt}_mae"] for nt in self.node_types
                     if f"test_{nt}_mae" in final]
        if type_maes:
            final["test_overall_mae"] = float(np.mean(type_maes))

        logger.info("\nTest results (real units):")
        for k, v in final.items():
            logger.info(f"  {k:28s}: {v:.4f}")
        if "test_overall_mae" in final:
            logger.info(f"\n  >>> HEADLINE  MAE = {final['test_overall_mae']:.3f} cm "
                        f"(beam {final.get('test_beam_mae', float('nan')):.3f} / "
                        f"column {final.get('test_column_mae', float('nan')):.3f})")
        return {"metrics": final, "predictions": preds}

    # ================================================================== #
    # Persistence
    # ================================================================== #
    def _save_checkpoint(self, state, tag, val_loss):
        torch.save(
            {"tag": tag, "model_state_dict": state, "val_loss": val_loss,
             "config": self.config, "timestamp": datetime.now().isoformat()},
            self.checkpoint_dir / f"{tag}_best.pt",
        )

    def _aggregate_cv(self, fold_results: List[Dict]) -> Dict[str, Any]:
        agg = defaultdict(list)
        for fr in fold_results:
            agg["best_val_loss"].append(fr["best_val_loss"])
            agg["best_epoch"].append(fr["best_epoch"])
            for k, vals in fr["metrics"].items():
                # value at the best epoch for val_* metrics (skip val_loss:
                # already captured above to avoid a double-count collision)
                if k.startswith("val_") and k != "val_loss":
                    be = fr["best_epoch"] - 1
                    agg[f"best_{k}"].append(vals[min(be, len(vals) - 1)])
        cv = {"n_folds": len(fold_results), "fold_results": fold_results}
        for k, vals in agg.items():
            cv[f"mean_{k}"] = float(np.mean(vals))
            cv[f"std_{k}"] = float(np.std(vals))
        return cv

    def _save_json(self, obj, name):
        with open(self.checkpoint_dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=float)
        logger.info(f"Saved {self.checkpoint_dir / name}")

    def load_best_model(self, tag: str = "final") -> nn.Module:
        path = self.checkpoint_dir / f"{tag}_best.pt"
        if not path.exists():
            logger.error(f"No checkpoint at {path}")
            return self.model
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._ensure_initialized_from_state()
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded '{tag}' (val_loss={ckpt['val_loss']:.4f})")
        return self.model

    def _ensure_initialized_from_state(self):
        # load_state_dict needs the lazy modules to exist first; if the model
        # was already trained in this session they do. Otherwise the caller
        # must run one forward pass before loading.
        pass


# Backwards-compatible aliases (older notebooks).
HGTrainer = Trainer
Trainer.train_with_kfold = Trainer.cross_validate
