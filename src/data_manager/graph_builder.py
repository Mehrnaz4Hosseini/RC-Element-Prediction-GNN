"""
Heterogeneous Graph Builder for RC Element Prediction.
Converts sample data into heterogeneous graphs with positional encoding.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

# Import processors and logger
try:
    from src.data_manager.data_processor import (
        normalize_features,
        add_positional_encoding,
    )
    from src.utils.logger import get_graph_logger, get_system_logger
except ImportError:
    # Fallback imports
    import logging

    get_graph_logger = lambda: logging.getLogger("GRAPH")
    get_system_logger = lambda: logging.getLogger("SYSTEM")

    # Local fallback functions
    def normalize_features(features, scaler=None):
        if scaler is None:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        return scaler.fit_transform(features), scaler

    def add_positional_encoding(data, **kwargs):
        return data


class HeteroGraphBuilder:
    """
    Builds heterogeneous graphs from structural engineering data.
    Handles node feature extraction, edge construction, and positional encoding.
    """

    def __init__(self, config: Dict):
        """
        Initialize Heterogeneous Graph Builder.

        Args:
            config: Configuration dictionary from YAML
        """
        self.logger = get_graph_logger()
        self.sys_logger = get_system_logger()

        self.logger.info("Initializing HeteroGraphBuilder")

        self.config = config
        self._init_configuration()

        # Initialize state
        self.scalers = {"beam": None, "column": None}
        self.original_feat_counts = {}
        self.feat_cols_dict = {}

        # Performance tracking
        self.stats = {
            "graphs_built": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "failed_builds": 0,
            "avg_connectivity": 0.0,
        }

        self.logger.info("GraphBuilder initialized successfully")
        self.logger.debug(f"PE enabled: {self.pe_enabled}, type: {self.pe_type}")

    def _init_configuration(self):
        """Initialize configuration parameters."""
        data_config = self.config.get("data", {})
        pe_config = data_config.get("pe", {})
        cols = data_config.get("columns", {})

        # PE configuration
        self.pe_enabled = pe_config.get("enabled", True)
        self.pe_dim = pe_config.get("dim", 8)
        self.pe_type = pe_config.get("type", "hetero_laplacian")
        self.include_cross_type = pe_config.get("include_cross_type", False)
        self.pe_normalize = pe_config.get("normalize", True)

        # Column mappings
        self.source_col = cols.get("source_node", "Source")
        self.target_col = cols.get("target_node", "Target")
        self.width_col = cols.get("label_width", "Width (b)")
        self.height_col = cols.get("label_height", "Height (h)")

        # Internal columns
        self.row_index_col = "row_index"
        self.exclude_from_features = ["Ele_Type", self.row_index_col]

        # Logging level
        self.verbose = False

    def set_verbose(self, verbose: bool):
        """Set verbose logging mode."""
        self.verbose = verbose
        # Get logging level from the logging module
        import logging

        level = logging.DEBUG if verbose else logging.INFO
        self.logger.setLevel(level)
        self.logger.debug(f"Verbose mode set to: {verbose}")

    def build_hetero_graph(self, sample_data: Dict) -> Optional[HeteroData]:
        """
        Convert sample dictionary into HeteroData graph.

        Args:
            sample_data: Dictionary containing nodes, edges, and labels

        Returns:
            HeteroData object or None if build fails
        """
        sample_name = sample_data.get("sample_name", "unknown")

        if self.verbose:
            self.logger.info(f"Building graph for sample: {sample_name}")
        else:
            self.logger.debug(f"Building graph: {sample_name}")

        try:
            data = HeteroData()

            # Process data
            self._process_nodes(data, sample_data, sample_name)
            self._process_edges(data, sample_data, sample_name)

            # Add positional encoding if enabled
            if self.pe_enabled and self.pe_dim > 0:
                self._add_positional_encoding(data, sample_name)

            # Add metadata
            self._add_metadata(data, sample_data, sample_name)

            # Update statistics
            self._update_statistics(data, sample_name)

            self.stats["graphs_built"] += 1
            self.logger.info(f"Successfully built graph for {sample_name}")

            return data

        except Exception as e:
            self.logger.error(f"Failed to build graph for {sample_name}: {e}")
            self.stats["failed_builds"] += 1
            return None

    def _process_nodes(self, data: HeteroData, sample_data: Dict, sample_name: str):
        """Process beam and column nodes."""
        beam_df = sample_data["nodes"]["beam"]
        column_df = sample_data["nodes"]["column"]
        labels_df = sample_data["labels_raw"]

        # Create label mapping
        label_mapping = self._create_label_mapping(labels_df, sample_name)

        # Process each node type
        for node_type, df in [("beam", beam_df), ("column", column_df)]:
            if df.empty:
                self.logger.warning(f"Empty {node_type} dataframe in {sample_name}")
                continue

            self._process_node_type(data, node_type, df, label_mapping, sample_name)

    def _create_label_mapping(
        self, labels_df: pd.DataFrame, sample_name: str
    ) -> Dict[int, List[float]]:
        """Create mapping from row index to labels."""
        label_mapping = {}

        if labels_df.empty:
            self.logger.warning(f"No labels found for {sample_name}")
            return label_mapping

        # Ensure row_index column exists
        if self.row_index_col not in labels_df.columns:
            self.logger.debug(f"Adding row_index to labels for {sample_name}")
            labels_df[self.row_index_col] = labels_df.index

        # Parse labels
        valid_labels = 0
        for _, row in labels_df.iterrows():
            try:
                row_index = int(row[self.row_index_col])
                width = float(row.get(self.width_col, 0))
                height = float(row.get(self.height_col, 0))

                label_mapping[row_index] = [width, height]
                valid_labels += 1

            except (ValueError, KeyError) as e:
                self.logger.warning(f"Could not parse label row: {e}")
                continue

        self.logger.debug(
            f"Created label mapping with {valid_labels} entries for {sample_name}"
        )
        return label_mapping

    def _process_node_type(
        self,
        data: HeteroData,
        node_type: str,
        df: pd.DataFrame,
        label_mapping: Dict,
        sample_name: str,
    ):
        """Process nodes of a specific type."""
        try:
            # Extract features
            feat_cols = [
                col for col in df.columns if col not in self.exclude_from_features
            ]

            self.feat_cols_dict[node_type] = feat_cols
            self.original_feat_counts[node_type] = len(feat_cols)

            if self.verbose:
                self.logger.debug(f"  {node_type}: {len(feat_cols)} feature columns")

            x_raw = df[feat_cols].values.astype(np.float32)

            # Normalize features
            x_norm, self.scalers[node_type] = normalize_features(
                x_raw, self.scalers[node_type]
            )

            data[node_type].x = torch.from_numpy(x_norm).float()

            # Attach labels
            y_list = []
            for _, row in df.iterrows():
                try:
                    row_index = int(row[self.row_index_col])
                    labels = label_mapping.get(row_index, [0.0, 0.0])
                    y_list.append([float(labels[0]), float(labels[1])])
                except (ValueError, KeyError):
                    y_list.append([0.0, 0.0])

            y = np.array(y_list, dtype=np.float32)
            data[node_type].y = torch.from_numpy(y).float()

            if self.verbose:
                self.logger.debug(
                    f"  {node_type}: {data[node_type].x.shape[0]} nodes, "
                    f"{data[node_type].x.shape[1]} features"
                )

        except Exception as e:
            self.logger.error(f"Error processing {node_type} nodes: {e}")
            raise

    def _process_edges(self, data: HeteroData, sample_data: Dict, sample_name: str):
        """Process edges and create adjacency."""
        edges_df = sample_data["edges_raw"]

        if edges_df.empty:
            self.logger.warning(f"No edges found for {sample_name}")
            return

        # Create row index to node mapping
        row_to_node_info = self._create_node_mapping(sample_data, sample_name)

        # Process edges
        edge_bundles, match_stats = self._build_edge_bundles(
            edges_df, row_to_node_info, sample_name
        )

        # Assign edges to graph
        self._assign_edges_to_graph(data, edge_bundles)

        # Store connectivity statistics
        data.match_percentage = match_stats["percentage"]
        data.valid_edges = match_stats["valid"]
        data.total_edges = match_stats["total"]

        if self.verbose:
            self.logger.debug(
                f"  Edge matching: {match_stats['valid']}/{match_stats['total']} "
                f"({match_stats['percentage']:.1f}%)"
            )

    def _create_node_mapping(
        self, sample_data: Dict, sample_name: str
    ) -> Dict[int, Tuple[str, int]]:
        """Create mapping from row index to (node_type, local_index)."""
        row_to_node_info = {}

        # Process beams
        beam_idx = 0
        beam_df = sample_data["nodes"]["beam"]
        for _, row in beam_df.iterrows():
            try:
                row_index = int(row[self.row_index_col])
                row_to_node_info[row_index] = ("beam", beam_idx)
                beam_idx += 1
            except (ValueError, KeyError):
                continue

        # Process columns
        col_idx = 0
        column_df = sample_data["nodes"]["column"]
        for _, row in column_df.iterrows():
            try:
                row_index = int(row[self.row_index_col])
                row_to_node_info[row_index] = ("column", col_idx)
                col_idx += 1
            except (ValueError, KeyError):
                continue

        self.logger.debug(
            f"Created node mapping for {sample_name}: "
            f"{beam_idx} beams, {col_idx} columns"
        )

        return row_to_node_info

    def _build_edge_bundles(
        self, edges_df: pd.DataFrame, row_to_node_info: Dict, sample_name: str
    ) -> Tuple[Dict, Dict]:
        """Build edge bundles from raw edge data."""
        edge_bundles = {
            ("beam", "to", "beam"): ([], []),
            ("column", "to", "column"): ([], []),
            ("beam", "to", "column"): ([], []),
            ("column", "to", "beam"): ([], []),  # Add reverse direction
        }

        valid_edges = 0
        invalid_edges = 0
        invalid_examples = []

        for idx, row in edges_df.iterrows():
            try:
                src_row_idx = int(row[self.source_col])
                dst_row_idx = int(row[self.target_col])

                if src_row_idx in row_to_node_info and dst_row_idx in row_to_node_info:
                    src_type, src_local_idx = row_to_node_info[src_row_idx]
                    dst_type, dst_local_idx = row_to_node_info[dst_row_idx]

                    # Add forward edge
                    forward_key = (src_type, "to", dst_type)
                    edge_bundles[forward_key][0].append(src_local_idx)
                    edge_bundles[forward_key][1].append(dst_local_idx)

                    # Add reverse edge for cross-type connections
                    if src_type != dst_type:
                        reverse_key = (dst_type, "to", src_type)
                        edge_bundles[reverse_key][0].append(dst_local_idx)
                        edge_bundles[reverse_key][1].append(src_local_idx)

                    valid_edges += 1
                else:
                    invalid_edges += 1
                    if len(invalid_examples) < 3:
                        missing = []
                        if src_row_idx not in row_to_node_info:
                            missing.append(f"source={src_row_idx}")
                        if dst_row_idx not in row_to_node_info:
                            missing.append(f"target={dst_row_idx}")
                        invalid_examples.append(
                            f"{src_row_idx}->{dst_row_idx} ({', '.join(missing)})"
                        )

            except (ValueError, KeyError) as e:
                invalid_edges += 1
                if len(invalid_examples) < 3:
                    invalid_examples.append(f"row {idx}: {str(e)[:50]}")

        # Log invalid edges if any
        if invalid_edges > 0:
            self.logger.warning(
                f"Sample {sample_name}: {invalid_edges} invalid edges "
                f"({invalid_edges/(valid_edges + invalid_edges)*100:.1f}%)"
            )
            if self.verbose and invalid_examples:
                self.logger.debug(
                    f"  Invalid examples: {', '.join(invalid_examples[:3])}"
                )

        # Calculate statistics
        total_edges = valid_edges + invalid_edges
        match_percentage = (valid_edges / total_edges * 100) if total_edges > 0 else 0

        stats = {
            "valid": valid_edges,
            "invalid": invalid_edges,
            "total": total_edges,
            "percentage": match_percentage,
        }

        return edge_bundles, stats

    def _assign_edges_to_graph(self, data: HeteroData, edge_bundles: Dict):
        """Assign edge bundles to graph data structure."""
        for (src_t, rel, dst_t), (src_list, dst_list) in edge_bundles.items():
            if src_list and dst_list:
                data[src_t, rel, dst_t].edge_index = torch.tensor(
                    [src_list, dst_list], dtype=torch.long
                )
            else:
                # Create empty edge index for consistency
                data[src_t, rel, dst_t].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )

    def _add_positional_encoding(self, data: HeteroData, sample_name: str):
        """Add positional encoding to the graph."""
        try:
            self.logger.debug(
                f"Adding {self.pe_type} positional encoding to {sample_name}"
            )

            data = add_positional_encoding(
                data=data,
                pe_type=self.pe_type,
                pe_dim=self.pe_dim,
                include_cross_type=self.include_cross_type,
                normalize=self.pe_normalize,
                verbose=self.verbose,
            )

            if self.verbose:
                for node_type in data.node_types:
                    if hasattr(data[node_type], "x"):
                        orig_features = data[node_type].x.shape[1] - self.pe_dim
                        self.logger.debug(
                            f"  {node_type}: {data[node_type].x.shape[1]} features "
                            f"({orig_features} orig + {self.pe_dim} PE)"
                        )

        except Exception as e:
            self.logger.error(
                f"Failed to add positional encoding to {sample_name}: {e}"
            )

    def _add_metadata(self, data: HeteroData, sample_data: Dict, sample_name: str):
        """Add metadata to graph."""
        data.sample_name = sample_name
        data.split = sample_data.get("split", "unknown")

        # Calculate connectivity
        connected_nodes = 0
        total_nodes = 0

        for node_type in data.node_types:
            if hasattr(data[node_type], "x"):
                num_nodes = data[node_type].x.shape[0]
                total_nodes += num_nodes

                # Find connected nodes
                connected_set = set()
                for edge_type in data.edge_types:
                    edges = data[edge_type].edge_index
                    if edges.shape[1] > 0:
                        src_type, _, dst_type = edge_type
                        if src_type == node_type:
                            connected_set.update(edges[0].tolist())
                        if dst_type == node_type:
                            connected_set.update(edges[1].tolist())

                connected_nodes += len(connected_set)

        connectivity = connected_nodes / total_nodes if total_nodes > 0 else 0
        data.connectivity_ratio = connectivity

        if self.verbose:
            self.logger.debug(
                f"  Connectivity: {connected_nodes}/{total_nodes} ({connectivity*100:.1f}%)"
            )

    def _update_statistics(self, data: HeteroData, sample_name: str):
        """Update builder statistics."""
        # Count nodes and edges
        total_nodes = 0
        total_edges = 0

        for node_type in data.node_types:
            if hasattr(data[node_type], "x"):
                total_nodes += data[node_type].x.shape[0]

        for edge_type in data.edge_types:
            if hasattr(data[edge_type], "edge_index"):
                total_edges += data[edge_type].edge_index.shape[1]

        # Update running averages
        self.stats["total_nodes"] += total_nodes
        self.stats["total_edges"] += total_edges

        if hasattr(data, "connectivity_ratio"):
            self.stats["avg_connectivity"] = (
                (
                    self.stats["avg_connectivity"] * (self.stats["graphs_built"] - 1)
                    + data.connectivity_ratio
                )
                / self.stats["graphs_built"]
                if self.stats["graphs_built"] > 0
                else data.connectivity_ratio
            )

    def get_feature_dimensions(
        self, sample_data: Optional[Dict] = None
    ) -> Dict[str, int]:
        """
        Get feature dimensions for each node type.

        Args:
            sample_data: Optional sample to compute dimensions from

        Returns:
            Dictionary mapping node_type -> feature_dimension
        """
        dims = {}

        if sample_data is not None:
            # Compute from sample
            for node_type in ["beam", "column"]:
                df = sample_data["nodes"][node_type]
                if not df.empty:
                    feat_cols = [
                        col
                        for col in df.columns
                        if col not in self.exclude_from_features
                    ]
                    original_dim = len(feat_cols)
                    pe_dim = self.pe_dim if self.pe_enabled else 0
                    dims[node_type] = original_dim + pe_dim
        elif self.original_feat_counts:
            # Use cached values
            for node_type, original_dim in self.original_feat_counts.items():
                pe_dim = self.pe_dim if self.pe_enabled else 0
                dims[node_type] = original_dim + pe_dim
        else:
            # Fallback
            original_dim = 42  # Common case from your data
            pe_dim = self.pe_dim if self.pe_enabled else 0
            dims = {"beam": original_dim + pe_dim, "column": original_dim + pe_dim}

        self.logger.debug(f"Feature dimensions: {dims}")
        return dims

    def verify_feature_consistency(self, sample_data_list: List[Dict]) -> bool:
        """
        Verify that all samples have consistent feature columns.

        Args:
            sample_data_list: List of sample dictionaries

        Returns:
            True if consistent, False otherwise
        """
        self.logger.info("Verifying feature consistency across samples")

        all_feat_cols = {"beam": set(), "column": set()}

        for sample in sample_data_list:
            for node_type in ["beam", "column"]:
                df = sample["nodes"][node_type]
                if not df.empty:
                    feat_cols = [
                        col
                        for col in df.columns
                        if col not in self.exclude_from_features
                    ]
                    all_feat_cols[node_type].update(feat_cols)

        # Check consistency
        is_consistent = True
        issues = []

        for node_type, cols_set in all_feat_cols.items():
            if cols_set:
                first_sample = sample_data_list[0]
                df = first_sample["nodes"][node_type]

                if not df.empty:
                    first_cols = [
                        col
                        for col in df.columns
                        if col not in self.exclude_from_features
                    ]

                    if len(cols_set) != len(first_cols):
                        issues.append(
                            f"{node_type}: {len(cols_set)} vs {len(first_cols)} columns"
                        )
                        is_consistent = False

        if is_consistent:
            self.logger.info("✅ Feature columns are consistent across all samples")
            for node_type, cols_set in all_feat_cols.items():
                if cols_set:
                    self.logger.debug(f"  {node_type}: {len(cols_set)} features")
        else:
            self.logger.error("❌ Inconsistent feature columns detected")
            for issue in issues:
                self.logger.error(f"  {issue}")

        return is_consistent

    def reset_scalers(self):
        """Reset feature scalers."""
        self.scalers = {"beam": None, "column": None}
        self.logger.info("Reset feature scalers")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get builder statistics.

        Returns:
            Dictionary with builder statistics
        """
        avg_nodes = (
            self.stats["total_nodes"] / self.stats["graphs_built"]
            if self.stats["graphs_built"] > 0
            else 0
        )
        avg_edges = (
            self.stats["total_edges"] / self.stats["graphs_built"]
            if self.stats["graphs_built"] > 0
            else 0
        )

        return {
            "graphs_built": self.stats["graphs_built"],
            "failed_builds": self.stats["failed_builds"],
            "success_rate": (
                self.stats["graphs_built"]
                / (self.stats["graphs_built"] + self.stats["failed_builds"])
                * 100
                if (self.stats["graphs_built"] + self.stats["failed_builds"]) > 0
                else 0
            ),
            "average_nodes": avg_nodes,
            "average_edges": avg_edges,
            "average_connectivity": self.stats["avg_connectivity"] * 100,
        }
