"""
Heterogeneous Graph Builder for RC Element Prediction.
Converts sample data into heterogeneous graphs with robust positional encoding.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

from src.data_manager.data_processor import (
    add_positional_encoding,
)
from src.utils.logger import get_graph_logger, get_system_logger


class HeteroGraphBuilder:
    """
    Builds heterogeneous graphs from structural engineering data.
    Handles node feature extraction, edge construction, and positional encoding.
    """

    def __init__(self, config: Dict):
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
        self.logger.debug(f"PE enabled: {self.pe_enabled}, dim: {self.pe_dim}")

        pe_func = globals().get(
            "add_positional_encoding", globals().get("fallback_add_pe", None)
        )
        self.logger.debug(f"PE function = {pe_func}")

    # --------------------------------------------------------------------------
    def _init_configuration(self):
        """Initialize configuration parameters."""
        data_config = self.config.get("data", {})
        pe_config = data_config.get("pe", {})
        cols = data_config.get("columns", {})
        feature_handling = data_config.get("feature_handling", {})

        # PE configuration (if missing keys, safe defaults)
        self.pe_enabled = pe_config.get("enabled", True)
        self.pe_dim = pe_config.get("dim", 8)

        # NOTE: this now uses our unified robust wrapper,
        # so 'type', 'include_cross_type', 'normalize' no longer matter
        self.pe_type = pe_config.get("type", "auto")
        self.include_cross_type = pe_config.get("include_cross_type", False)
        self.pe_normalize = pe_config.get("normalize", True)

        self.source_col = cols.get("source_node", "Source")
        self.target_col = cols.get("target_node", "Target")
        self.width_col = cols.get("label_width", "Width (b)")
        self.height_col = cols.get("label_height", "Height (h)")
        self.row_index_col = "row_index"
        self.exclude_from_features = feature_handling.get(
            "exclude_from_features", ["Ele_Type", "row_index", "Element_Name"]
        )

        # Drop elements whose label is (width==0 AND height==0) or missing.
        # Done in-memory at build time only -- the source files are untouched.
        # Visualization can keep all elements by passing drop_invalid=False.
        self.drop_invalid_labels = data_config.get("drop_invalid_labels", False)

        self.verbose = False

    # --------------------------------------------------------------------------
    def set_verbose(self, verbose: bool):
        """Set verbose logging mode."""
        import logging

        level = logging.DEBUG if verbose else logging.INFO
        self.verbose = verbose
        self.logger.setLevel(level)
        self.logger.debug(f"Verbose mode set to: {verbose}")

    # --------------------------------------------------------------------------
    def build_hetero_graph(
        self, sample_data: Dict, drop_invalid: Optional[bool] = None
    ) -> Optional[HeteroData]:
        sample_name = sample_data.get("sample_name", "unknown")

        # Resolve the drop policy (explicit arg overrides config). Pass
        # drop_invalid=False when building graphs for VISUALIZATION so every
        # element is kept.
        do_drop = self.drop_invalid_labels if drop_invalid is None else drop_invalid

        self.logger.info(f"Building hetero graph for sample: {sample_name}")
        try:
            if do_drop:
                sample_data = self._drop_invalid_nodes(sample_data, sample_name)

            data = HeteroData()
            self._process_nodes(data, sample_data, sample_name)
            self._process_edges(data, sample_data, sample_name)

            # Add positional encoding if enabled
            if self.pe_enabled and self.pe_dim > 0:
                # Use our new robust automatic fallback PE
                self._add_positional_encoding(data, sample_name)

            self._add_metadata(data, sample_data, sample_name)
            self._update_statistics(data, sample_name)

            self.stats["graphs_built"] += 1
            self.logger.info(f"✅ Built graph for {sample_name}")

            return data

        except Exception as e:
            self.logger.error(f"❌ Failed to build graph for {sample_name}: {e}")
            self.stats["failed_builds"] += 1
            return None

    # --------------------------------------------------------------------------
    def _drop_invalid_nodes(self, sample_data: Dict, sample_name: str) -> Dict:
        """
        Return a copy of sample_data with beam/column rows removed when their
        label is missing or (width==0 AND height==0). Edges that referenced a
        removed node are dropped automatically downstream (the node mapping no
        longer contains them). The original dict / source files are untouched.
        """
        label_map = self._create_label_mapping(
            sample_data["labels_raw"], sample_name
        )
        new_nodes, dropped = {}, 0
        for nt in ["beam", "column"]:
            df = sample_data["nodes"][nt]
            if df.empty:
                new_nodes[nt] = df
                continue
            keep = []
            for _, row in df.iterrows():
                rid = int(row[self.row_index_col])
                lab = label_map.get(rid)
                invalid = (lab is None) or (lab[0] == 0 and lab[1] == 0)
                keep.append(not invalid)
            kept = df[pd.Series(keep, index=df.index)]
            dropped += len(df) - len(kept)
            new_nodes[nt] = kept

        if dropped:
            self.logger.info(
                f"{sample_name}: dropped {dropped} invalid (0,0)/missing-label nodes"
            )
        sd = dict(sample_data)
        sd["nodes"] = new_nodes
        return sd

    # --------------------------------------------------------------------------
    def _process_nodes(self, data: HeteroData, sample_data: Dict, sample_name: str):
        beam_df = sample_data["nodes"]["beam"]
        column_df = sample_data["nodes"]["column"]
        labels_df = sample_data["labels_raw"]

        label_mapping = self._create_label_mapping(labels_df, sample_name)

        for node_type, df in [("beam", beam_df), ("column", column_df)]:
            if df.empty:
                self.logger.warning(f"Empty {node_type} dataframe in {sample_name}")
                continue
            self._process_node_type(data, node_type, df, label_mapping, sample_name)

    # --------------------------------------------------------------------------
    def _create_label_mapping(
        self, labels_df: pd.DataFrame, sample_name: str
    ) -> Dict[int, List[float]]:
        label_mapping = {}
        if labels_df.empty:
            self.logger.warning(f"No labels found for {sample_name}")
            return label_mapping

        if self.row_index_col not in labels_df.columns:
            labels_df[self.row_index_col] = labels_df.index

        valid_labels = 0
        for _, row in labels_df.iterrows():
            try:
                idx = int(row[self.row_index_col])
                width = float(row.get(self.width_col, 0))
                height = float(row.get(self.height_col, 0))
                label_mapping[idx] = [width, height]
                valid_labels += 1
            except Exception:
                continue

        self.logger.debug(
            f"Label mapping: {valid_labels} valid entries for {sample_name}"
        )
        return label_mapping

    # --------------------------------------------------------------------------
    def _process_node_type(
        self,
        data: HeteroData,
        node_type: str,
        df: pd.DataFrame,
        label_mapping: Dict,
        sample_name: str,
    ):
        try:
            feat_cols = [
                col for col in df.columns if col not in self.exclude_from_features
            ]
            self.feat_cols_dict[node_type] = feat_cols
            self.original_feat_counts[node_type] = len(feat_cols)

            # Store RAW (un-scaled) features. Normalization is done AFTER the
            # train/test split via FeatureNormalizer to avoid data leakage.
            x_raw = df[feat_cols].values.astype(np.float32)
            data[node_type].x = torch.from_numpy(x_raw).float()
            data[node_type].num_raw_features = x_raw.shape[1]
            # Keep the column order so the PositionalEncoder can locate
            # coordinate columns (X1,Y1,Z1,X2,Y2,Z2) downstream.
            data[node_type].feature_names = list(feat_cols)

            if all(col in df.columns for col in ["X1", "Y1", "Z1"]):
                coords = df[["X1", "Y1", "Z1"]].values.astype(np.float32)
                data[node_type].pos = torch.tensor(coords, dtype=torch.float)
            else:
                print(f"[WARNING] {node_type} has no X1,Y1,Z1 columns")

            y_list = []
            for _, row in df.iterrows():
                rid = int(row[self.row_index_col])
                y_list.append(label_mapping.get(rid, [0.0, 0.0]))
            y = np.array(y_list, dtype=np.float32)
            data[node_type].y = torch.from_numpy(y).float()

            if self.verbose:
                self.logger.debug(
                    f"{node_type}: {data[node_type].x.shape[0]} nodes, {data[node_type].x.shape[1]} features"
                )

        except Exception as e:
            self.logger.error(f"Error processing {node_type}: {e}")
            raise

    # --------------------------------------------------------------------------
    def _process_edges(self, data: HeteroData, sample_data: Dict, sample_name: str):
        edges_df = sample_data["edges_raw"]
        if edges_df.empty:
            self.logger.warning(f"No edges for {sample_name}")
            return

        row_to_node_info = self._create_node_mapping(sample_data, sample_name)
        bundles, stats = self._build_edge_bundles(
            edges_df, row_to_node_info, sample_name
        )
        self._assign_edges_to_graph(data, bundles)

        data.match_percentage = stats["percentage"]
        data.valid_edges = stats["valid"]
        data.total_edges = stats["total"]

        if self.verbose:
            self.logger.debug(
                f"Edge match: {stats['valid']}/{stats['total']} ({stats['percentage']:.1f}%)"
            )

    # --------------------------------------------------------------------------
    def _create_node_mapping(
        self, sample_data: Dict, sample_name: str
    ) -> Dict[int, Tuple[str, int]]:
        mapping = {}
        beam_idx = 0
        for _, row in sample_data["nodes"]["beam"].iterrows():
            try:
                rid = int(row[self.row_index_col])
                mapping[rid] = ("beam", beam_idx)
                beam_idx += 1
            except Exception:
                continue
        col_idx = 0
        for _, row in sample_data["nodes"]["column"].iterrows():
            try:
                rid = int(row[self.row_index_col])
                mapping[rid] = ("column", col_idx)
                col_idx += 1
            except Exception:
                continue

        self.logger.debug(f"{sample_name}: mapped {beam_idx} beams, {col_idx} columns")
        return mapping

    # --------------------------------------------------------------------------
    def _build_edge_bundles(
        self, edges_df: pd.DataFrame, mapping: Dict, sample_name: str
    ):
        bundles = {
            ("beam", "to", "beam"): ([], []),
            ("column", "to", "column"): ([], []),
            ("beam", "to", "column"): ([], []),
            ("column", "to", "beam"): ([], []),
        }

        valid, invalid = 0, 0
        for _, row in edges_df.iterrows():
            try:
                s = int(row[self.source_col])
                t = int(row[self.target_col])
                if s in mapping and t in mapping:
                    stype, sidx = mapping[s]
                    dtype, didx = mapping[t]
                    bundles[(stype, "to", dtype)][0].append(sidx)
                    bundles[(stype, "to", dtype)][1].append(didx)
                    if stype != dtype:
                        bundles[(dtype, "to", stype)][0].append(didx)
                        bundles[(dtype, "to", stype)][1].append(sidx)
                    valid += 1
                else:
                    invalid += 1
            except Exception:
                invalid += 1

        stats = {
            "valid": valid,
            "invalid": invalid,
            "total": valid + invalid,
            "percentage": (
                valid / (valid + invalid) * 100 if (valid + invalid) > 0 else 0
            ),
        }
        self.logger.debug(f"{sample_name}: edge match {stats['percentage']:.1f}%")
        return bundles, stats

    # --------------------------------------------------------------------------
    def _assign_edges_to_graph(self, data: HeteroData, bundles: Dict):
        from torch_geometric.utils import coalesce, to_undirected

        for (src_t, rel, dst_t), (slist, dlist) in bundles.items():
            if slist and dlist:
                ei = torch.tensor([slist, dlist], dtype=torch.long)
                if src_t == dst_t:
                    # Same-type relations are physically undirected: enforce
                    # both directions and drop duplicates. Robust even if a
                    # source file lists an edge only one way.
                    ei = to_undirected(ei, num_nodes=data[src_t].num_nodes)
                else:
                    ei = coalesce(ei)  # dedup any repeated cross-type edges
                data[src_t, rel, dst_t].edge_index = ei
            else:
                data[src_t, rel, dst_t].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )

    # --------------------------------------------------------------------------
    def _add_positional_encoding(self, data: HeteroData, sample_name: str):
        """Hook for robust PE fallback."""
        try:
            self.logger.info(f"Adding positional encoding to {sample_name}")
            data = add_positional_encoding(
                data=data, pe_dim=self.pe_dim, verbose=self.verbose
            )
            self.logger.info(f"PE successfully added for {sample_name}")
        except Exception as e:
            self.logger.error(f"PE addition failed for {sample_name}: {e}")

    # --------------------------------------------------------------------------
    def _add_metadata(self, data: HeteroData, sample_data: Dict, sample_name: str):
        data.sample_name = sample_name
        data.split = sample_data.get("split", "unknown")

        total_nodes, connected_nodes = 0, 0
        for node_type in data.node_types:
            n = getattr(data[node_type], "x", torch.empty((0,))).shape[0]
            total_nodes += n
            conn_set = set()
            for et in data.edge_types:
                edges = data[et].edge_index
                if edges.shape[1] == 0:
                    continue
                s, _, d = et
                if s == node_type:
                    conn_set.update(edges[0].tolist())
                if d == node_type:
                    conn_set.update(edges[1].tolist())
            connected_nodes += len(conn_set)

        data.connectivity_ratio = connected_nodes / total_nodes if total_nodes else 0
        if self.verbose:
            self.logger.debug(
                f"{sample_name}: connectivity {data.connectivity_ratio*100:.1f}%"
            )

    # --------------------------------------------------------------------------
    def _update_statistics(self, data: HeteroData, sample_name: str):
        n, e = 0, 0
        for nt in data.node_types:
            x = getattr(data[nt], "x", None)
            if x is not None:
                n += x.shape[0]
        for et in data.edge_types:
            idx = getattr(data[et], "edge_index", None)
            if idx is not None:
                e += idx.shape[1]

        self.stats["total_nodes"] += n
        self.stats["total_edges"] += e

        if hasattr(data, "connectivity_ratio"):
            prev = self.stats["graphs_built"]
            self.stats["avg_connectivity"] = (
                self.stats["avg_connectivity"] * prev + data.connectivity_ratio
            ) / (prev + 1)

    # --------------------------------------------------------------------------
    def get_feature_dimensions(
        self, sample_data: Optional[Dict] = None
    ) -> Dict[str, int]:
        dims = {}
        if sample_data:
            for nt in ["beam", "column"]:
                df = sample_data["nodes"][nt]
                if not df.empty:
                    feat_cols = [
                        c for c in df.columns if c not in self.exclude_from_features
                    ]
                    dims[nt] = len(feat_cols) + (self.pe_dim if self.pe_enabled else 0)
        elif self.original_feat_counts:
            for nt, d in self.original_feat_counts.items():
                dims[nt] = d + (self.pe_dim if self.pe_enabled else 0)
        else:
            dims = {"beam": 42 + self.pe_dim, "column": 42 + self.pe_dim}

        return dims

    # --------------------------------------------------------------------------
    def verify_feature_consistency(self, sample_data_list: List[Dict]) -> bool:
        self.logger.info("Verifying feature column consistency")
        all_cols = {"beam": set(), "column": set()}

        for sample in sample_data_list:
            for nt in ["beam", "column"]:
                df = sample["nodes"][nt]
                if not df.empty:
                    cols = [
                        c for c in df.columns if c not in self.exclude_from_features
                    ]
                    all_cols[nt].update(cols)

        consistent = True
        for nt in ["beam", "column"]:
            if not all_cols[nt]:
                continue
            first = [
                c
                for c in sample_data_list[0]["nodes"][nt].columns
                if c not in self.exclude_from_features
            ]
            if len(all_cols[nt]) != len(first):
                self.logger.error(
                    f"Inconsistent {nt} features: {len(all_cols[nt])} vs {len(first)}"
                )
                consistent = False

        if consistent:
            self.logger.info("✅ Feature columns consistent across samples")
        else:
            self.logger.error("❌ Feature column mismatch detected")
        return consistent

    # --------------------------------------------------------------------------
    def reset_scalers(self):
        self.scalers = {"beam": None, "column": None}
        self.logger.info("Feature scalers reset")

    def get_statistics(self) -> Dict[str, any]:
        avg_nodes = (
            self.stats["total_nodes"] / self.stats["graphs_built"]
            if self.stats["graphs_built"]
            else 0
        )
        avg_edges = (
            self.stats["total_edges"] / self.stats["graphs_built"]
            if self.stats["graphs_built"]
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
