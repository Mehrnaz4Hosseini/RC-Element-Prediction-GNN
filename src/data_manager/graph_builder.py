import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from src.data_manager.data_processor import normalize_features, compute_laplacian_pe


class HeteroGraphBuilder:
    def __init__(self, config):
        self.config = config
        # Initialize separate scalers for beams and columns
        self.scalers = {"beam": None, "column": None}

        # Load hyperparams from config
        self.pe_dim = self.config.get("data", {}).get("pe_dim", 8)

        # Mapping column names from config
        cols = self.config.get("data", {}).get("columns", {})
        self.feat_name_col = cols.get("element_name", "Element_Name")
        self.labl_name_col = cols.get("label_element_name", "Element Name")
        self.source_col = cols.get("source_node", "Source")
        self.target_col = cols.get("target_node", "Target")

        # Target column names in Label excel
        self.width_col = cols.get("label_width", "Width (b)")
        self.height_col = cols.get("label_height", "Height (h)")

    def build_hetero_graph(self, sample_data: dict) -> HeteroData:
        """
        Main entry point: Converts sample dict from DataLoader into HeteroData.
        """

        # DEBUG: Let's find where the error is
        print(f"\n[DEBUG] Building graph for {sample_data['sample_name']}")

        # Check what columns are actually in the data
        for node_type in ["beam", "column"]:
            df = sample_data["nodes"][node_type]
            if not df.empty:
                print(f"[DEBUG] {node_type} dataframe shape: {df.shape}")
                print(f"[DEBUG] {node_type} columns: {list(df.columns)}")
                print(
                    f"[DEBUG] Looking for column '{self.feat_name_col}' in {node_type}: {self.feat_name_col in df.columns}"
                )

                # If column doesn't exist, show similar columns
                if self.feat_name_col not in df.columns:
                    similar = [
                        col
                        for col in df.columns
                        if "element" in col.lower() or "name" in col.lower()
                    ]
                    print(f"[DEBUG] Similar columns found: {similar}")

        # Also check edges and labels
        print(f"[DEBUG] Edges columns: {list(sample_data['edges_raw'].columns)}")
        if not sample_data["labels_raw"].empty:
            print(f"[DEBUG] Labels columns: {list(sample_data['labels_raw'].columns)}")

        data = HeteroData()
        all_names_map = {}  # Tracks Element_Name -> (type, index)

        # 1. PROCESS NODES
        for node_type in ["beam", "column"]:
            df = sample_data["nodes"][node_type]
            if df.empty:
                continue

            print(f"[DEBUG] Processing {node_type} nodes...")

            try:
                # DEBUG: Check if we can access the column
                print(f"[DEBUG] Trying to access column '{self.feat_name_col}'...")
                test_values = df[self.feat_name_col].values
                print(f"[DEBUG] Success! Got {len(test_values)} values")

                # A. Extract Features (exclude names and type identifiers)
                feat_cols = [
                    c for c in df.columns if c not in [self.feat_name_col, "Ele_Type"]
                ]
                print(f"[DEBUG] Feature columns to use: {len(feat_cols)} columns")
                x_raw = df[feat_cols].values
                print(f"[DEBUG] Raw feature shape: {x_raw.shape}")

                # B. Apply Normalization
                x_norm, self.scalers[node_type] = normalize_features(
                    x_raw, self.scalers[node_type]
                )
                data[node_type].x = torch.from_numpy(x_norm).float()
                print(f"[DEBUG] Normalized feature shape: {data[node_type].x.shape}")

                # C. Attach Labels (Targets for Prediction)
                labels_df = sample_data["labels_raw"]
                if not labels_df.empty:
                    print(f"[DEBUG] Merging labels for {node_type}...")
                    print(
                        f"[DEBUG] Using left_on='{self.feat_name_col}', right_on='{self.labl_name_col}'"
                    )

                    # Check if label column exists
                    if self.labl_name_col not in labels_df.columns:
                        print(
                            f"[ERROR] Label column '{self.labl_name_col}' not found in labels_df!"
                        )
                        print(
                            f"[ERROR] Available label columns: {list(labels_df.columns)}"
                        )

                    merged = df[[self.feat_name_col]].merge(
                        labels_df,
                        left_on=self.feat_name_col,
                        right_on=self.labl_name_col,
                        how="left",
                    )
                    print(f"[DEBUG] Merge result shape: {merged.shape}")

                    y = merged[[self.width_col, self.height_col]].fillna(0).values
                    data[node_type].y = torch.from_numpy(y).float()
                    print(f"[DEBUG] Labels shape: {data[node_type].y.shape}")

                # D. Index Mapping
                print(f"[DEBUG] Creating index mapping for {node_type}...")
                name_values = df[self.feat_name_col].values
                print(
                    f"[DEBUG] First few names: {name_values[:5] if len(name_values) > 5 else name_values}"
                )

                for i, name in enumerate(name_values):
                    all_names_map[name] = (node_type, i)
                print(f"[DEBUG] Added {len(name_values)} entries to mapping")

            except KeyError as e:
                print(f"[ERROR] KeyError for {node_type}: {e}")
                print(f"[ERROR] Available columns: {list(df.columns)}")
                print(f"[ERROR] Looking for: {self.feat_name_col}")
                raise
            except Exception as e:
                print(f"[ERROR] Other error for {node_type}: {e}")
                import traceback

                traceback.print_exc()
                raise

        # 2. PROCESS EDGES (Vectorized approach for speed)
        edges_df = sample_data["edges_raw"]
        edge_bundles = {
            ("beam", "to", "beam"): ([], []),
            ("column", "to", "column"): ([], []),
            ("beam", "to", "column"): ([], []),
            ("column", "to", "beam"): ([], []),
        }

        for _, row in edges_df.iterrows():
            src_name = row[self.source_col]
            dst_name = row[self.target_col]

            if src_name in all_names_map and dst_name in all_names_map:
                src_type, src_idx = all_names_map[src_name]
                dst_type, dst_idx = all_names_map[dst_name]

                bundle_key = (src_type, "to", dst_type)
                edge_bundles[bundle_key][0].append(src_idx)
                edge_bundles[bundle_key][1].append(dst_idx)

        # Assign edge_index to HeteroData object (only if edges exist)
        for (src_t, rel, dst_t), (src_list, dst_list) in edge_bundles.items():
            if src_list and dst_list:  # Check both lists are non-empty
                data[src_t, rel, dst_t].edge_index = torch.tensor(
                    [src_list, dst_list], dtype=torch.long
                )
            else:
                # Create empty edge index for consistency
                data[src_t, rel, dst_t].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )

        # Assign edge_index to HeteroData object
        for (src_t, rel, dst_t), (src_list, dst_list) in edge_bundles.items():
            if src_list:
                data[src_t, rel, dst_t].edge_index = torch.tensor(
                    [src_list, dst_list], dtype=torch.long
                )

        # 3. ADD POSITIONAL ENCODING
        if self.pe_dim > 0:
            data = self._add_laplacian_pe(data)

        return data

    def _add_laplacian_pe(self, data):
        """Adds global structural awareness via Laplacian Eigenvectors."""
        # Skip if no edges exist
        total_edges = sum(data.num_edges_dict.values())
        if total_edges == 0:
            print("[WARNING] Graph has no edges, skipping Laplacian PE")
            return data

        # Convert to homogeneous
        try:
            homo = data.to_homogeneous()

            # Calculate PE
            pe = compute_laplacian_pe(homo.edge_index, homo.num_nodes, self.pe_dim)

            # Distribute the PE back to beams and columns correctly
            start_idx = 0
            for node_type in data.node_types:
                num_nodes = data[node_type].x.size(0)
                if num_nodes > 0:
                    node_pe = pe[start_idx : start_idx + num_nodes]
                    # Concatenate PE to existing normalized features
                    data[node_type].x = torch.cat([data[node_type].x, node_pe], dim=1)
                    start_idx += num_nodes

        except Exception as e:
            print(f"[WARNING] Failed to compute Laplacian PE: {e}. Skipping.")

        return data
