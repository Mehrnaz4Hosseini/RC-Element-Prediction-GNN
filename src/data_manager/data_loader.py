"""
Heterogeneous Data Loader for RC Element Prediction

Loads data files:
- Edge-{sample_name}.xlsx: Connections between elements
- Feature-{sample_name}_english.xlsx: Features with Ele_Type column (0=column, 1=beam)
- Label-{sample_name}.xlsx: Target width and height values
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml


class HeteroDataLoader:
    def __init__(self, config_path: str = "configs/base.yaml"):
        # Load configuration
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Config loading failed: {e}. Using defaults.")
            self.config = {}

        # Mappings pulled directly from the updated YAML
        cols = self.config.get("data", {}).get("columns", {})

        self.element_name_col = cols.get("element_name", "Element_Name")
        self.element_type_col = cols.get("element_type", "Ele_Type")
        self.source_col = cols.get("source_node", "Source")
        self.target_col = cols.get("target_node", "Target")
        self.label_width_col = cols.get("label_width", "Width")
        self.label_height_col = cols.get("label_height", "Height")

        # Prefixes for fuzzy search
        patterns = self.config.get("data", {}).get("file_patterns", {})
        self.edge_prefix = patterns.get("edge_prefix", "Edge")
        self.feat_suffix = patterns.get("feature_suffix", "*english.xlsx")
        self.label_prefix = patterns.get("label_prefix", "Label")

    def _find_fuzzy(self, folder_path: str, prefix: str) -> Optional[str]:
        if not os.path.exists(folder_path):
            return None

        if prefix == "Feature":
            # Uses the suffix from YAML
            pattern = os.path.join(folder_path, f"Feature-{self.feat_suffix}")
        else:
            pattern = os.path.join(folder_path, f"{prefix}-*.xlsx")

        files = glob.glob(pattern)
        return files[0] if files else None

    def load_sample(self, sample_name: str, split: str = "train") -> Optional[Dict]:
        """Loads and separates a single sample into hetero-ready dictionaries."""

        # Determine base directory (checks if running from /notebooks or project root)
        if os.path.exists(os.path.join("data", split)):
            base_dir = os.path.join("data", split, sample_name)
        else:
            base_dir = os.path.join("..", "data", split, sample_name)

        # 1. Find Files
        edge_path = self._find_fuzzy(base_dir, "Edge")
        feat_path = self._find_fuzzy(base_dir, "Feature")
        labl_path = self._find_fuzzy(base_dir, "Label")

        if not edge_path or not feat_path:
            print(
                f"❌ Missing files in {sample_name} (Edge: {bool(edge_path)}, Feat: {bool(feat_path)})"
            )
            return None

        # 2. Load DataFrames
        try:
            edges_df = pd.read_excel(edge_path)
            features_df = pd.read_excel(feat_path)
            labels_df = pd.read_excel(labl_path) if labl_path else pd.DataFrame()

            # 3. Categorize nodes by Ele_Type (0=Column, 1=Beam)
            # This is the core requirement for a Heterogeneous Graph
            beam_mask = features_df[self.element_type_col] == 1
            column_mask = features_df[self.element_type_col] == 0

            return {
                "sample_name": sample_name,
                "split": split,
                "nodes": {
                    "beam": features_df[beam_mask],
                    "column": features_df[column_mask],
                },
                "edges_raw": edges_df,
                "labels_raw": labels_df,
            }
        except Exception as e:
            print(f"❌ Error reading Excel in {sample_name}: {e}")
            return None

    def load_all_samples(self, split: str = "train") -> List[Dict]:
        """Finds all sample folders and loads them."""
        # Check path relative to execution point
        search_path = "data" if os.path.exists("data") else "../data"
        split_path = os.path.join(search_path, split)

        if not os.path.exists(split_path):
            print(f"❌ Directory not found: {split_path}")
            return []

        sample_folders = sorted(
            [
                d
                for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ]
        )

        all_data = []
        for folder in sample_folders:
            sample_data = self.load_sample(folder, split)
            if sample_data:
                all_data.append(sample_data)

        return all_data
