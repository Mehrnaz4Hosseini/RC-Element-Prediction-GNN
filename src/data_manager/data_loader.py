"""
Heterogeneous Data Loader for RC Element Prediction

Loads data files:
- Edge-{sample_name}.xlsx: Connections between elements
- Feature-{sample_name}_english.xlsx: Features with Ele_Type column (0=column, 1=beam)
- Label-{sample_name}.xlsx: Target width and height values
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml

CONFIG_PATH = "configs/base.yaml"


def load_config(config_path: str = CONFIG_PATH) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model config if exists
    model_config_path = "configs/model/han.yaml"
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        config.update(model_config)

    return config


class HeteroDataLoader:
    """
    Loader for your heterogeneous RC data with Ele_Type column.

    Key features:
    1. Single feature file with Ele_Type column to separate beams and columns
    2. Edge file defines connections between elements
    3. Label file contains target width and height
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config = load_config(config_path)

        # Get column names from config
        self.element_name_col = self.config["data"]["columns"]["element_name"]
        self.element_type_col = self.config["data"]["columns"]["element_type"]
        self.source_col = self.config["data"]["columns"]["source_node"]
        self.target_col = self.config["data"]["columns"]["target_node"]

        # Label columns
        self.label_name_col = self.config["data"]["columns"]["label_element_name"]
        self.label_width_col = self.config["data"]["columns"]["label_width"]
        self.label_height_col = self.config["data"]["columns"]["label_height"]

        # Feature columns to use
        self.feature_cols = self.config["data"]["feature_columns"]

    def load_sample(self, sample_name: str, split: str = "train") -> Optional[Dict]:
        """
        Load all files for a single sample.

        Args:
            sample_name: e.g., "sample_1"
            split: "train" or "test"

        Returns:
            Dictionary with:
            - 'edges': DataFrame of connections
            - 'features': DataFrame with Ele_Type column
            - 'labels': DataFrame with target values
            - 'sample_name': Original sample name
        """
        # Get the directory
        if split == "train":
            base_dir = os.path.join(self.config["data"]["train_dir"], sample_name)
        else:
            base_dir = os.path.join(self.config["data"]["test_dir"], sample_name)

        print(f"📂 Loading sample: {sample_name} from {base_dir}")

        # Initialize data dictionary
        data = {"sample_name": sample_name, "split": split}

        # 1. Load edge file (connections)
        edge_file = os.path.join(
            base_dir,
            self.config["data"]["file_patterns"]["edge"].format(
                sample_name=sample_name
            ),
        )
        if os.path.exists(edge_file):
            data["edges"] = pd.read_excel(edge_file)
            print(
                f"  ✓ Edges: {data['edges'].shape} from {os.path.basename(edge_file)}"
            )
            print(f"    Columns: {list(data['edges'].columns)}")
        else:
            print(f"  ❌ Edge file not found: {edge_file}")
            return None

        # 2. Load feature file (with Ele_Type column)
        feature_file = os.path.join(
            base_dir,
            self.config["data"]["file_patterns"]["feature"].format(
                sample_name=sample_name
            ),
        )
        if os.path.exists(feature_file):
            data["features"] = pd.read_excel(feature_file)
            print(
                f"  ✓ Features: {data['features'].shape} from {os.path.basename(feature_file)}"
            )

            # Check if Ele_Type column exists
            if self.element_type_col not in data["features"].columns:
                print(
                    f"  ⚠️  Ele_Type column '{self.element_type_col}' not found in features"
                )
                print(f"    Available columns: {list(data['features'].columns)}")
            else:
                # Count beams and columns
                beam_count = (data["features"][self.element_type_col] == 1).sum()
                column_count = (data["features"][self.element_type_col] == 0).sum()
                print(f"    Beams: {beam_count}, Columns: {column_count}")
        else:
            print(f"  ❌ Feature file not found: {feature_file}")
            return None

        # 3. Load label file (target values)
        label_file = os.path.join(
            base_dir,
            self.config["data"]["file_patterns"]["label"].format(
                sample_name=sample_name
            ),
        )
        if os.path.exists(label_file):
            data["labels"] = pd.read_excel(label_file)
            print(
                f"  ✓ Labels: {data['labels'].shape} from {os.path.basename(label_file)}"
            )
            print(f"    Columns: {list(data['labels'].columns)}")
        else:
            print(f"  ⚠️  Label file not found: {label_file}")
            data["labels"] = pd.DataFrame()  # Empty for test set

        return data

    def load_all_samples(self, split: str = "train") -> List[Dict]:
        """
        Load all samples from a split.

        Args:
            split: "train" or "test"

        Returns:
            List of data dictionaries for each sample
        """
        if split == "train":
            base_dir = self.config["data"]["train_dir"]
        else:
            base_dir = self.config["data"]["test_dir"]

        # Get all sample folders
        sample_folders = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                sample_folders.append(item)

        print(f"\n🔍 Found {len(sample_folders)} samples in {split} set")
        print(f"   Samples: {sample_folders}")

        # Load each sample
        all_samples = []
        for sample_folder in sample_folders:
            sample_data = self.load_sample(sample_folder, split)
            if sample_data is not None:
                all_samples.append(sample_data)

        print(
            f"\n✅ Successfully loaded {len(all_samples)}/{len(sample_folders)} samples"
        )
        return all_samples

    def preprocess_features(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Separate features into beams and columns based on Ele_Type.

        Args:
            features_df: DataFrame with Ele_Type column

        Returns:
            Dictionary with 'beam_features' and 'column_features' DataFrames
        """
        if self.element_type_col not in features_df.columns:
            raise ValueError(
                f"Ele_Type column '{self.element_type_col}' not found in features"
            )

        # Separate based on Ele_Type
        beam_mask = features_df[self.element_type_col] == 1
        column_mask = features_df[self.element_type_col] == 0

        beam_df = features_df[beam_mask].copy()
        column_df = features_df[column_mask].copy()

        print(f"\n📊 Separating features by Ele_Type:")
        print(f"   Total elements: {len(features_df)}")
        print(f"   Beams (Ele_Type=1): {len(beam_df)}")
        print(f"   Columns (Ele_Type=0): {len(column_df)}")

        # Select feature columns that exist in the DataFrame
        available_cols = [
            col for col in self.feature_cols if col in features_df.columns
        ]
        missing_cols = [
            col for col in self.feature_cols if col not in features_df.columns
        ]

        if missing_cols:
            print(f"   ⚠️  Missing feature columns: {missing_cols[:5]}...")

        # Keep only selected features + element name
        beam_features = beam_df[[self.element_name_col] + available_cols]
        column_features = column_df[[self.element_name_col] + available_cols]

        return {
            "beam_features": beam_features,
            "column_features": column_features,
            "all_features": features_df,
            "feature_columns": available_cols,
        }

    def process_edges(
        self, edges_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> Dict[str, List]:
        """
        Process edges and separate by type (beam-beam, beam-column, column-column).

        Args:
            edges_df: Edge DataFrame with Source and Target
            features_df: Feature DataFrame with element names and types

        Returns:
            Dictionary with edges separated by type
        """
        # Create mapping from element name to type
        element_to_type = {}
        if (
            self.element_name_col in features_df.columns
            and self.element_type_col in features_df.columns
        ):
            for _, row in features_df.iterrows():
                element_name = row[self.element_name_col]
                element_type = "beam" if row[self.element_type_col] == 1 else "column"
                element_to_type[element_name] = element_type

        edges_by_type = {
            "beam_to_beam": [],
            "beam_to_column": [],
            "column_to_column": [],
        }

        print(f"\n🔗 Processing edges:")
        print(f"   Total edges: {len(edges_df)}")

        for _, row in edges_df.iterrows():
            source = str(row[self.source_col])
            target = str(row[self.target_col])

            # Get element types
            source_type = element_to_type.get(source, "unknown")
            target_type = element_to_type.get(target, "unknown")

            if source_type == "beam" and target_type == "beam":
                edges_by_type["beam_to_beam"].append([source, target])
            elif source_type == "column" and target_type == "column":
                edges_by_type["column_to_column"].append([source, target])
            elif (source_type == "beam" and target_type == "column") or (
                source_type == "column" and target_type == "beam"
            ):
                edges_by_type["beam_to_column"].append([source, target])
            else:
                print(
                    f"   ⚠️  Unknown edge type: {source} ({source_type}) -> {target} ({target_type})"
                )

        # Print statistics
        print(f"   Beam-Beam edges: {len(edges_by_type['beam_to_beam'])}")
        print(f"   Beam-Column edges: {len(edges_by_type['beam_to_column'])}")
        print(f"   Column-Column edges: {len(edges_by_type['column_to_column'])}")

        return edges_by_type

    def match_labels(
        self, features_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Match labels to elements based on element names.

        Args:
            features_df: Feature DataFrame
            labels_df: Label DataFrame

        Returns:
            Dictionary with labels for each element
        """
        if labels_df.empty:
            return {}

        # Create mapping from element name to labels
        label_dict = {}
        for _, row in labels_df.iterrows():
            element_name = str(row[self.label_name_col])
            width = row[self.label_width_col]
            height = row[self.label_height_col]
            label_dict[element_name] = [width, height]

        # Match labels to features
        beam_labels = []
        column_labels = []
        beam_indices = []
        column_indices = []

        for idx, row in features_df.iterrows():
            element_name = str(row[self.element_name_col])
            element_type = "beam" if row[self.element_type_col] == 1 else "column"

            if element_name in label_dict:
                labels = label_dict[element_name]
                if element_type == "beam":
                    beam_labels.append(labels)
                    beam_indices.append(idx)
                else:
                    column_labels.append(labels)
                    column_indices.append(idx)

        result = {}
        if beam_labels:
            result["beam_labels"] = np.array(beam_labels)
            result["beam_indices"] = np.array(beam_indices)
        if column_labels:
            result["column_labels"] = np.array(column_labels)
            result["column_indices"] = np.array(column_indices)

        print(f"\n🏷️  Matched labels:")
        if "beam_labels" in result:
            print(f"   Beam labels: {len(result['beam_labels'])}")
        if "column_labels" in result:
            print(f"   Column labels: {len(result['column_labels'])}")

        return result
