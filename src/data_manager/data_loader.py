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
from typing import Dict, List, Optional, Tuple
import yaml
from datetime import datetime

# Import logger
try:
    from src.utils.logger import get_data_logger, get_system_logger
except ImportError:
    # Fallback if logger not available
    import logging

    get_data_logger = lambda: logging.getLogger("DATA")
    get_system_logger = lambda: logging.getLogger("SYSTEM")


class HeteroDataLoader:
    """Loads and processes heterogeneous structural data for graph construction."""

    def __init__(self, config_path: str = "configs/base.yaml"):
        """
        Initialize Heterogeneous Data Loader.

        Args:
            config_path: Path to configuration YAML file
        """
        self.logger = get_data_logger()
        self.sys_logger = get_system_logger()

        self.logger.info(f"Initializing HeteroDataLoader with config: {config_path}")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize mappings from config
        self._init_mappings()

        # Performance tracking
        self.stats = {
            "samples_loaded": 0,
            "samples_failed": 0,
            "total_files": 0,
            "start_time": datetime.now(),
        }

        self.logger.info("DataLoader initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            self.logger.debug(f"Configuration loaded from {config_path}")
            return config

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in config file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

    def _init_mappings(self):
        """Initialize column mappings from configuration."""
        data_config = self.config.get("data", {})
        cols = data_config.get("columns", {})
        patterns = data_config.get("file_patterns", {})

        # Column mappings
        self.element_type_col = cols.get("element_type", "Ele_Type")
        self.source_col = cols.get("source_node", "Source")
        self.target_col = cols.get("target_node", "Target")
        self.label_width_col = cols.get("label_width", "Width")
        self.label_height_col = cols.get("label_height", "Height")
        self.row_index_col = "row_index"

        # File patterns
        self.edge_prefix = patterns.get("edge_prefix", "Edge")
        self.feat_suffix = patterns.get("feature_suffix", "*english.xlsx")
        self.label_prefix = patterns.get("label_prefix", "Label")

        self.logger.debug(
            f"Column mappings initialized: {self.element_type_col}, {self.source_col}, {self.target_col}"
        )

    def _find_file(self, folder_path: str, prefix: str) -> Optional[str]:
        """
        Find file using fuzzy matching with glob patterns.

        Args:
            folder_path: Directory to search
            prefix: File prefix (Edge, Feature, Label)

        Returns:
            Path to found file or None
        """
        if not os.path.exists(folder_path):
            self.logger.warning(f"Directory does not exist: {folder_path}")
            return None

        try:
            if prefix == "Feature":
                pattern = os.path.join(folder_path, f"Feature-{self.feat_suffix}")
            else:
                pattern = os.path.join(folder_path, f"{prefix}-*.xlsx")

            files = glob.glob(pattern)

            if not files:
                self.logger.debug(f"No files found for pattern: {pattern}")
                return None

            if len(files) > 1:
                self.logger.warning(
                    f"Multiple files found for pattern {pattern}, using first: {files[0]}"
                )

            return files[0]

        except Exception as e:
            self.logger.error(f"Error searching for {prefix} files: {e}")
            return None

    def _validate_sample_files(
        self, sample_name: str, files: Dict[str, Optional[str]]
    ) -> Tuple[bool, str]:
        """
        Validate that required files exist for a sample.

        Args:
            sample_name: Name of the sample
            files: Dictionary with file paths

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing = []

        if not files.get("edge"):
            missing.append("Edge file")
        if not files.get("feature"):
            missing.append("Feature file")

        if missing:
            error_msg = f"Missing files: {', '.join(missing)}"
            self.logger.error(f"Sample {sample_name}: {error_msg}")
            return False, error_msg

        return True, "All files present"

    def load_sample(self, sample_name: str, split: str = "train") -> Optional[Dict]:
        """
        Load a single sample into hetero-ready dictionaries.

        Args:
            sample_name: Name of the sample folder
            split: Data split (train/test)

        Returns:
            Dictionary with sample data or None if failed
        """
        self.logger.info(f"Loading sample: {sample_name} ({split})")

        # Determine base directory
        base_dir = self._get_base_dir(sample_name, split)
        if not base_dir:
            return None

        # Find files
        files = {
            "edge": self._find_file(base_dir, "Edge"),
            "feature": self._find_file(base_dir, "Feature"),
            "label": self._find_file(base_dir, "Label"),
        }

        # Validate files
        is_valid, error_msg = self._validate_sample_files(sample_name, files)
        if not is_valid:
            self.stats["samples_failed"] += 1
            return None

        try:
            # Load dataframes
            data = self._load_dataframes(files, sample_name)
            if data is None:
                return None

            # Categorize nodes by Ele_Type
            nodes = self._categorize_nodes(data["features"])

            result = {
                "sample_name": sample_name,
                "split": split,
                "nodes": nodes,
                "edges_raw": data["edges"],
                "labels_raw": data["labels"],
                "features_raw": data["features"],
            }

            self.stats["samples_loaded"] += 1
            self.logger.info(f"Successfully loaded sample {sample_name}")
            self.logger.debug(
                f"Sample stats: {len(nodes['beam'])} beams, {len(nodes['column'])} columns"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing sample {sample_name}: {e}")
            self.stats["samples_failed"] += 1
            return None

    def _get_base_dir(self, sample_name: str, split: str) -> Optional[str]:
        """Get base directory for sample, checking multiple possible locations."""
        possible_paths = [
            os.path.join("data", split, sample_name),
            os.path.join("..", "data", split, sample_name),
            os.path.join("../data", split, sample_name),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                self.logger.debug(f"Found sample directory: {path}")
                return path

        self.logger.error(
            f"Sample directory not found for {sample_name}. Checked: {possible_paths}"
        )
        return None

    def _load_dataframes(
        self, files: Dict[str, str], sample_name: str
    ) -> Optional[Dict]:
        """Load data from Excel files."""
        try:
            edges_df = pd.read_excel(files["edge"])
            features_df = pd.read_excel(files["feature"])

            # Add row index
            features_df[self.row_index_col] = features_df.index

            # Load labels if available
            if files["label"]:
                labels_df = pd.read_excel(files["label"])
                labels_df[self.row_index_col] = labels_df.index
            else:
                labels_df = pd.DataFrame()
                self.logger.warning(f"No label file found for {sample_name}")

            self.logger.debug(
                f"Data loaded - Edges: {edges_df.shape}, Features: {features_df.shape}, "
                f"Labels: {labels_df.shape if not labels_df.empty else 'Empty'}"
            )

            return {"edges": edges_df, "features": features_df, "labels": labels_df}

        except Exception as e:
            self.logger.error(f"Error reading Excel files for {sample_name}: {e}")
            return None

    def _categorize_nodes(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Categorize nodes into beams and columns based on Ele_Type."""
        try:
            beam_mask = features_df[self.element_type_col] == 1
            column_mask = features_df[self.element_type_col] == 0

            beams = features_df[beam_mask].copy()
            columns = features_df[column_mask].copy()

            self.logger.debug(
                f"Categorized nodes: {len(beams)} beams, {len(columns)} columns"
            )

            return {"beam": beams, "column": columns}

        except KeyError as e:
            self.logger.error(f"Missing column in features: {e}")
            raise

    def load_all_samples(self, split: str = "train") -> List[Dict]:
        """
        Load all samples from a split directory.

        Args:
            split: Data split to load (train/test)

        Returns:
            List of sample dictionaries
        """
        self.logger.info(f"Loading all samples from split: {split}")

        # Find split directory
        search_paths = ["data", "../data"]
        split_path = None

        for base_path in search_paths:
            path = os.path.join(base_path, split)
            if os.path.exists(path):
                split_path = path
                break

        if not split_path:
            self.logger.error(f"Split directory not found: {split}")
            return []

        # Get sample folders
        try:
            sample_folders = sorted(
                [
                    d
                    for d in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, d))
                ]
            )
        except Exception as e:
            self.logger.error(f"Error listing sample folders: {e}")
            return []

        self.logger.info(f"Found {len(sample_folders)} sample folders")

        # Load samples
        all_data = []
        for folder in sample_folders:
            sample_data = self.load_sample(folder, split)
            if sample_data:
                all_data.append(sample_data)

        # Log statistics
        success_rate = (
            len(all_data) / len(sample_folders) * 100 if sample_folders else 0
        )
        self.logger.info(
            f"Loading complete: {len(all_data)}/{len(sample_folders)} samples loaded "
            f"({success_rate:.1f}% success rate)"
        )

        if self.stats["samples_failed"] > 0:
            self.logger.warning(
                f"{self.stats['samples_failed']} samples failed to load"
            )

        return all_data

    def get_statistics(self) -> Dict[str, any]:
        """
        Get loading statistics.

        Returns:
            Dictionary with loading statistics
        """
        elapsed = datetime.now() - self.stats["start_time"]

        return {
            "samples_loaded": self.stats["samples_loaded"],
            "samples_failed": self.stats["samples_failed"],
            "total_samples": self.stats["samples_loaded"]
            + self.stats["samples_failed"],
            "success_rate": (
                self.stats["samples_loaded"]
                / (self.stats["samples_loaded"] + self.stats["samples_failed"])
                * 100
                if (self.stats["samples_loaded"] + self.stats["samples_failed"]) > 0
                else 0
            ),
            "elapsed_time": str(elapsed),
            "average_time_per_sample": (
                elapsed.total_seconds() / self.stats["samples_loaded"]
                if self.stats["samples_loaded"] > 0
                else 0
            ),
        }
