"""
data_loader.py

This module contains functions to load graph-related data from Excel files.
It expects files named in the format:
- Edge-{graph_name}.xlsx
- Feature-{graph_name}.xlsx
- Label-{graph_name}.xlsx

Functions:
- load_excel_files: Reads edge, feature, and label data for a given graph.
"""

import os
import pandas as pd

def load_excel_files(name, root_dir):
    edge_file = os.path.join(root_dir, f"Edge-{name}.xlsx")
    feature_file = os.path.join(root_dir, f"Feature-{name}.xlsx")
    label_file = os.path.join(root_dir, f"Label-{name}.xlsx")

    edge_df = pd.read_excel(edge_file)
    feature_df = pd.read_excel(feature_file)
    label_df = pd.read_excel(label_file)

    return edge_df, feature_df, label_df