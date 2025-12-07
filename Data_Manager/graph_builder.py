"""
graph_builder.py

This module defines the GraphBuilder class that coordinates loading, preprocessing,
encoding, and graph construction using PyTorch Geometric's `Data` objects.

Class:
- GraphBuilder: Manages loading, normalization, positional encoding, and graph creation.

"""

import numpy
import torch
from torch_geometric.data import Data
from .data_loader import load_excel_files
from .data_process import normalize_features, to_tensor, compute_laplacian_pe
from sklearn.preprocessing import StandardScaler
import numpy as np

class GraphBuilder:
    """
    Constructs PyTorch Geometric graph objects from Excel input files.

    Args:
        graph_names (list): List of graph name identifiers.
        root_dir (str): Directory where Excel files are stored.
        pe_dim (int): Number of dimensions for positional encoding.
    """
    def __init__(self, graph_names, root_dir, pe_dim):
        """
        Converts dataframes and arrays to PyTorch tensors.

        Returns:
            tuple: edge_index, x, y
        """
        self.graph_names = graph_names
        self.root_dir = root_dir
        self.pe_dim = pe_dim
        self.scaler = StandardScaler()

    def build_graph(self, edge_index, x, y, pe):
        """
        Combines feature and positional encodings and returns a Data object.

        Returns:
            torch_geometric.data.Data: A graph data object.
        """ 
        x = torch.cat([x, pe], dim=1)
        return Data(x=x, edge_index=edge_index, y=y)
    
    def get_one_graph(self):
        name = self.graph_names
        edge_df, feature_df, label_df = load_excel_files(name, self.root_dir)
        x_np = normalize_features(feature_df.iloc[:, 1:].values, self.scaler)
        y_np = label_df.iloc[:, 1:].values
        edge_index, x, y = to_tensor(edge_df, x_np, y_np)
        pe = compute_laplacian_pe(edge_index, x.size(0), self.pe_dim)
        graph = self.build_graph(edge_index, x, y, pe)
        return  graph , feature_df

    def get_all_graphs(self):
        """
        Processes all graphs in the list and returns them as Data objects.

        Returns:
            list: List of torch_geometric.data.Data objects.
        """
        graphs = []
        for name in self.graph_names:
            edge_df, feature_df, label_df = load_excel_files(name, self.root_dir)
            x_np = normalize_features(feature_df.iloc[:, 1:].values, self.scaler)
            y_np = label_df.iloc[:, 1:].values
            edge_index, x, y = to_tensor(edge_df, x_np, y_np)
            pe = compute_laplacian_pe(edge_index, x.size(0), self.pe_dim)
            graph = self.build_graph(edge_index, x, y, pe)
            graphs.append(graph)
        return graphs
    
    
