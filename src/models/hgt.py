"""
Heterogeneous Graph Transformer (HGT) for RC Element Prediction.

Predicts width (b) and height (h) of structural elements (beams & columns)
on heterogeneous graphs that respect structural-engineering semantics.

Why HGT (not a homogeneous GAT):
  1. Beams and columns are fundamentally different elements. The SAME feature
     can mean different things (e.g. "Length" = span for a beam, height for a
     column), so each type needs its own input projection.
  2. The four relations carry different physics:
        beam   -> beam    : lateral load transfer
        column -> column  : vertical load path
        beam   -> column  : gravity load transfer
        column -> beam    : support condition
     HGT learns relation-specific attention for each of these.
  3. Beam vs column dimensions follow different design rules, so each type
     gets its own prediction head.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

from src.models.base import BaseHeteroGNN

logger = logging.getLogger("HGT")


class HGT(BaseHeteroGNN):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        node_types: Optional[List[str]] = None,
        output_dim: int = 2,
        use_structural_encoding: bool = True,
    ):
        super().__init__(node_types=node_types, output_dim=output_dim)

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.use_structural_encoding = use_structural_encoding

        # Built lazily on the first forward pass (once we see graph metadata).
        self.input_projections = nn.ModuleDict()
        self.structural_biases = nn.ParameterDict()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.output_heads = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"HGT created: {num_layers} layers, {num_heads} heads, "
            f"{hidden_channels} hidden"
        )

    # ------------------------------------------------------------------ #
    def _initialize_from_graph(self, graph):
        """Build all submodules once, using the graph's node/edge metadata."""
        if self._initialized:
            return

        metadata = (graph.node_types, graph.edge_types)
        logger.info(f"Initializing HGT with metadata: {metadata}")

        # 1. Type-specific input projections (lazy: in_channels=-1 is inferred).
        for nt in self.node_types:
            self.input_projections[nt] = nn.Sequential(
                Linear(-1, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )

        # 2. Optional learnable structural bias per node type.
        if self.use_structural_encoding:
            for nt in self.node_types:
                self.structural_biases[nt] = nn.Parameter(
                    torch.zeros(1, self.hidden_channels)
                )

        # 3. HGT message-passing layers (FIX: use self.num_layers / self.num_heads).
        for _ in range(self.num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    metadata=metadata,
                    heads=self.num_heads,
                )
            )
            self.norms.append(nn.LayerNorm(self.hidden_channels))

        # 4. Type-specific prediction heads -> [width, height].
        for nt in self.node_types:
            self.output_heads[nt] = nn.Sequential(
                Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.LayerNorm(self.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                Linear(self.hidden_channels // 2, self.hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                Linear(self.hidden_channels // 4, self.output_dim),
            )

        # Move freshly-created modules onto the graph's device.
        self.to(graph["beam"].x.device if "beam" in graph.node_types else "cpu")
        self._initialized = True
        logger.info("HGT submodules initialized")

    # ------------------------------------------------------------------ #
    def _encode(self, graph) -> Dict[str, torch.Tensor]:
        """Shared encoder: projections -> structural bias -> HGT layers."""
        if not self._initialized:
            self._initialize_from_graph(graph)

        # Project each node type into the common hidden space.
        h_dict = {}
        for nt in self.node_types:
            if nt in graph.node_types and hasattr(graph[nt], "x"):
                h_dict[nt] = self.input_projections[nt](graph[nt].x)

        # Add structural inductive bias.
        if self.use_structural_encoding:
            for nt in h_dict:
                if nt in self.structural_biases:
                    h_dict[nt] = h_dict[nt] + self.structural_biases[nt]

        # Relation-aware message passing with residual + norm.
        edge_index_dict = graph.edge_index_dict
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h_dict, edge_index_dict)
            for nt in list(h_dict.keys()):
                if nt in out:
                    h_dict[nt] = norm(h_dict[nt] + out[nt])
                    h_dict[nt] = self.dropout(F.relu(h_dict[nt]))
        return h_dict

    # ------------------------------------------------------------------ #
    def forward(self, graph) -> Dict[str, torch.Tensor]:
        h_dict = self._encode(graph)
        preds = {}
        for nt in self.node_types:
            if nt in h_dict:
                preds[nt] = self.output_heads[nt](h_dict[nt])
        return preds

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_embeddings(self, graph) -> Dict[str, torch.Tensor]:
        """Node embeddings before the prediction head (for viz / transfer)."""
        self.eval()
        return {k: v.cpu() for k, v in self._encode(graph).items()}

    # ------------------------------------------------------------------ #
    def reset_all_parameters(self):
        super().reset_all_parameters()
        # structural biases need explicit re-zeroing for fresh k-fold starts.
        for nt in self.structural_biases:
            nn.init.zeros_(self.structural_biases[nt])

    # ------------------------------------------------------------------ #
    @staticmethod
    def analyze_relationships() -> Dict[tuple, str]:
        """Human-readable meaning of each learned relation (HGT-specific)."""
        return {
            ("beam", "to", "beam"): "Lateral load transfer between beams",
            ("column", "to", "column"): "Vertical load path continuity",
            ("beam", "to", "column"): "Gravity load from beams to columns",
            ("column", "to", "beam"): "Support condition for beams",
        }
