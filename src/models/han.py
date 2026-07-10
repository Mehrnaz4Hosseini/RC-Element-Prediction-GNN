"""
Heterogeneous Graph Attention Network (HAN) for RC Element Prediction.

Predicts width (b) and height (h) of structural elements (beams & columns)
on the same heterogeneous graphs used by the HGT model, so the two are
directly comparable (identical feature pipeline, PE, isolated-node handling,
trainer, loss and metrics — only the message-passing operator differs).

HAN (Wang et al., WWW 2019) — two levels of attention:
  1. Node-level attention: within each metapath, a GAT-style multi-head
     attention weighs a node's metapath-based neighbours and produces a
     metapath-specific embedding.
  2. Semantic-level attention: a learnable, shared attention vector scores how
     important each metapath is, then combines the metapath-specific
     embeddings into one vector per node.

`torch_geometric.nn.HANConv` implements exactly this, treating EACH edge type
(relation) as a length-1 metapath. So for our four structural relations
    beam   -> beam    (lateral load transfer)
    column -> column  (vertical load path)
    beam   -> column  (gravity load transfer)
    column -> beam    (support condition)
HAN first attends over neighbours within each relation (node-level), then
learns a semantic weight per relation and fuses them. This is conceptually
different from HGT, which learns a full per-relation transformer — comparing
the two is the point of this experiment.

Design mirrors src/models/hgt.py on purpose (same constructor signature, lazy
type-specific input projections, optional structural bias, N attention layers
with residual + norm, type-specific output heads) so config/wiring/metrics are
uniform. The only substantive difference is HGTConv -> HANConv plus a couple of
HAN-specific safeguards documented inline.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear

from src.models.base import BaseHeteroGNN

logger = logging.getLogger("HAN")


class HAN(BaseHeteroGNN):
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
            f"HAN created: {num_layers} layers, {num_heads} heads, "
            f"{hidden_channels} hidden"
        )

    # ------------------------------------------------------------------ #
    def _initialize_from_graph(self, graph):
        """Build all submodules once, using the graph's node/edge metadata."""
        if self._initialized:
            return

        metadata = (graph.node_types, graph.edge_types)
        logger.info(f"Initializing HAN with metadata: {metadata}")

        # 1. Type-specific input projections (lazy: in_channels=-1 is inferred).
        #    Beams and columns carry the same feature *columns* but different
        #    physics, so each type gets its own projection into the shared space.
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

        # 3. HAN message-passing layers. HANConv does node-level attention within
        #    each relation (metapath) + semantic attention across relations.
        #    in/out channels are the shared hidden size because we already
        #    projected every node type into it above.
        for _ in range(self.num_layers):
            self.convs.append(
                HANConv(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    metadata=metadata,
                    heads=self.num_heads,
                    dropout=self.dropout_p,
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
        logger.info("HAN submodules initialized")

    # ------------------------------------------------------------------ #
    def _encode(self, graph) -> Dict[str, torch.Tensor]:
        """Shared encoder: projections -> structural bias -> HAN layers."""
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

        # Two-level (node + semantic) attention message passing.
        edge_index_dict = graph.edge_index_dict
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h_dict, edge_index_dict)
            for nt in list(h_dict.keys()):
                # HAN-specific: HANConv returns None (not a missing key) for a
                # node type that received no metapath messages in THIS graph
                # (e.g. every node of that type is isolated under
                # isolated.strategy="none"). HGT's "if nt in out" is NOT enough
                # here — we must check for None explicitly and, in that case,
                # carry the node's current representation forward unchanged.
                out_nt = out.get(nt)
                if out_nt is None:
                    continue
                h_dict[nt] = norm(h_dict[nt] + out_nt)      # residual keeps self-info
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
        """Human-readable meaning of each metapath (relation) HAN attends over."""
        return {
            ("beam", "to", "beam"): "Lateral load transfer between beams",
            ("column", "to", "column"): "Vertical load path continuity",
            ("beam", "to", "column"): "Gravity load from beams to columns",
            ("column", "to", "beam"): "Support condition for beams",
        }
