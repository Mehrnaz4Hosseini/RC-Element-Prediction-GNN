"""
Heterogeneous Graph Transformer for RC Element Prediction

Purpose: Predict width and height of structural elements (beams and columns)
using heterogeneous graph neural networks that respect structural engineering semantics.

Why HGT over GAT for this problem:
1. Beams and columns are FUNDAMENTALLY DIFFERENT structural elements
   - Same features have different meanings (e.g., "Length" means span for beams, height for columns)
   - Different physical behaviors (bending vs compression)
   - Different design constraints

2. Edge relationships are NOT uniform
   - Beam-to-beam: lateral load transfer
   - Column-to-column: vertical load path
   - Beam-to-column: gravity load transfer
   - Column-to-beam: support condition

3. Type-specific predictions needed
   - Beam dimensions follow span/load patterns
   - Column dimensions follow axial load/height patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger("HGT")


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer for structural element dimension prediction.

    Architecture tailored for beam-column structural graphs:
    - Type-specific input projections (different feature semantics per type)
    - Multi-relation attention (learns structural relationship patterns)
    - Separate prediction heads (different design rules for beams vs columns)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        node_types: Optional[List[str]] = None,
        output_dim: int = 2,  # Width and Height
        use_structural_encoding: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.use_structural_encoding = use_structural_encoding

        # Node types from your graph: beams and columns
        self.node_types = node_types or ["beam", "column"]

        # These will be initialized on first forward pass
        self.input_projections = nn.ModuleDict()
        self.structural_biases = nn.ParameterDict()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.output_heads = nn.ModuleDict()

        self.dropout = nn.Dropout(dropout)
        self._initialized = False

        logger.info(
            f"HGT initialized: {num_layers} layers, {num_heads} heads, {hidden_channels} hidden"
        )

    def _initialize_from_graph(self, graph):
        """
        Initialize model architecture based on the first graph's structure.
        This makes the model adaptable to different feature dimensions.
        """
        if self._initialized:
            return

        # Extract metadata from graph
        metadata = (graph.node_types, graph.edge_types)
        logger.info(f"Initializing HGT with metadata: {metadata}")

        # 1. TYPE-SPECIFIC INPUT PROJECTIONS
        # WHY: Beam features and column features have DIFFERENT physical meanings
        # Example: "Length" in a beam = span length (horizontal)
        #          "Length" in a column = story height (vertical)
        # They need DIFFERENT transformations to map to a common latent space
        for node_type in self.node_types:
            if hasattr(graph[node_type], "x") and graph[node_type].x is not None:
                in_channels = graph[node_type].x.size(-1)

                # Projection: [features] → [hidden_channels]
                # LayerNorm helps with training stability
                self.input_projections[node_type] = nn.Sequential(
                    Linear(in_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(self.dropout.p),
                )

                logger.info(
                    f"  {node_type} projection: {in_channels} → {hidden_channels}"
                )

        # 2. STRUCTURAL ENCODING (Optional but recommended)
        # WHY: Adds learned bias that beams are typically horizontal, columns vertical
        if self.use_structural_encoding:
            for node_type in self.node_types:
                # Learnable type-specific bias vector
                self.structural_biases[node_type] = nn.Parameter(
                    torch.zeros(1, hidden_channels)
                )
            logger.info("  Structural encoding enabled")

        # 3. HGT CONVOLUTION LAYERS
        # WHY: Each layer performs message passing with RELATION-AWARE attention
        # Layer 1: Local neighborhood aggregation
        # Layer 2: 2-hop structural patterns
        # Layer 3: Global structural context
        for i in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                    dropout=self.dropout.p,
                )
            )
            # LayerNorm for stable training with residual connections
            self.norms.append(nn.LayerNorm(hidden_channels))

        logger.info(f"  {num_layers} HGT layers initialized")

        # 4. TYPE-SPECIFIC PREDICTION HEADS
        # WHY: Beam width/height follows DIFFERENT patterns than column width/height
        # Beams: Wider and shorter (width > height typically)
        # Columns: More square or taller (height ≥ width typically)
        # Each head learns its own structural design rules
        for node_type in self.node_types:
            self.output_heads[node_type] = nn.Sequential(
                # First dense block
                Linear(hidden_channels, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout.p),
                # Second dense block
                Linear(hidden_channels // 2, hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout.p),
                # Output: Width and Height
                Linear(hidden_channels // 4, output_dim),
            )

        logger.info(f"  Output heads initialized for: {self.node_types}")
        self._initialized = True

    def forward(self, graph):
        """
        Forward pass through the HGT.

        Args:
            graph: PyTorch Geometric HeteroData object with:
                - graph['beam'].x: Beam node features [N_beam, F_beam]
                - graph['column'].x: Column node features [N_column, F_column]
                - graph.edge_index_dict: Edge indices for all 4 relation types

        Returns:
            Dictionary of predictions: {'beam': [N_beam, 2], 'column': [N_column, 2]}
            Each [width_prediction, height_prediction]
        """
        # Lazy initialization on first forward pass
        if not self._initialized:
            self._initialize_from_graph(graph)

        # Extract features
        x_dict = {}
        for node_type in self.node_types:
            if hasattr(graph[node_type], "x"):
                x_dict[node_type] = graph[node_type].x

        edge_index_dict = graph.edge_index_dict

        # STEP 1: Type-specific feature projection
        # Transform beam and column features SEPARATELY into a common hidden space
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and node_type in self.input_projections:
                h_dict[node_type] = self.input_projections[node_type](x_dict[node_type])

        # STEP 2: Add structural inductive bias (optional)
        if self.use_structural_encoding:
            for node_type in self.node_types:
                if node_type in h_dict and node_type in self.structural_biases:
                    # Add learned bias that captures structural type (horizontal vs vertical)
                    h_dict[node_type] = (
                        h_dict[node_type] + self.structural_biases[node_type]
                    )

        # STEP 3: Heterogeneous message passing
        # Each HGT layer performs:
        # - Type-specific linear projections for Query, Key, Value
        # - Relation-specific attention (different for beam→beam vs beam→column)
        # - Heterogeneous message aggregation
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Multi-relation message passing
            h_dict_new = conv(h_dict, edge_index_dict)

            # Residual connection + normalization for each node type
            for node_type in h_dict.keys():
                if node_type in h_dict_new:
                    # Skip connection: preserves original features
                    residual = h_dict[node_type]
                    updated = h_dict_new[node_type]

                    # Add & Normalize (Transformer-style)
                    h_dict[node_type] = norm(residual + updated)

                    # Activation and dropout
                    h_dict[node_type] = F.relu(h_dict[node_type])
                    h_dict[node_type] = self.dropout(h_dict[node_type])

        # STEP 4: Type-specific predictions
        predictions = {}
        for node_type in self.node_types:
            if node_type in h_dict and node_type in self.output_heads:
                # Each node type gets its own prediction head
                # This is CRUCIAL because:
                # - Beam dimensions follow span-load relationships
                # - Column dimensions follow axial-load relationships
                predictions[node_type] = self.output_heads[node_type](h_dict[node_type])

        return predictions

    def predict(self, graph) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference.
        Returns predictions detached from computation graph.
        """
        self.eval()
        with torch.no_grad():
            return {k: v.cpu() for k, v in self.forward(graph).items()}

    def get_embeddings(self, graph) -> Dict[str, torch.Tensor]:
        """
        Extract learned node embeddings before the output head.
        Useful for visualization or transfer learning.

        Returns embeddings that capture structural context:
        - Beam embeddings: Encode their role in the structural system
        - Column embeddings: Encode their load-bearing characteristics
        """
        self.eval()
        with torch.no_grad():
            # Forward pass without output heads
            if not self._initialized:
                self._initialize_from_graph(graph)

            x_dict = {
                nt: graph[nt].x for nt in self.node_types if hasattr(graph[nt], "x")
            }
            edge_index_dict = graph.edge_index_dict

            # Project and pass through convolutions only
            h_dict = {}
            for node_type in self.node_types:
                if node_type in x_dict and node_type in self.input_projections:
                    h_dict[node_type] = self.input_projections[node_type](
                        x_dict[node_type]
                    )

            if self.use_structural_encoding:
                for node_type in self.node_types:
                    if node_type in h_dict and node_type in self.structural_biases:
                        h_dict[node_type] = (
                            h_dict[node_type] + self.structural_biases[node_type]
                        )

            for conv, norm in zip(self.convs, self.norms):
                h_dict_new = conv(h_dict, edge_index_dict)
                for node_type in h_dict.keys():
                    if node_type in h_dict_new:
                        h_dict[node_type] = norm(
                            h_dict[node_type] + h_dict_new[node_type]
                        )
                        h_dict[node_type] = F.relu(h_dict[node_type])

            return {k: v.cpu() for k, v in h_dict.items()}

    def analyze_relationships(self, graph) -> Dict[str, str]:
        """
        Explain what structural relationships the model learns.
        This is uniquely possible with HGT (not possible with GAT).
        """
        relationship_meanings = {
            ("beam", "to", "beam"): "Lateral load transfer between beams",
            ("column", "to", "column"): "Vertical load path continuity",
            ("beam", "to", "column"): "Gravity load from beams to columns",
            ("column", "to", "beam"): "Support condition for beams",
        }
        return relationship_meanings
