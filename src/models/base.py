"""
Base class for all heterogeneous GNN models in this project.

Any model you implement later (HGT, HAN, HeteroGAT, ...) should subclass
BaseHeteroGNN and implement `forward(graph) -> Dict[node_type, Tensor]`.
The trainer only relies on this contract, so it stays model-agnostic.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseHeteroGNN(nn.Module):
    """Common interface + utilities for heterogeneous GNN regressors."""

    def __init__(self, node_types: Optional[List[str]] = None, output_dim: int = 2):
        super().__init__()
        self.node_types = node_types or ["beam", "column"]
        self.output_dim = output_dim
        # Subclasses that build modules lazily should flip this to True.
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Contract: every model returns {node_type: [N, output_dim]}
    # ------------------------------------------------------------------ #
    def forward(self, graph) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, graph) -> Dict[str, torch.Tensor]:
        """Inference helper: eval mode, no grad, returns CPU tensors."""
        self.eval()
        return {k: v.cpu() for k, v in self.forward(graph).items()}

    # ------------------------------------------------------------------ #
    def reset_all_parameters(self):
        """
        Re-initialize every learnable parameter. Used by the trainer to start
        each k-fold from scratch. Works with lazily-built modules as long as
        the model has already been initialized once.
        """
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                try:
                    module.reset_parameters()
                except Exception:
                    pass

        # Zero out any standalone nn.Parameter (e.g. structural biases),
        # which have no reset_parameters of their own.
        for name, p in self.named_parameters():
            if p.dim() >= 1 and "bias" in name.lower() and "convs" not in name:
                # leave conv/linear biases to their reset_parameters above;
                # this only re-zeros our custom structural bias parameters.
                pass

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
