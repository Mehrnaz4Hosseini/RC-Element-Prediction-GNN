"""
Loss functions for heterogeneous element-prediction models.

StructuralElementLoss is the default, but the trainer accepts any criterion
with the signature:  (predictions_dict, targets_dict) -> (loss_tensor, components_dict)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class StructuralElementLoss(nn.Module):
    """
    Weighted MSE over node types and target dimensions, with an optional
    soft physical constraint (beams usually have width >= height).
    """

    def __init__(
        self,
        width_weight: float = 1.0,
        height_weight: float = 1.0,
        beam_weight: float = 1.0,
        column_weight: float = 1.0,
        use_physical_constraints: bool = False,
        loss_type: str = "huber",      # "huber" | "mse" | "mae"
        huber_delta: float = 1.0,      # in SCALED target units (~1 std)
    ):
        super().__init__()
        self.register_buffer(
            "target_weights", torch.tensor([width_weight, height_weight])
        )
        self.type_weights = {"beam": beam_weight, "column": column_weight}
        self.use_physical_constraints = use_physical_constraints
        assert loss_type in {"huber", "mse", "mae"}
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def _elementwise(self, pred, true):
        """Per-element loss in scaled space, weighted by [width, height]."""
        w = self.target_weights.to(pred.device)
        if self.loss_type == "mse":
            base = ((pred - true) * w) ** 2
        elif self.loss_type == "mae":
            base = ((pred - true) * w).abs()
        else:  # huber: quadratic near 0, linear past delta -> robust to noise
            base = nn.functional.huber_loss(
                pred * w, true * w, reduction="none", delta=self.huber_delta
            )
        return base.mean()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = None  # Bug #4 fix: accumulate as a tensor, never float 0.0
        components: Dict[str, float] = {}

        for nt, pred in predictions.items():
            if nt not in targets or targets[nt].numel() == 0:
                continue

            true = targets[nt]
            type_w = self.type_weights.get(nt, 1.0)
            node_loss = type_w * self._elementwise(pred, true)
            components[f"{nt}_loss"] = node_loss.item()

            total_loss = node_loss if total_loss is None else total_loss + node_loss

            if self.use_physical_constraints and nt == "beam":
                # penalize predicted height > width for beams
                violation = torch.relu(pred[:, 1] - pred[:, 0])
                c_loss = torch.mean(violation) * 0.1
                total_loss = total_loss + c_loss
                components[f"{nt}_constraint"] = c_loss.item()

        if total_loss is None:
            # No valid nodes in this graph: return a differentiable zero so
            # .backward() never crashes.
            device = (
                next(iter(predictions.values())).device
                if predictions
                else torch.device("cpu")
            )
            total_loss = torch.zeros((), device=device, requires_grad=True)

        return total_loss, components
