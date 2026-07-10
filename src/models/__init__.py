"""
Heterogeneous GNN models for RC element dimension prediction.

`build_model(config)` is the single, config-driven factory used by the training
notebooks so model choice lives in the YAML (`model.type`), not in notebook
code. Add a new model by subclassing BaseHeteroGNN and registering it below.

Imports are done lazily inside the factory so that merely importing this
package doesn't pull in torch_geometric until a model is actually built.
"""

from typing import Any, Dict


def build_model(config: Dict[str, Any]):
    """Instantiate the model named by config['model']['type'] ('hgt' | 'han').

    Both models share the exact same constructor signature, so the wiring is
    identical — only the operator differs. Raises ValueError on unknown types.
    """
    mc = config["model"]
    mtype = str(mc.get("type", "hgt")).lower()

    kwargs = dict(
        hidden_channels=mc["hidden_channels"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        dropout=mc["dropout"],
        node_types=mc.get("node_types", ["beam", "column"]),
        output_dim=mc.get("output_dim", 2),
        use_structural_encoding=mc.get("use_structural_encoding", True),
    )

    if mtype == "hgt":
        from src.models.hgt import HGT
        return HGT(**kwargs)
    if mtype == "han":
        from src.models.han import HAN
        return HAN(**kwargs)

    raise ValueError(
        f"Unknown model.type '{mtype}'. Supported: 'hgt', 'han'."
    )
