"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    """Save a model checkpoint and create parent directories automatically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": dict(config),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"[checkpoint] saved: {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, optionally into an existing model with non-strict matching."""

    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    print(f"[checkpoint] loaded: {path}")
    print(f"[checkpoint] tensors: {len(state_dict)}")

    if model is not None:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f"[checkpoint][warning] missing keys: {missing}")
        if unexpected:
            print(f"[checkpoint][warning] unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print("[checkpoint] model parameters matched exactly")

    return checkpoint
