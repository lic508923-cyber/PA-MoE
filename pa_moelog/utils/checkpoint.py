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
    if hasattr(model, "checkpoint_signature"):
        checkpoint["model_signature"] = model.checkpoint_signature()
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"[checkpoint] saved: {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint; model restoration is strict unless explicitly opted out."""

    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    print(f"[checkpoint] loaded: {path}")
    print(f"[checkpoint] tensors: {len(state_dict)}")

    if model is not None:
        incompatible = restore_checkpoint(checkpoint, model, strict=strict)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if not strict and missing:
            print(f"[checkpoint][warning] missing keys: {missing}")
        if not strict and unexpected:
            print(f"[checkpoint][warning] unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print("[checkpoint] model parameters matched exactly")

    return checkpoint


def restore_checkpoint(checkpoint: dict[str, Any], model: nn.Module, strict: bool = True):
    """Restore an already-read checkpoint and validate architecture metadata."""
    if strict and hasattr(model, "checkpoint_signature"):
        expected = model.checkpoint_signature()
        actual = checkpoint.get("model_signature")
        if actual is None:
            raise RuntimeError("checkpoint is missing model_signature; explicit migration is required")
        if actual != expected:
            differences = {key: {"checkpoint": actual.get(key), "model": expected.get(key)}
                           for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
            raise RuntimeError(f"checkpoint model signature mismatch: {differences}")
    return model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=strict)
