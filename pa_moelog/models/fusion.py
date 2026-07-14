"""Static support-calibrated fusion with an explicit trained-expert mask."""

from __future__ import annotations

import torch
from torch import nn


class LightweightExpertFusion(nn.Module):
    def __init__(self, num_experts: int, shrinkage_strength: float = 16.0) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.shrinkage_strength = float(shrinkage_strength)
        self.register_buffer("weights", torch.full((num_experts,), 1.0 / num_experts))
        self.register_buffer("trained_mask", torch.ones(num_experts, dtype=torch.bool))

    @torch.no_grad()
    def set_trained_mask(self, mask: torch.Tensor) -> None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.weights.device)
        if mask.shape != (self.num_experts,) or not bool(mask.any()):
            raise ValueError("trained mask must select at least one expert")
        self.trained_mask.copy_(mask)
        self.set_weights(torch.where(mask, self.weights, torch.zeros_like(self.weights)))

    @torch.no_grad()
    def set_weights(self, weights: torch.Tensor) -> None:
        weights = torch.as_tensor(weights, dtype=self.weights.dtype, device=self.weights.device)
        if weights.shape != (self.num_experts,):
            raise ValueError(f"weights must have shape ({self.num_experts},)")
        weights = weights.clamp_min(0) * self.trained_mask
        if float(weights.sum()) <= 0:
            weights = self.trained_mask.to(weights.dtype)
        self.weights.copy_(weights / weights.sum())

    @torch.no_grad()
    def calibrate_from_distances(self, distances: torch.Tensor, temperature: float = 1.0,
                                 label_budget: int | None = None) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        distances = torch.as_tensor(distances, dtype=self.weights.dtype, device=self.weights.device)
        if distances.shape != (self.num_experts,):
            raise ValueError(f"distances must have shape ({self.num_experts},)")
        logits = -distances / temperature
        logits = logits.masked_fill(~self.trained_mask, float("-inf"))
        calibrated = torch.softmax(logits, dim=0)
        if label_budget is not None:
            if label_budget < 0:
                raise ValueError("label_budget cannot be negative")
            prior = self.trained_mask.to(calibrated.dtype); prior = prior / prior.sum()
            reliability = label_budget / (label_budget + self.shrinkage_strength)
            calibrated = reliability * calibrated + (1.0 - reliability) * prior
        self.set_weights(calibrated)

    def forward(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.weights.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
