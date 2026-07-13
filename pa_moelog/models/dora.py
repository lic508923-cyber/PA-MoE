"""Weight-decomposed DoRA linear layer for target-domain adaptation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """Apply DoRA: normalize ``W0 + BA`` by row, then learn row magnitudes."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, bias: bool = True, alpha: float = 1.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = alpha
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
        self.reset_parameters()
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # An identity base makes the residual adapter neutral at initialization.
        with torch.no_grad():
            if self.base_linear.in_features == self.base_linear.out_features:
                self.base_linear.weight.copy_(torch.eye(self.base_linear.in_features))
                if self.base_linear.bias is not None:
                    self.base_linear.bias.zero_()
            row_norm = self.base_linear.weight.norm(dim=1).clamp_min(1e-8)
            self.magnitude.copy_(row_norm)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_b @ self.lora_a) * (self.alpha / self.rank)
        direction = self.base_linear.weight + delta
        weight = self.magnitude[:, None] * F.normalize(direction, p=2, dim=1)
        return F.linear(x, weight, self.base_linear.bias)
