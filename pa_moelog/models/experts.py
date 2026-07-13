"""Lightweight source-domain experts operating on one shared event encoding."""

from __future__ import annotations

import torch
from torch import nn


class LogExpert(nn.Module):
    """A small domain branch; parameter-aware encoding is shared upstream."""

    def __init__(self, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        bottleneck_dim = max(rank, hidden_dim // 4)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, shared_hidden: torch.Tensor) -> dict:
        hidden = self.projection(shared_hidden)
        logit = self.classifier(hidden).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}


class ExpertPool(nn.Module):
    """Run and fuse all lightweight experts over a shared representation."""

    def __init__(self, num_experts: int, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [LogExpert(hidden_dim=hidden_dim, rank=rank, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, shared_hidden: torch.Tensor, fusion_weights: torch.Tensor) -> dict:
        if fusion_weights.shape != (shared_hidden.size(0), len(self.experts)):
            raise ValueError("fusion_weights must have shape [batch, num_experts]")
        outputs = [expert(shared_hidden) for expert in self.experts]
        probs = torch.stack([item["prob"] for item in outputs], dim=-1)
        logits = torch.stack([item["logit"] for item in outputs], dim=-1)
        hiddens = torch.stack([item["hidden"] for item in outputs], dim=1)
        return {
            "classifier_score": (probs * fusion_weights).sum(dim=-1),
            "combined_logit": (logits * fusion_weights).sum(dim=-1),
            "hidden": (hiddens * fusion_weights.unsqueeze(-1)).sum(dim=1),
            "expert_hiddens": hiddens,
            "expert_probs": probs,
            "expert_logits": logits,
        }
