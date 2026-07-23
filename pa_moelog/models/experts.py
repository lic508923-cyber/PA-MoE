"""Lightweight source-domain experts operating on one shared event encoding."""

from __future__ import annotations

import torch
from torch import nn

from .dora import DoRALinear


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
        self.target_projection: DoRALinear | None = None

    def enable_target_dora(self, rank: int = 4, alpha: float | None = None) -> None:
        """Attach DoRA to the expert's trained output projection.

        The zero-initialized low-rank branch exactly preserves the source expert
        at activation time; target adaptation then learns only direction and
        magnitude changes around that trained source-domain weight.
        """
        source = self.projection[3]
        if not isinstance(source, nn.Linear):
            raise TypeError("expert output projection must be linear")
        adapter = DoRALinear(source.in_features, source.out_features, rank=rank,
                             bias=source.bias is not None,
                             alpha=float(rank if alpha is None else alpha)).to(
                                 device=source.weight.device, dtype=source.weight.dtype)
        adapter.load_base_from_linear(source)
        self.target_projection = adapter

    def _hidden(self, shared_hidden: torch.Tensor, *, target_adapted: bool) -> torch.Tensor:
        hidden = self.projection[0](shared_hidden)
        hidden = self.projection[1](hidden)
        hidden = self.projection[2](hidden)
        if target_adapted:
            if self.target_projection is None:
                raise RuntimeError("target DoRA has not been enabled")
            hidden = self.target_projection(hidden)
        else:
            hidden = self.projection[3](hidden)
        return self.projection[4](hidden)

    def forward(self, shared_hidden: torch.Tensor) -> dict:
        hidden = self._hidden(shared_hidden, target_adapted=False)
        logit = self.classifier(hidden).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}

    def forward_target(self, shared_hidden: torch.Tensor) -> dict:
        hidden = self._hidden(shared_hidden, target_adapted=True)
        logit = self.classifier(hidden).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}


class ExpertPool(nn.Module):
    """Run and fuse all lightweight experts over a shared representation."""

    def __init__(self, num_experts: int, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [LogExpert(hidden_dim=hidden_dim, rank=rank, dropout=dropout) for _ in range(num_experts)]
        )

    def enable_target_dora(self, rank: int = 4, alpha: float | None = None) -> None:
        for expert in self.experts:
            expert.enable_target_dora(rank=rank, alpha=alpha)

    def forward(self, shared_hidden: torch.Tensor, fusion_weights: torch.Tensor,
                *, target_adapted: bool = False) -> dict:
        if fusion_weights.shape != (shared_hidden.size(0), len(self.experts)):
            raise ValueError("fusion_weights must have shape [batch, num_experts]")
        outputs = [(expert.forward_target(shared_hidden) if target_adapted else expert(shared_hidden))
                   for expert in self.experts]
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
