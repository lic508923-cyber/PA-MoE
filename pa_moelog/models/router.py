"""用于混合专家路由的 Top-k 稀疏路由器。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TopKSparseRouter(nn.Module):
    """将每个样本仅路由到 top-k 个专家。"""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k 必须位于 [1, num_experts] 范围内。")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, hidden: torch.Tensor) -> dict:
        logits = self.gate(hidden)
        top_values, selected_experts = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        weights = torch.zeros_like(logits)
        weights.scatter_(dim=-1, index=selected_experts, src=top_weights)

        importance = weights.mean(dim=0)
        target = torch.full_like(importance, 1.0 / self.num_experts)
        aux_loss = F.mse_loss(importance, target) * self.num_experts

        return {
            "weights": weights,
            "selected_experts": selected_experts,
            "aux_loss": aux_loss,
        }
