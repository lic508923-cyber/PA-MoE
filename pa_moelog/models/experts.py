"""PA-MoELog 的多源领域专家池。"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .parameter_attention import ParameterAwareEncoder


class LogExpert(nn.Module):
    """学习单个源系统日志分布的轻量领域专家。"""

    def __init__(self, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = ParameterAwareEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        parameters: List[Dict[str, List[str]]] | None = None,
        parameter_embeddings: torch.Tensor | None = None,
    ) -> dict:
        """返回单个专家的分类结果与隐藏表示。"""
        hidden = self.encoder(text_embeddings, parameters, parameter_embeddings)
        hidden = self.projection(hidden)
        logit = self.classifier(hidden).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}


class ExpertPool(nn.Module):
    """并行计算全部源专家输出，并使用静态权重完成融合。"""

    def __init__(self, num_experts: int, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [LogExpert(hidden_dim=hidden_dim, rank=rank, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        fusion_weights: torch.Tensor,
        parameters: List[Dict[str, List[str]]] | None = None,
        parameter_embeddings: torch.Tensor | None = None,
    ) -> dict:
        """根据目标域静态权重融合专家概率、分类值和隐藏表示。"""
        outputs = [
            expert(text_embeddings, parameters, parameter_embeddings)
            for expert in self.experts
        ]
        probs = torch.stack([item["prob"] for item in outputs], dim=-1)
        logits = torch.stack([item["logit"] for item in outputs], dim=-1)
        hiddens = torch.stack([item["hidden"] for item in outputs], dim=1)
        return {
            "classifier_score": (probs * fusion_weights).sum(dim=-1),
            "combined_logit": (logits * fusion_weights).sum(dim=-1),
            "hidden": (hiddens * fusion_weights.unsqueeze(-1)).sum(dim=1),
            "expert_probs": probs,
            "expert_logits": logits,
        }
