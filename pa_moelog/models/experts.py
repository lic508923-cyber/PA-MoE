"""PA-MoELog 的多源专家池。"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .dora_adalora import DoRALinear
from .parameter_attention import ParameterAwareEncoder


class LogExpert(nn.Module):
    """一个带参数感知融合和预测头的轻量源专家。"""

    def __init__(self, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = ParameterAwareEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.projection = nn.Sequential(
            DoRALinear(hidden_dim, hidden_dim, rank=rank),
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
        hidden = self.encoder(
            text_embeddings=text_embeddings,
            parameters=parameters,
            parameter_embeddings=parameter_embeddings,
        )
        hidden = self.projection(hidden)
        logit = self.classifier(hidden).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}


class ExpertPool(nn.Module):
    """计算专家输出，并用稀疏路由权重进行组合。"""

    def __init__(self, num_experts: int, hidden_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [LogExpert(hidden_dim=hidden_dim, rank=rank, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        router_weights: torch.Tensor,
        parameters: List[Dict[str, List[str]]] | None = None,
        parameter_embeddings: torch.Tensor | None = None,
    ) -> dict:
        outputs = [
            expert(
                text_embeddings=text_embeddings,
                parameters=parameters,
                parameter_embeddings=parameter_embeddings,
            )
            for expert in self.experts
        ]
        probs = torch.stack([item["prob"] for item in outputs], dim=-1)
        logits = torch.stack([item["logit"] for item in outputs], dim=-1)
        hiddens = torch.stack([item["hidden"] for item in outputs], dim=1)

        classifier_score = (probs * router_weights).sum(dim=-1)
        combined_hidden = (hiddens * router_weights.unsqueeze(-1)).sum(dim=1)
        combined_logit = (logits * router_weights).sum(dim=-1)

        return {
            "classifier_score": classifier_score,
            "combined_logit": combined_logit,
            "hidden": combined_hidden,
            "expert_probs": probs,
        }
