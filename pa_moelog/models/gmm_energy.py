"""稳定的对角协方差 GMM 能量边界。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GMMEnergy(nn.Module):
    """基于可训练 GMM 计算负对数似然能量。"""

    def __init__(self, hidden_dim: int, num_components: int = 4, min_var: float = 1e-4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_var = min_var
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, hidden_dim) * 0.02)
        self.log_vars = nn.Parameter(torch.zeros(num_components, hidden_dim))

    def compute_energy(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_weights = F.log_softmax(self.mixture_logits, dim=0)
        vars_ = F.softplus(self.log_vars) + self.min_var
        diff = hidden.unsqueeze(1) - self.means.unsqueeze(0)
        log_det = torch.log(vars_).sum(dim=-1).unsqueeze(0)
        mahalanobis = (diff.pow(2) / vars_.unsqueeze(0)).sum(dim=-1)
        log_prob = -0.5 * (mahalanobis + log_det + self.hidden_dim * math.log(2.0 * math.pi))
        component_log_prob = log_prob + log_weights.unsqueeze(0)
        log_likelihood = torch.logsumexp(component_log_prob, dim=-1)
        responsibility = F.softmax(component_log_prob, dim=-1)
        return -log_likelihood, responsibility

    def forward(self, hidden: torch.Tensor) -> dict:
        energy, responsibility = self.compute_energy(hidden)
        return {"energy": energy, "responsibility": responsibility}
