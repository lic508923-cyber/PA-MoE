"""用于目标域异常边界校准的对角协方差 GMM。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GMMEnergy(nn.Module):
    """根据目标表示的负对数似然计算异常能量。"""

    def __init__(self, hidden_dim: int, num_components: int = 4, min_var: float = 1e-4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_var = min_var
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, hidden_dim) * 0.02)
        self.log_vars = nn.Parameter(torch.zeros(num_components, hidden_dim))

    def compute_energy(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算每个样本的负对数似然能量与混合分量责任度。"""
        log_weights = F.log_softmax(self.mixture_logits, dim=0)
        variances = F.softplus(self.log_vars) + self.min_var
        difference = hidden.unsqueeze(1) - self.means.unsqueeze(0)
        log_det = torch.log(variances).sum(dim=-1).unsqueeze(0)
        mahalanobis = (difference.pow(2) / variances.unsqueeze(0)).sum(dim=-1)
        log_prob = -0.5 * (mahalanobis + log_det + self.hidden_dim * math.log(2.0 * math.pi))
        component_log_prob = log_prob + log_weights.unsqueeze(0)
        log_likelihood = torch.logsumexp(component_log_prob, dim=-1)
        responsibility = F.softmax(component_log_prob, dim=-1)
        return -log_likelihood, responsibility

    def forward(self, hidden: torch.Tensor) -> dict:
        """返回能量分数及各高斯分量责任度。"""
        energy, responsibility = self.compute_energy(hidden)
        return {"energy": energy, "responsibility": responsibility}

    @torch.no_grad()
    def fit_normal(self, hidden: torch.Tensor) -> None:
        """使用目标域正常支持样本重新估计对角协方差密度模型。"""
        if hidden.ndim != 2 or hidden.size(1) != self.hidden_dim or hidden.size(0) < 2:
            raise ValueError("正常支持样本表示必须为 [N, hidden_dim]，且 N 不小于 2。")
        mean = hidden.mean(dim=0)
        variance = hidden.var(dim=0, unbiased=False).clamp_min(self.min_var)
        self.means.copy_(mean.unsqueeze(0).expand_as(self.means))
        # 反变换到未约束空间，前向时通过 softplus 保证方差始终为正。
        target = (variance - self.min_var).clamp_min(1e-8)
        self.log_vars.copy_(torch.log(torch.expm1(target)).unsqueeze(0).expand_as(self.log_vars))
        self.mixture_logits.zero_()
