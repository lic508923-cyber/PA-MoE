"""Stable diagonal-covariance GMM anomaly energy."""

from __future__ import annotations

import math

import torch
from torch import nn


class GMMEnergy(nn.Module):
    """A fitted density model; parameters are buffers, not gradient targets."""

    def __init__(self, hidden_dim: int, num_components: int = 1, min_var: float = 1e-4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_var = min_var
        self.register_buffer("mixture_weights", torch.full((num_components,), 1.0 / num_components))
        self.register_buffer("means", torch.zeros(num_components, hidden_dim))
        self.register_buffer("variances", torch.ones(num_components, hidden_dim))
        self.register_buffer("active_components", torch.tensor(1, dtype=torch.long))
        self.register_buffer("is_fitted", torch.tensor(False))

    def compute_energy(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = int(self.active_components.item())
        weights = self.mixture_weights[:k].clamp_min(1e-12)
        means = self.means[:k]
        variances = self.variances[:k].clamp_min(self.min_var)
        difference = hidden.unsqueeze(1) - means.unsqueeze(0)
        log_prob = -0.5 * (
            (difference.square() / variances.unsqueeze(0)).sum(dim=-1)
            + torch.log(variances).sum(dim=-1).unsqueeze(0)
            + self.hidden_dim * math.log(2.0 * math.pi)
        )
        component_log_prob = log_prob + torch.log(weights).unsqueeze(0)
        log_likelihood = torch.logsumexp(component_log_prob, dim=-1)
        return -log_likelihood, torch.softmax(component_log_prob, dim=-1)

    def forward(self, hidden: torch.Tensor) -> dict:
        energy, responsibility = self.compute_energy(hidden)
        return {"energy": energy, "responsibility": responsibility}

    @torch.no_grad()
    def fit_normal(self, hidden: torch.Tensor, max_iter: int = 50, tolerance: float = 1e-4) -> None:
        if hidden.ndim != 2 or hidden.size(1) != self.hidden_dim or hidden.size(0) < 2:
            raise ValueError("normal hidden states must have shape [N, hidden_dim] with N >= 2")
        # At least two samples per component avoids an unstable few-shot mixture.
        k = min(self.num_components, max(1, hidden.size(0) // 2))
        indices = torch.linspace(0, hidden.size(0) - 1, steps=k, device=hidden.device).long()
        means = hidden[indices].clone()
        variances = hidden.var(dim=0, unbiased=False).clamp_min(self.min_var).expand(k, -1).clone()
        weights = hidden.new_full((k,), 1.0 / k)
        previous = None
        for _ in range(max_iter):
            diff = hidden[:, None, :] - means[None, :, :]
            log_prob = -0.5 * (
                (diff.square() / variances[None, :, :]).sum(-1)
                + torch.log(variances).sum(-1)[None, :]
                + self.hidden_dim * math.log(2.0 * math.pi)
            ) + torch.log(weights.clamp_min(1e-12))[None, :]
            responsibility = torch.softmax(log_prob, dim=1)
            counts = responsibility.sum(0).clamp_min(1e-6)
            weights = counts / counts.sum()
            means = (responsibility.T @ hidden) / counts[:, None]
            diff = hidden[:, None, :] - means[None, :, :]
            variances = (responsibility[:, :, None] * diff.square()).sum(0) / counts[:, None]
            variances.clamp_(min=self.min_var)
            likelihood = torch.logsumexp(log_prob, dim=1).mean()
            if previous is not None and abs(float(likelihood - previous)) < tolerance:
                break
            previous = likelihood
        self.mixture_weights.zero_(); self.mixture_weights[:k].copy_(weights)
        self.means.zero_(); self.means[:k].copy_(means)
        self.variances.fill_(1.0); self.variances[:k].copy_(variances)
        self.active_components.fill_(k)
        self.is_fitted.fill_(True)
