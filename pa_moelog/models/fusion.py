"""PA-MoELog 的稳定轻量专家融合模块。"""

from __future__ import annotations

import torch
from torch import nn


class LightweightExpertFusion(nn.Module):
    """为一个批次生成均匀或由支持集校准的静态专家权重。"""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("专家数量必须为正整数。")
        self.num_experts = num_experts
        self.register_buffer("weights", torch.full((num_experts,), 1.0 / num_experts))

    @torch.no_grad()
    def set_weights(self, weights: torch.Tensor) -> None:
        """设置静态专家权重，并自动完成非负约束和归一化。"""
        weights = torch.as_tensor(weights, dtype=self.weights.dtype, device=self.weights.device)
        if weights.shape != (self.num_experts,):
            raise ValueError(f"权重形状必须为 ({self.num_experts},)。")
        weights = weights.clamp_min(0)
        if float(weights.sum()) <= 0:
            raise ValueError("至少需要一个大于零的专家权重。")
        self.weights.copy_(weights / weights.sum())

    @torch.no_grad()
    def calibrate_from_distances(self, distances: torch.Tensor, temperature: float = 1.0) -> None:
        """根据目标支持集与各源专家中心的距离计算静态权重。"""
        if temperature <= 0:
            raise ValueError("温度系数必须大于零。")
        distances = torch.as_tensor(distances, dtype=self.weights.dtype, device=self.weights.device)
        self.set_weights(torch.softmax(-distances / temperature, dim=0))

    def forward(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """将同一组目标域静态权重广播到批次中的所有样本。"""
        return self.weights.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
