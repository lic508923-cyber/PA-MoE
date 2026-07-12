"""用于目标域参数高效适配的 DoRA 层。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """在线性基础层冻结时，仅训练低秩方向增量和幅值参数。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        bias: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 基础投影保存源域知识，在目标域适配阶段保持冻结。
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        # 低秩矩阵只调整权重方向，幅值参数负责独立缩放各输出维度。
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化低秩方向，使模型初始行为等价于冻结基础层。"""
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算冻结基础投影与 DoRA 低秩增量之和。"""
        base_output = self.base_linear(x)
        direction = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_output + direction * self.magnitude * (self.alpha / max(self.rank, 1))
