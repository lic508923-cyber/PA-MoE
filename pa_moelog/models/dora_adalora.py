"""最小 DoRA-AdaLoRA 风格轻量适配层。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """冻结线性层加可训练的低秩 DoRA 方向更新。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        bias: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 冻结基础权重：保持原始投影不变。
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        # 低秩方向更新：B @ A 生成可训练的增量方向。
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))

        # 幅值缩放：近似 DoRA 中独立的范数/幅值路径。
        self.magnitude = nn.Parameter(torch.ones(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_linear(x)
        active_rank = min(max(int(self.rank), 1), self.lora_a.size(0), self.lora_b.size(1))
        direction = F.linear(F.linear(x, self.lora_a[:active_rank]), self.lora_b[:, :active_rank])
        scaled_update = direction * self.magnitude * (self.alpha / max(self.rank, 1))
        return base_output + scaled_update


class AdaLoRAController:
    """供后续自适应秩逻辑使用的小型秩预算控制器接口。"""

    def __init__(self, initial_rank: int = 4, min_rank: int = 1, max_rank: int = 16) -> None:
        self.current_rank = initial_rank
        self.min_rank = min_rank
        self.max_rank = max_rank

    def get_current_rank(self) -> int:
        return self.current_rank

    def update_rank(self, new_rank: int) -> int:
        self.current_rank = max(self.min_rank, min(self.max_rank, int(new_rank)))
        return self.current_rank

    def apply_budget(
        self,
        importance_scores: Mapping[object, float] | Sequence[float] | torch.Tensor | None = None,
        total_budget: int | None = None,
        modules: Mapping[object, DoRALinear] | Sequence[DoRALinear] | None = None,
    ):
        if importance_scores is None:
            return self.current_rank

        keys = list(importance_scores) if isinstance(importance_scores, Mapping) else None
        values = list(importance_scores.values()) if keys is not None else importance_scores
        if isinstance(values, torch.Tensor):
            scores = values.detach().to(dtype=torch.float32, device="cpu").flatten()
        else:
            scores = torch.as_tensor(values, dtype=torch.float32).flatten()
        count = int(scores.numel())
        if count == 0:
            return {} if keys is not None else []

        budget = self.current_rank * count if total_budget is None else int(total_budget)
        budget = max(self.min_rank * count, min(self.max_rank * count, budget))
        ranks = torch.full((count,), self.min_rank, dtype=torch.long)
        remaining = budget - self.min_rank * count
        capacity = self.max_rank - self.min_rank

        if remaining > 0 and capacity > 0:
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp_min_(0.0)
            if float(scores.sum()) == 0.0:
                scores.fill_(1.0)

            extra = torch.zeros(count, dtype=torch.long)
            cap_left = torch.full((count,), capacity, dtype=torch.long)
            while remaining > 0:
                active = cap_left > 0
                if not bool(active.any()):
                    break

                active_scores = scores[active]
                weight_sum = float(active_scores.sum())
                if weight_sum == 0.0:
                    active_scores = torch.ones_like(active_scores)
                    weight_sum = float(active_scores.sum())

                raw = active_scores * (remaining / weight_sum)
                add = torch.minimum(raw.floor().long(), cap_left[active])
                if int(add.sum()) == 0:
                    active_indices = active.nonzero(as_tuple=False).flatten()
                    take = min(remaining, active_indices.numel())
                    chosen = active_indices[torch.topk(scores[active_indices], k=take).indices]
                    extra[chosen] += 1
                    cap_left[chosen] -= 1
                    remaining -= int(take)
                else:
                    active_indices = active.nonzero(as_tuple=False).flatten()
                    extra[active_indices] += add
                    cap_left[active_indices] -= add
                    remaining -= int(add.sum())

            ranks += extra

        result = ranks.tolist()
        if modules is not None:
            if isinstance(modules, Mapping):
                module_values = [modules[key] for key in keys] if keys is not None else list(modules.values())
            else:
                module_values = modules
            for module, rank in zip(module_values, result):
                module.rank = min(int(rank), module.lora_a.size(0), module.lora_b.size(1))

        if keys is not None:
            return dict(zip(keys, result))
        return result
