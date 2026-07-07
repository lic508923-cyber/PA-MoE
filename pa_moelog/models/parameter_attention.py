"""用于免解析日志表示的参数感知注意力编码器。"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


PARAMETER_TYPES = ["IP", "PATH", "URL", "HEX", "NUM", "PORT", "USER", "PID", "FILE", "TIME"]


class ParameterEncoder(nn.Module):
    """将带类型的参数字典编码为稠密嵌入。"""

    def __init__(self, hidden_dim: int, value_vocab_size: int = 4096) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_vocab_size = value_vocab_size
        self.type_to_index = {name: index for index, name in enumerate(PARAMETER_TYPES)}
        self.type_embedding = nn.Embedding(len(PARAMETER_TYPES), hidden_dim)
        self.value_embedding = nn.Embedding(value_vocab_size, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, parameters: List[Dict[str, List[str]]], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.type_embedding.weight.device

        encoded_rows = []
        for item in parameters:
            token_embeddings = []
            for param_type in PARAMETER_TYPES:
                values = item.get(param_type, [])
                if not values:
                    continue
                type_index = torch.tensor(self.type_to_index[param_type], device=device)
                type_emb = self.type_embedding(type_index)
                for value in values:
                    value_index = torch.tensor(self._hash_value(value), device=device)
                    value_emb = self.value_embedding(value_index)
                    token_embeddings.append(type_emb + value_emb)
            if token_embeddings:
                row = torch.stack(token_embeddings, dim=0).mean(dim=0)
            else:
                row = torch.zeros(self.hidden_dim, device=device)
            encoded_rows.append(row)

        return self.output_norm(torch.stack(encoded_rows, dim=0))

    def encode_sequence(
        self,
        parameters: List[Dict[str, List[str]]],
        seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pooled = self.forward(parameters, device=device)
        return pooled.unsqueeze(1).expand(-1, seq_len, -1)

    def _hash_value(self, value: str) -> int:
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest, 16) % self.value_vocab_size


class ParameterAwareAttention(nn.Module):
    """在注意力分数中加入参数偏置的自注意力模块。"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除。")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.parameter_bias = nn.Linear(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeddings: torch.Tensor, parameter_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = text_embeddings.shape
        if parameter_embeddings.dim() == 2:
            parameter_embeddings = parameter_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        query = self._split_heads(self.q_proj(text_embeddings))
        key = self._split_heads(self.k_proj(text_embeddings))
        value = self._split_heads(self.v_proj(text_embeddings))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.parameter_bias(parameter_embeddings).permute(0, 2, 1).unsqueeze(2)
        scores = scores + bias

        attention = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)


class ParameterAwareEncoder(nn.Module):
    """使用参数感知注意力融合 token 级文本嵌入。"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.parameter_encoder = ParameterEncoder(hidden_dim)
        self.attention = ParameterAwareAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        parameters: List[Dict[str, List[str]]] | None = None,
        parameter_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parameter_embeddings is None:
            if parameters is None:
                raise ValueError("必须提供 parameters 或 parameter_embeddings。")
            parameter_embeddings = self.parameter_encoder.encode_sequence(
                parameters,
                seq_len=text_embeddings.size(1),
                device=text_embeddings.device,
            )

        attended = self.attention(text_embeddings, parameter_embeddings)
        hidden = self.norm(text_embeddings + attended)
        hidden = self.norm(hidden + self.ffn(hidden))
        return hidden.mean(dim=1)
