"""用于免模板解析日志表示的参数感知注意力编码器。"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


PARAMETER_TYPES = ["IP", "PATH", "URL", "HEX", "NUM", "PORT", "USER", "PID", "FILE", "TIME"]


class ParameterEncoder(nn.Module):
    """将参数类型与原始参数值联合编码为稠密向量。"""

    def __init__(self, hidden_dim: int, value_vocab_size: int = 4096) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_vocab_size = value_vocab_size
        self.type_to_index = {name: index for index, name in enumerate(PARAMETER_TYPES)}
        self.type_embedding = nn.Embedding(len(PARAMETER_TYPES), hidden_dim)
        self.value_embedding = nn.Embedding(value_vocab_size, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, parameters: List[Dict[str, List[str]]], device: torch.device | None = None) -> torch.Tensor:
        """对每条日志中的全部显式参数进行平均池化编码。"""
        device = device or self.type_embedding.weight.device
        encoded_rows = []
        for item in parameters:
            embeddings = []
            for parameter_type in PARAMETER_TYPES:
                type_embedding = self.type_embedding(
                    torch.tensor(self.type_to_index[parameter_type], device=device)
                )
                for value in item.get(parameter_type, []):
                    value_index = torch.tensor(self._hash_value(value), device=device)
                    embeddings.append(type_embedding + self.value_embedding(value_index))
            row = torch.stack(embeddings).mean(dim=0) if embeddings else torch.zeros(self.hidden_dim, device=device)
            encoded_rows.append(row)
        return self.output_norm(torch.stack(encoded_rows))

    def encode_sequence(self, parameters: List[Dict[str, List[str]]], seq_len: int, device=None) -> torch.Tensor:
        """将日志级参数表示广播到文本序列的每个位置。"""
        return self.forward(parameters, device).unsqueeze(1).expand(-1, seq_len, -1)

    def _hash_value(self, value: str) -> int:
        """使用稳定哈希将任意参数值映射到有限词表。"""
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest, 16) % self.value_vocab_size


class ParameterAwareAttention(nn.Module):
    """将显式参数信息转换为注意力偏置的自注意力模块。"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("隐藏维度必须能够被注意力头数整除。")
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
        """在文本自注意力分数中注入参数感知偏置。"""
        batch_size, seq_len, _ = text_embeddings.shape
        if parameter_embeddings.dim() == 2:
            parameter_embeddings = parameter_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        query = self._split_heads(self.q_proj(text_embeddings))
        key = self._split_heads(self.k_proj(text_embeddings))
        value = self._split_heads(self.v_proj(text_embeddings))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.parameter_bias(parameter_embeddings).permute(0, 2, 1).unsqueeze(2)
        output = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """将隐藏维拆分为多个注意力头。"""
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class ParameterAwareEncoder(nn.Module):
    """使用参数感知注意力融合词元级文本表示。"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.parameter_encoder = ParameterEncoder(hidden_dim)
        self.attention = ParameterAwareAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, text_embeddings, parameters=None, parameter_embeddings=None) -> torch.Tensor:
        """生成同时包含日志语义与显式参数信息的日志级表示。"""
        if parameter_embeddings is None:
            if parameters is None:
                raise ValueError("必须提供 parameters 或 parameter_embeddings。")
            parameter_embeddings = self.parameter_encoder.encode_sequence(
                parameters, text_embeddings.size(1), text_embeddings.device
            )
        attended = self.attention(text_embeddings, parameter_embeddings)
        hidden = self.norm(text_embeddings + attended)
        return self.norm(hidden + self.ffn(hidden)).mean(dim=1)
