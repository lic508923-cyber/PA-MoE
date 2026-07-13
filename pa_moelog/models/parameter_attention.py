"""Parameter-token encoding and parameter-aware cross-attention."""

from __future__ import annotations

import hashlib
from typing import Dict, List

import torch
from torch import nn


PARAMETER_TYPES = ["IP", "PATH", "URL", "HEX", "NUM", "PORT", "USER", "PID", "FILE", "TIME"]


class ParameterEncoder(nn.Module):
    """Encode every extracted parameter as an independent typed token."""

    def __init__(self, hidden_dim: int, value_vocab_size: int = 4096) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_vocab_size = value_vocab_size
        self.type_to_index = {name: index for index, name in enumerate(PARAMETER_TYPES)}
        self.type_embedding = nn.Embedding(len(PARAMETER_TYPES), hidden_dim)
        self.value_embedding = nn.Embedding(value_vocab_size, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def encode_tokens(
        self,
        parameters: List[Dict[str, List[str]]],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded parameter tokens and a ``True`` mask for real tokens."""
        if not parameters:
            raise ValueError("parameters must contain at least one sample")
        device = device or self.type_embedding.weight.device
        rows: list[list[tuple[int, int]]] = []
        for item in parameters:
            row: list[tuple[int, int]] = []
            for parameter_type in PARAMETER_TYPES:
                type_index = self.type_to_index[parameter_type]
                row.extend((type_index, self._hash_value(value)) for value in item.get(parameter_type, []))
            rows.append(row)

        # Keep one masked padding position so all-empty batches remain valid tensors.
        max_tokens = max(1, max(len(row) for row in rows))
        tokens = self.type_embedding.weight.new_zeros((len(rows), max_tokens, self.hidden_dim))
        mask = torch.zeros((len(rows), max_tokens), dtype=torch.bool, device=device)
        for row_index, row in enumerate(rows):
            if not row:
                continue
            type_ids = torch.tensor([item[0] for item in row], dtype=torch.long, device=device)
            value_ids = torch.tensor([item[1] for item in row], dtype=torch.long, device=device)
            count = len(row)
            tokens[row_index, :count] = self.type_embedding(type_ids) + self.value_embedding(value_ids)
            mask[row_index, :count] = True
        return self.output_norm(tokens), mask

    def forward(self, parameters: List[Dict[str, List[str]]], device: torch.device | None = None) -> torch.Tensor:
        """Return a masked mean for callers that need a single parameter vector."""
        tokens, mask = self.encode_tokens(parameters, device)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode_sequence(self, parameters, seq_len: int, device=None) -> torch.Tensor:
        """Compatibility helper; prefer :meth:`encode_tokens` for cross-attention."""
        return self.forward(parameters, device).unsqueeze(1).expand(-1, seq_len, -1)

    def _hash_value(self, value: str) -> int:
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest, 16) % self.value_vocab_size


class ParameterAwareAttention(nn.Module):
    """Cross-attend from text tokens to independent parameter tokens."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        parameter_embeddings: torch.Tensor,
        parameter_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parameter_embeddings.dim() != 3:
            raise ValueError("parameter_embeddings must have shape [batch, parameters, hidden]")
        if parameter_mask is None:
            parameter_mask = torch.ones(parameter_embeddings.shape[:2], dtype=torch.bool, device=parameter_embeddings.device)
        if parameter_mask.shape != parameter_embeddings.shape[:2]:
            raise ValueError("parameter_mask must match the first two parameter embedding dimensions")

        # MultiheadAttention cannot consume a row whose every key is masked. A
        # zero sentinel makes that row numerically safe; its output is removed.
        has_parameters = parameter_mask.any(dim=1)
        safe_mask = parameter_mask.clone()
        safe_mask[~has_parameters, 0] = True
        output, _ = self.attention(
            query=text_embeddings,
            key=parameter_embeddings,
            value=parameter_embeddings,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        return output * has_parameters[:, None, None].to(output.dtype)


class ParameterAwareEncoder(nn.Module):
    """Fuse text with parameters once, before the lightweight expert branches."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.parameter_encoder = ParameterEncoder(hidden_dim)
        self.attention = ParameterAwareAttention(hidden_dim, num_heads, dropout)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        parameters: List[Dict[str, List[str]]] | None = None,
        parameter_embeddings: torch.Tensor | None = None,
        parameter_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parameter_embeddings is None:
            if parameters is None:
                raise ValueError("parameters or parameter_embeddings must be provided")
            parameter_embeddings, parameter_mask = self.parameter_encoder.encode_tokens(
                parameters, text_embeddings.device
            )
        attended = self.attention(text_embeddings, parameter_embeddings, parameter_mask)
        hidden = self.attention_norm(text_embeddings + torch.tanh(self.gate) * attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        return hidden.mean(dim=1)
