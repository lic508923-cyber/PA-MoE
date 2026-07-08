"""PA-MoELog：最小可运行模型骨架。"""

from __future__ import annotations

import hashlib
import warnings
from typing import Dict, List

import torch
from torch import nn

from .experts import ExpertPool
from .gmm_energy import GMMEnergy
from .parameter_attention import ParameterAwareEncoder, ParameterEncoder
from .router import TopKSparseRouter


class SimpleTextEncoder(nn.Module):
    """基于哈希 token 的文本编码器，后续可替换为 BERT/ModernBERT。"""

    def __init__(self, hidden_dim: int, vocab_size: int = 8192, max_length: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, semantic_texts: List[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        token_ids = [self._tokenize(text) for text in semantic_texts]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        return self.norm(embeddings)

    def _tokenize(self, text: str) -> list[int]:
        tokens = text.lower().split()[: self.max_length]
        ids = [self._hash_token(token) for token in tokens]
        ids.extend([0] * (self.max_length - len(ids)))
        return ids

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % (self.vocab_size - 1) + 1


class BertTextEncoder(nn.Module):
    """Hugging Face BERT encoder that returns token-level hidden states."""

    def __init__(self, hidden_dim: int, backbone_name: str = "bert-base-uncased", max_length: int = 32) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERT backbone requires the 'transformers' package. Install it with "
                "`pip install transformers` or set backbone_name='simple-hash-encoder'."
            ) from exc

        self.hidden_dim = hidden_dim
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.bert = AutoModel.from_pretrained(backbone_name)
        bert_hidden = int(self.bert.config.hidden_size)
        self.projection = nn.Identity() if bert_hidden == hidden_dim else nn.Linear(bert_hidden, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, semantic_texts: List[str]) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        encoded = self.tokenizer(
            semantic_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.bert(**encoded)
        hidden = self.projection(outputs.last_hidden_state)
        return self.norm(hidden)


def build_text_encoder(
    hidden_dim: int,
    backbone_name: str,
    max_length: int = 32,
    allow_hash_fallback: bool = True,
) -> tuple[nn.Module, str]:
    """Build the requested text encoder and return the actual encoder name."""

    if backbone_name in {"simple-hash-encoder", "hash"}:
        return SimpleTextEncoder(hidden_dim=hidden_dim, max_length=max_length), "simple-hash-encoder"

    try:
        return BertTextEncoder(hidden_dim=hidden_dim, backbone_name=backbone_name, max_length=max_length), backbone_name
    except (ImportError, OSError, RuntimeError) as exc:
        if not allow_hash_fallback:
            raise
        warnings.warn(
            f"Falling back to simple-hash-encoder because BERT backbone '{backbone_name}' is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return SimpleTextEncoder(hidden_dim=hidden_dim, max_length=max_length), "simple-hash-encoder"


class PAMoELog(nn.Module):
    """仅包含前向传播的最小 PA-MoELog 骨架。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_experts: int = 3,
        top_k: int = 2,
        num_gmm_components: int = 4,
        alpha: float = 0.7,
        beta: float = 0.3,
        backbone_name: str = "bert-base-uncased",
        max_length: int = 32,
        allow_hash_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.requested_backbone_name = backbone_name

        self.text_encoder, self.backbone_name = build_text_encoder(
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            max_length=max_length,
            allow_hash_fallback=allow_hash_fallback,
        )
        self.parameter_encoder = ParameterEncoder(hidden_dim=hidden_dim)
        self.fusion_encoder = ParameterAwareEncoder(hidden_dim=hidden_dim)
        self.router = TopKSparseRouter(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
        self.expert_pool = ExpertPool(num_experts=num_experts, hidden_dim=hidden_dim)
        self.gmm_energy = GMMEnergy(hidden_dim=hidden_dim, num_components=num_gmm_components)

    def forward(self, semantic_texts: List[str], parameters: List[Dict[str, List[str]]]) -> dict:
        text_embeddings = self.text_encoder(semantic_texts)
        parameter_embeddings = self.parameter_encoder.encode_sequence(
            parameters,
            seq_len=text_embeddings.size(1),
            device=text_embeddings.device,
        )
        fused_hidden = self.fusion_encoder(
            text_embeddings=text_embeddings,
            parameter_embeddings=parameter_embeddings,
        )

        router_output = self.router(fused_hidden)
        expert_output = self.expert_pool(
            text_embeddings=text_embeddings,
            parameter_embeddings=parameter_embeddings,
            router_weights=router_output["weights"],
        )
        gmm_output = self.gmm_energy(fused_hidden)

        classifier_score = expert_output["classifier_score"]
        energy_score = self._normalize_energy(gmm_output["energy"])
        final_score = torch.clamp(self.alpha * classifier_score + self.beta * energy_score, 0.0, 1.0)

        return {
            "final_score": final_score,
            "classifier_score": classifier_score,
            "logit": expert_output["combined_logit"],
            "energy_score": energy_score,
            "router_weights": router_output["weights"],
            "selected_experts": router_output["selected_experts"],
            "router_aux_loss": router_output["aux_loss"],
            "expert_logits": expert_output["expert_logits"],
        }

    @staticmethod
    def _normalize_energy(energy: torch.Tensor) -> torch.Tensor:
        if energy.numel() <= 1:
            return torch.sigmoid(energy)
        mean = energy.mean()
        std = energy.std(unbiased=False).clamp_min(1e-6)
        return torch.sigmoid((energy - mean) / std)
