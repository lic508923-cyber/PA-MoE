"""PA-MoELog 最新轻量多源专家模型。"""

from __future__ import annotations

import hashlib
import warnings
from typing import Dict, List

import torch
from torch import nn

from .dora import DoRALinear
from .experts import ExpertPool
from .fusion import LightweightExpertFusion
from .gmm_energy import GMMEnergy
from .parameter_attention import ParameterAwareEncoder, ParameterEncoder


class SimpleTextEncoder(nn.Module):
    """用于离线验证的轻量哈希文本编码器。"""

    def __init__(self, hidden_dim: int, vocab_size: int = 8192, max_length: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, semantic_texts: List[str]) -> torch.Tensor:
        """将文本转换为带位置编码的词元级隐藏状态。"""
        device = self.embedding.weight.device
        input_ids = torch.tensor([self._tokenize(text) for text in semantic_texts], device=device)
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)
        return self.norm(self.embedding(input_ids) + self.position_embedding(positions))

    def _tokenize(self, text: str) -> list[int]:
        """按空格切分文本并映射到稳定哈希词表。"""
        ids = [self._hash_token(token) for token in text.lower().split()[: self.max_length]]
        return ids + [0] * (self.max_length - len(ids))

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % (self.vocab_size - 1) + 1


class BertTextEncoder(nn.Module):
    """调用 Hugging Face BERT 并返回词元级隐藏状态。"""

    def __init__(self, hidden_dim: int, backbone_name: str = "bert-base-uncased", max_length: int = 32) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("BERT 主干需要 transformers 包；请安装依赖或改用 simple-hash-encoder。") from exc
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.bert = AutoModel.from_pretrained(backbone_name)
        bert_hidden = int(self.bert.config.hidden_size)
        self.projection = nn.Identity() if bert_hidden == hidden_dim else nn.Linear(bert_hidden, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, semantic_texts: List[str]) -> torch.Tensor:
        """完成分词、BERT 编码与目标隐藏维投影。"""
        device = next(self.bert.parameters()).device
        encoded = self.tokenizer(
            semantic_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        outputs = self.bert(**{key: value.to(device) for key, value in encoded.items()})
        return self.norm(self.projection(outputs.last_hidden_state))


def build_text_encoder(hidden_dim, backbone_name, max_length=32, allow_hash_fallback=True):
    """构建指定文本编码器，并返回实际启用的编码器名称。"""
    if backbone_name in {"simple-hash-encoder", "hash"}:
        return SimpleTextEncoder(hidden_dim, max_length=max_length), "simple-hash-encoder"
    try:
        return BertTextEncoder(hidden_dim, backbone_name, max_length), backbone_name
    except (ImportError, OSError, RuntimeError) as exc:
        if not allow_hash_fallback:
            raise
        warnings.warn(
            f"BERT 主干 {backbone_name!r} 不可用，已回退到 simple-hash-encoder：{exc}",
            RuntimeWarning, stacklevel=2,
        )
        return SimpleTextEncoder(hidden_dim, max_length=max_length), "simple-hash-encoder"


class PAMoELog(nn.Module):
    """组合参数感知编码、多源专家、共享 DoRA 和目标感知 GMM。"""

    def __init__(
        self, hidden_dim=128, num_experts=3, num_gmm_components=4,
        alpha=0.7, beta=0.3, backbone_name="bert-base-uncased",
        max_length=32, allow_hash_fallback=True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.num_experts = num_experts
        self.requested_backbone_name = backbone_name
        self.text_encoder, self.backbone_name = build_text_encoder(
            hidden_dim, backbone_name, max_length, allow_hash_fallback
        )
        self.parameter_encoder = ParameterEncoder(hidden_dim)
        self.fusion_encoder = ParameterAwareEncoder(hidden_dim)
        # 静态融合默认均匀利用全部专家，也可由目标支持集一次性校准。
        self.fusion = LightweightExpertFusion(num_experts)
        self.expert_pool = ExpertPool(num_experts, hidden_dim)
        # 共享 DoRA 位于专家融合之后，是目标域主要的可训练适配参数。
        self.target_adapter = DoRALinear(hidden_dim, hidden_dim, rank=4)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.target_classifier = nn.Linear(hidden_dim, 1)
        self.gmm_energy = GMMEnergy(hidden_dim, num_gmm_components)

    def forward(self, semantic_texts: List[str], parameters: List[Dict[str, List[str]]]) -> dict:
        """返回分类证据、能量证据、静态专家权重和最终异常分数。"""
        text_embeddings = self.text_encoder(semantic_texts)
        parameter_embeddings = self.parameter_encoder.encode_sequence(
            parameters, text_embeddings.size(1), text_embeddings.device
        )
        shared_hidden = self.fusion_encoder(text_embeddings, parameter_embeddings=parameter_embeddings)
        fusion_weights = self.fusion(
            shared_hidden.size(0), device=shared_hidden.device, dtype=shared_hidden.dtype
        )
        expert_output = self.expert_pool(
            text_embeddings, fusion_weights, parameter_embeddings=parameter_embeddings
        )
        target_hidden = self.target_norm(self.target_adapter(expert_output["hidden"]))
        target_logit = self.target_classifier(target_hidden).squeeze(-1)
        classifier_score = torch.sigmoid(target_logit)
        energy_score = self._normalize_energy(self.gmm_energy(target_hidden)["energy"])
        final_score = torch.clamp(self.alpha * classifier_score + self.beta * energy_score, 0.0, 1.0)
        return {
            "final_score": final_score,
            "classifier_score": classifier_score,
            "logit": target_logit,
            "energy_score": energy_score,
            "fusion_weights": fusion_weights,
            "target_hidden": target_hidden,
            "expert_logits": expert_output["expert_logits"],
        }

    @staticmethod
    def _normalize_energy(energy: torch.Tensor) -> torch.Tensor:
        """将批次能量标准化到零到一之间。"""
        if energy.numel() <= 1:
            return torch.sigmoid(energy)
        return torch.sigmoid((energy - energy.mean()) / energy.std(unbiased=False).clamp_min(1e-6))
