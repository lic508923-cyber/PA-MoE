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

    def forward(self, semantic_texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """将文本转换为带位置编码的词元级隐藏状态。"""
        device = self.embedding.weight.device
        input_ids = torch.tensor([self._tokenize(text) for text in semantic_texts], device=device)
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)
        mask = input_ids.ne(0)
        hidden = self.norm(self.embedding(input_ids) + self.position_embedding(positions))
        return hidden * mask.unsqueeze(-1).to(hidden.dtype), mask

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

    def forward(self, semantic_texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """完成分词、BERT 编码与目标隐藏维投影。"""
        device = next(self.bert.parameters()).device
        encoded = self.tokenizer(
            semantic_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.bert(**encoded)
        mask = encoded["attention_mask"].bool()
        hidden = self.norm(self.projection(outputs.last_hidden_state))
        return hidden * mask.unsqueeze(-1).to(hidden.dtype), mask


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
        max_length=32, allow_hash_fallback=False, sequence_layers=1,
        max_events=512, gmm_projection_dim=32, fusion_shrinkage_strength=16.0, dora_rank=4,
        disable_parameters=False, disable_gmm=False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.num_experts = num_experts
        self.requested_backbone_name = backbone_name
        self.max_events = max_events
        self.sequence_layers = sequence_layers
        self.disable_parameters = disable_parameters
        self.disable_gmm = disable_gmm
        self.text_encoder, self.backbone_name = build_text_encoder(
            hidden_dim, backbone_name, max_length, allow_hash_fallback
        )
        self.parameter_encoder = ParameterEncoder(hidden_dim)
        self.fusion_encoder = ParameterAwareEncoder(hidden_dim)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True, norm_first=False,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=sequence_layers, enable_nested_tensor=False
        )
        self.event_position_embedding = nn.Embedding(max_events, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        # 静态融合默认均匀利用全部专家，也可由目标支持集一次性校准。
        self.fusion = LightweightExpertFusion(num_experts, shrinkage_strength=fusion_shrinkage_strength)
        self.expert_pool = ExpertPool(num_experts, hidden_dim)
        # 共享 DoRA 位于专家融合之后，是目标域主要的可训练适配参数。
        self.target_adapter = DoRALinear(hidden_dim, hidden_dim, rank=dora_rank)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.target_classifier = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.target_classifier.weight)
        nn.init.zeros_(self.target_classifier.bias)
        self.gmm_energy = GMMEnergy(hidden_dim, num_gmm_components, projection_dim=gmm_projection_dim)
        self.register_buffer("energy_location", torch.tensor(0.0))
        self.register_buffer("energy_scale", torch.tensor(1.0))
        self.register_buffer("energy_stats_fitted", torch.tensor(False))

    def encode_events(self, semantic_texts, parameters) -> torch.Tensor:
        text_embeddings, text_mask = self.text_encoder(semantic_texts)
        if self.disable_parameters:
            parameter_embeddings=text_embeddings.new_zeros((text_embeddings.size(0),1,self.hidden_dim))
            parameter_mask=torch.zeros((text_embeddings.size(0),1),dtype=torch.bool,device=text_embeddings.device)
        else:
            parameter_embeddings, parameter_mask = self.parameter_encoder.encode_tokens(parameters, text_embeddings.device)
        return self.fusion_encoder(
            text_embeddings,
            parameter_embeddings=parameter_embeddings,
            parameter_mask=parameter_mask,
            text_mask=text_mask,
        )

    def checkpoint_signature(self) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "hidden_dim": self.hidden_dim,
            "num_experts": self.num_experts,
            "sequence_layers": self.sequence_layers,
            "max_events": self.max_events,
            "dora_rank": self.target_adapter.rank,
            "gmm_projection_dim": self.gmm_energy.projection_dim,
            "disable_parameters": self.disable_parameters,
            "disable_gmm": self.disable_gmm,
        }

    def encode_sequences(self, semantic_sequences, parameter_sequences, event_mask=None) -> torch.Tensor:
        lengths = [len(sequence) for sequence in semantic_sequences]
        if not lengths or min(lengths) < 1:
            raise ValueError("every sequence must contain at least one event")
        flat_texts = [text for sequence in semantic_sequences for text in sequence]
        flat_parameters = [item for sequence in parameter_sequences for item in sequence]
        flat_hidden = self.encode_events(flat_texts, flat_parameters)
        max_events = max(lengths)
        if max_events > self.max_events:
            raise ValueError(f"sequence has {max_events} events but max_events={self.max_events}")
        padded = flat_hidden.new_zeros((len(lengths), max_events, self.hidden_dim))
        mask = torch.zeros((len(lengths), max_events), dtype=torch.bool, device=flat_hidden.device)
        cursor = 0
        for index, length in enumerate(lengths):
            padded[index, :length] = flat_hidden[cursor:cursor + length]
            mask[index, :length] = True
            cursor += length
        if event_mask is not None:
            if tuple(event_mask.shape) != tuple(mask.shape):
                raise ValueError(f"event_mask shape {tuple(event_mask.shape)} does not match {tuple(mask.shape)}")
            supplied = event_mask.to(device=mask.device, dtype=torch.bool)
            if not torch.equal(supplied.sum(1).cpu(), torch.tensor(lengths)):
                raise ValueError("event_mask counts must match sequence lengths")
            mask = supplied
        positions = torch.arange(max_events, device=padded.device)
        padded = padded + self.event_position_embedding(positions).unsqueeze(0)
        encoded = self.sequence_encoder(padded, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        return self.sequence_norm((encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0))

    def forward(self, semantic_texts, parameters, event_mask: torch.Tensor | None = None) -> dict:
        """返回分类证据、能量证据、静态专家权重和最终异常分数。"""
        if semantic_texts and isinstance(semantic_texts[0], (list, tuple)):
            shared_hidden = self.encode_sequences(semantic_texts, parameters, event_mask)
        else:
            shared_hidden = self.encode_sequences(
                [[text] for text in semantic_texts], [[item] for item in parameters]
            )
        fusion_weights = self.fusion(
            shared_hidden.size(0), device=shared_hidden.device, dtype=shared_hidden.dtype
        )
        expert_output = self.expert_pool(shared_hidden, fusion_weights)
        target_hidden = self.target_norm(self.target_adapter(expert_output["hidden"]))
        target_logit = expert_output["combined_logit"] + self.target_classifier(target_hidden).squeeze(-1)
        classifier_score = torch.sigmoid(target_logit)
        raw_energy = self.gmm_energy(target_hidden)["energy"]
        energy_score = (self._normalize_energy(raw_energy) if not self.disable_gmm and bool(self.gmm_energy.is_fitted)
                        else torch.full_like(raw_energy, 0.5))
        final_score = (classifier_score if self.disable_gmm else
                       torch.clamp(self.alpha * classifier_score + self.beta * energy_score, 0.0, 1.0))
        return {
            "final_score": final_score,
            "classifier_score": classifier_score,
            "logit": target_logit,
            "energy_score": energy_score,
            "raw_energy": raw_energy,
            "fusion_weights": fusion_weights,
            "target_hidden": target_hidden,
            "shared_hidden": shared_hidden,
            "expert_logits": expert_output["expert_logits"],
            "expert_hiddens": expert_output["expert_hiddens"],
        }

    def _normalize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """将批次能量标准化到零到一之间。"""
        return torch.sigmoid((energy - self.energy_location) / self.energy_scale.clamp_min(1e-6))

    @torch.no_grad()
    def fit_energy_statistics(self, hidden: torch.Tensor) -> None:
        energy = self.gmm_energy(hidden)["energy"]
        self.energy_location.copy_(energy.mean())
        self.energy_scale.copy_(energy.std(unbiased=False).clamp_min(1e-6))
        self.energy_stats_fitted.fill_(True)
