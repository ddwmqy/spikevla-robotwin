from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    encoder_type: str = "bert"
    model_name_or_path: str = "bert-base-uncased"
    timesteps: int = 1
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    encoder_type: str = "dinov3"
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1
    pretrained_checkpoint: str | None = None
    model_source_path: str | None = None
    output_grid_size: int = 14


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"
    cross_attention_type: str = "ann"
    cross_timesteps: int = 4
    cross_gradient_checkpointing: bool = True


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    # v4 port: default "legacy" = official upstream behavior (mask repeat semantics +
    # final-[SEP] rule), so official checkpoints / C1 reproduce official numerics by
    # default. Spike arms (B/A) set "corrected" explicitly in the framework config.
    text_mask_version: str = "legacy"
    text_sentence_mask_version: str | None = None
    text_head_mask_version: str | None = None
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.text_mask_version not in {"legacy", "corrected"}:
            raise ValueError("text_mask_version must be legacy or corrected")
        for name in ("text_sentence_mask_version", "text_head_mask_version"):
            if getattr(self, name) not in {None, "legacy", "corrected"}:
                raise ValueError(f"{name} must be legacy, corrected, or None")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.encoder_type not in {"dinov3", "sdtv3_19m"}:
            raise ValueError("vision.encoder_type must be 'dinov3' or 'sdtv3_19m'")
        if self.text.encoder_type not in {"bert", "smoothspike_bert"}:
            raise ValueError("text.encoder_type must be 'bert' or 'smoothspike_bert'")
        if self.text.timesteps < 1:
            raise ValueError("text.timesteps must be positive")
        if self.text.encoder_type == "smoothspike_bert" and self.text.attention_implementation not in {None, "eager"}:
            raise ValueError("SmoothSpike text encoder requires eager spike-driven attention")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.cross_attention_type not in {"ann", "spike_sdsa"}:
            raise ValueError("interaction.cross_attention_type must be ann or spike_sdsa")
        if self.interaction.cross_timesteps < 1:
            raise ValueError("interaction.cross_timesteps must be positive")
        if self.interaction.cross_attention_type == "spike_sdsa" and self.interaction.fusion_dropout != 0:
            raise ValueError("spike_sdsa requires fusion_dropout=0 to preserve binary spike operands")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")

    @property
    def sentence_mask_version(self) -> str:
        return self.text_sentence_mask_version or self.text_mask_version

    @property
    def head_mask_version(self) -> str:
        return self.text_head_mask_version or self.text_mask_version

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            # Unversioned checkpoints were trained with the original mask behavior.
            text_mask_version=str(data.get("text_mask_version", "legacy")),
            text_sentence_mask_version=data.get("text_sentence_mask_version"),
            text_head_mask_version=data.get("text_head_mask_version"),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
        )
