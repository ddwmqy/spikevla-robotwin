from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from .configuration import VisionEncoderConfig


def _load_pretrained_model(config: VisionEncoderConfig):
    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
    }
    if config.attention_implementation:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                attn_implementation=config.attention_implementation,
                **kwargs,
            )
        except Exception as error:
            if config.attention_implementation == "flash_attention_2":
                try:
                    return AutoModel.from_pretrained(
                        config.model_name_or_path,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                except Exception:
                    pass
            print(
                f"[TurboVLA] vision attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class DINOv3VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _load_pretrained_model(config)
        self.hidden_size = self._hidden_size(self.backbone.config)
        self.patch_size = self._patch_size(self.backbone.config)
        self.prefix_tokens = self._prefix_tokens(self.backbone.config, default=5)
        self.num_patches = (config.image_size // self.patch_size) ** 2

        embeddings = getattr(self.backbone, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)
        if config.frozen:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _hidden_size(config) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("cannot infer DINOv3 hidden size")

    @staticmethod
    def _patch_size(config) -> int:
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("cannot infer DINOv3 patch size")

    @staticmethod
    def _prefix_tokens(config, default: int = 0) -> int:
        if hasattr(config, "num_register_tokens"):
            return int(config.num_register_tokens) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(config.vision_config.num_register_tokens) + 1
        return int(default)

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported DINOv3 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            with autocast_context:
                outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        tokens = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        if tokens.shape[1] == expected_patches + self.prefix_tokens:
            return tokens[:, self.prefix_tokens :, :]
        if tokens.shape[1] == expected_patches:
            return tokens
        raise RuntimeError(
            f"DINOv3 produced {tokens.shape[1]} tokens; expected {expected_patches} patches "
            f"or {expected_patches + self.prefix_tokens} tokens including prefixes"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            return torch.stack(
                [self._encode_images(pixel_values[:, view_idx]) for view_idx in range(num_views)],
                dim=1,
            )
        flat = pixel_values.flatten(0, 1)
        tokens = self._encode_images(flat)
        return tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2])


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("SDT-V3 checkpoint does not contain a state dict")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in checkpoint.items()}


class SDTV3VisionEncoder(nn.Module):
    """Adapter for the official 19M Spike-driven Transformer V3 base model.

    The official SFA training model emits a [B, 360, 14, 14] integer-spike
    feature map for 224px inputs. We keep the spatial map and expose it as
    TurboVLA tokens instead of applying the ImageNet pooling/classification head.
    """

    hidden_size = 360
    patch_size = 16

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.image_size != 224:
            raise ValueError("SDT-V3 19M pretrained backbone requires image_size=224")
        if config.output_grid_size != 14:
            raise ValueError("SDT-V3 19M adapter currently exposes the final 14x14 feature map")
        self.num_patches = config.output_grid_size ** 2
        self.backbone = self._load_official_model(config)
        self.backbone.head = nn.Identity()
        if config.frozen:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _default_source_path() -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "Spike-Driven-Transformer-V3"
            / "SDT_V3"
            / "Classification"
            / "Model_Base"
            / "models.py"
        )

    def _load_official_model(self, config: VisionEncoderConfig) -> nn.Module:
        source_path = Path(config.model_source_path) if config.model_source_path else self._default_source_path()
        if not source_path.is_file():
            raise FileNotFoundError(f"official SDT-V3 model source not found: {source_path}")
        spec = importlib.util.spec_from_file_location("turbovla_official_sdtv3_models", source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import official SDT-V3 model source: {source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backbone = module.Efficient_Spiking_Transformer_l()

        checkpoint_path = config.pretrained_checkpoint or config.model_name_or_path
        if not checkpoint_path:
            raise ValueError("SDT-V3 requires vision.pretrained_checkpoint or model_name_or_path")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"SDT-V3 pretrained checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
        checkpoint_model = getattr(checkpoint_args, "model", None)
        if checkpoint_model is None and isinstance(checkpoint_args, dict):
            checkpoint_model = checkpoint_args.get("model")
        # The bundled 19M weights use the pre-renaming architecture label.
        # Tensor keys/shapes are still checked below; this is not a partial-load fallback.
        compatible_metadata_names = {
            "Efficient_Spiking_Transformer_l",
            "spikformer_8_15M_CAFormer",
        }
        if checkpoint_model and checkpoint_model not in compatible_metadata_names:
            raise RuntimeError(
                "SDT-V3 19M requires Efficient_Spiking_Transformer_l weights, but checkpoint metadata says "
                f"{checkpoint_model!r}. The upstream README currently links a 10M checkpoint from its 19M entry."
            )
        state = _extract_checkpoint_state(checkpoint)
        incompatible = backbone.load_state_dict(state, strict=False)
        allowed_missing = {"head.weight", "head.bias"}
        missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(
                "SDT-V3 checkpoint is incompatible with the official 19M architecture: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        return backbone

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported SDT-V3 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context, autocast_context:
            features = self.backbone.forward_features(pixel_values)
        if features.ndim != 4 or tuple(features.shape[-2:]) != (14, 14):
            raise RuntimeError(f"SDT-V3 expected [B,360,14,14], got {tuple(features.shape)}")
        return features.flatten(2).transpose(1, 2).contiguous()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            return torch.stack(
                [self._encode_images(pixel_values[:, view_idx]) for view_idx in range(num_views)], dim=1
            )
        tokens = self._encode_images(pixel_values.flatten(0, 1))
        return tokens.view(batch_size, num_views, self.num_patches, self.hidden_size)


def build_vision_encoder(config: VisionEncoderConfig) -> nn.Module:
    if config.encoder_type == "dinov3":
        return DINOv3VisionEncoder(config)
    if config.encoder_type == "sdtv3_19m":
        return SDTV3VisionEncoder(config)
    raise ValueError(f"unsupported vision encoder type: {config.encoder_type}")
