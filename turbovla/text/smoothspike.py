from __future__ import annotations

from pathlib import Path

from transformers import BertConfig


def load_smoothspike_model(model_name_or_path: str, timesteps: int):
    """Strictly load the official fused SmoothSpike BERT checkpoint."""
    model_dir = Path(model_name_or_path)
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"SmoothSpike config not found: {config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"SmoothSpike fused checkpoint not found: {weight_path}")

    smooth_config = BertConfig.from_pretrained(str(model_dir), local_files_only=True)
    published_timesteps = int(getattr(smooth_config, "T", timesteps))
    if published_timesteps != int(timesteps):
        raise ValueError(
            f"SmoothSpike checkpoint was trained with T={published_timesteps}, "
            f"but text.timesteps={timesteps}"
        )
    smooth_config.T = int(timesteps)
    smooth_config._attn_implementation = "eager"

    from safetensors.torch import load_file
    from third_party.SmoothSpike.spikingbert_rot_inf import BertForMaskedLM

    full_model = BertForMaskedLM(smooth_config)
    state = load_file(str(weight_path), device="cpu")
    if any(".H2" in key or ".H3" in key for key in state):
        raise ValueError(
            "SmoothSpike runtime expects the official fused checkpoint; "
            "the supplied file still contains per-layer H2/H3 tensors"
        )

    missing_backbone = sorted(
        key for key in full_model.bert.state_dict() if f"bert.{key}" not in state
    )
    if missing_backbone:
        raise RuntimeError(
            f"SmoothSpike checkpoint is missing backbone tensors: {missing_backbone[:20]}"
        )

    missing, unexpected = full_model.load_state_dict(state, strict=False)
    allowed_tied_aliases = {
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
    }
    disallowed_missing = sorted(set(missing) - allowed_tied_aliases)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "SmoothSpike checkpoint is not strictly compatible: "
            f"missing={disallowed_missing[:20]}, unexpected={unexpected[:20]}"
        )
    full_model.tie_weights()

    backbone = full_model.bert
    backbone.load_report = {
        "checkpoint_tensors": len(state),
        "missing_tied_aliases": sorted(set(missing) & allowed_tied_aliases),
        "unexpected": list(unexpected),
    }
    return backbone
