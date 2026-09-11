"""P0A gate #1: build the full A-arm spike config in v4_code and run one CPU forward.

A arm = smoothspike text + SDT-V3 vision + spike_sdsa cross-attention, 3 views,
14-D dual-arm action. Verifies: strict weight loading (fused SmoothSpike + 19M SDT-V3),
config validation, and numerical health (finite outputs, no NaN) on CPU.
"""
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path.insert(0, root)
sys.meta_path[:] = [m for m in sys.meta_path if "editable" not in type(m).__name__.lower()]

import torch  # noqa: E402
import turbovla  # noqa: E402
from turbovla.models import TurboVLAConfig, build_turbovla  # noqa: E402
from turbovla.models.configuration import (  # noqa: E402
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)

assert turbovla.__file__.startswith(root), f"turbovla resolved to {turbovla.__file__}"

RES = "/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained"
cfg = TurboVLAConfig(
    text_mask_version="corrected",
    text=TextEncoderConfig(
        encoder_type="smoothspike_bert",
        model_name_or_path=f"{RES}/SmoothSpike/smoothspike-bert-base-fused",
        timesteps=4,
        local_files_only=True,
        attention_implementation=None,
        max_length=64,
    ),
    vision=VisionEncoderConfig(
        encoder_type="sdtv3_19m",
        model_name_or_path=f"{RES}/V3_19.0M_1x4.pth",
        image_size=224,
        num_views=3,
        compute_precision="fp32",
        attention_implementation=None,
        local_files_only=True,
    ),
    interaction=InteractionConfig(
        cross_attention_type="spike_sdsa",
        cross_timesteps=4,
        attention_backend="sdpa",
        compute_precision="fp32",
    ),
    action=ActionHeadConfig(
        action_dim=14,
        state_dim=14,
        horizon=50,
        num_state_tokens=2,
        num_layers=3,
        mlp_hidden_dim=512,
        state_hidden_dim=256,
        dropout=0.1,
    ),
)
torch.manual_seed(0)
model = build_turbovla(cfg)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params total={n_params:,} trainable={trainable:,}")
print(f"text encoder: {type(model.text_encoder.bert).__name__}")
print(f"vision encoder: {type(model.vision_encoder).__name__} (backbone {type(model.vision_encoder.backbone).__name__})")
print(f"cross-attention: {type(model.vision_language_interaction.fusion_layers[0].attn).__name__}")

with torch.no_grad():
    torch.manual_seed(1)
    pixel_values = torch.rand(2, 3, 3, 224, 224)
    out = model(
        ["put both the moka pot and the yellow mug on the stove", "place the alphabet soup in the basket"],
        {"pixel_values": pixel_values},
        torch.rand(2, 14),
    )

print(f"out shape={list(out.shape)} finite={bool(torch.isfinite(out).all())} |out|sum={out.abs().sum().item():.6f}")
assert out.shape == (2, 50, 14) and torch.isfinite(out).all()
print("GATE #1 PASS")
