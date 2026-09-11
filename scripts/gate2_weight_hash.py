"""P0A gate #2: post-build weight-hash assertion (§5.2-style, applied to the A arm).

Builds the A-arm model, then verifies that spike text and vision weights inside the
model are bitwise identical to their source checkpoints — i.e. nothing between the
loader and the training entry point silently re-initializes or mutates them.
"""
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path.insert(0, root)
sys.meta_path[:] = [m for m in sys.meta_path if "editable" not in type(m).__name__.lower()]

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from turbovla.models import TurboVLAConfig, build_turbovla  # noqa: E402
from turbovla.models.configuration import (  # noqa: E402
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)
from turbovla.models.vision_encoder import SDTV3VisionEncoder  # noqa: E402

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
    interaction=InteractionConfig(cross_attention_type="spike_sdsa", cross_timesteps=4, attention_backend="sdpa", compute_precision="fp32"),
    action=ActionHeadConfig(action_dim=14, state_dim=14, horizon=50),
)
torch.manual_seed(0)
model = build_turbovla(cfg)

state = model.state_dict()

# --- text: compare every fused SmoothSpike tensor actually consumed by the loader ---
# Fused file keys are relative to BertForMaskedLM ("bert.H1..."); the model keeps the
# backbone (full_model.bert), whose state_dict drops that prefix.
fused = load_file(f"{RES}/SmoothSpike/smoothspike-bert-base-fused/model.safetensors")
checked = excluded = 0
for key, src in fused.items():
    if not key.startswith("bert."):
        excluded += 1  # MLM-head tensors (cls.*) are discarded by the loader by design
        continue
    tgt_key = f"text_encoder.bert.{key[len('bert.'):]}"
    assert tgt_key in state, f"missing in model: {tgt_key}"
    assert torch.equal(state[tgt_key], src), f"mismatch: {tgt_key}"
    checked += 1
print(f"text: {checked}/{len(fused)} fused tensors bitwise-equal ({excluded} cls.* excluded)")

# --- vision: rebuild the backbone independently and compare the backbone state ---
sdt_state = SDTV3VisionEncoder(cfg.vision).state_dict()
checked = 0
for key, src in sdt_state.items():
    rel = key[len("backbone.") :] if key.startswith("backbone.") else key
    tgt_key = f"vision_encoder.backbone.{rel}"
    assert tgt_key in state, f"missing in model: {tgt_key}"
    assert torch.equal(state[tgt_key], src), f"mismatch: {tgt_key}"
    checked += 1
print(f"vision: {checked}/{len(sdt_state)} backbone tensors bitwise-equal")

# --- bert backbone frozen (text_projection stays trainable), vision trainable ---
assert all(not p.requires_grad for p in model.text_encoder.bert.parameters()), "bert backbone must be frozen"
assert all(p.requires_grad for p in model.text_encoder.text_projection.parameters()), "text projection must stay trainable"
assert any(p.requires_grad for p in model.vision_encoder.parameters()), "vision encoder must be trainable"
print("freeze state: bert frozen / projection+vision trainable — OK")
print("GATE #2 PASS")
