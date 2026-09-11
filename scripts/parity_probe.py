"""P0A gate #0: official-config build parity for the v4 port.

Builds TurboVLA with a purely official config (bert + dinov3 + ann cross-attention,
mask_version default) in whichever repo root is given, dumps parameter-name/shape
table and a fixed-seed forward output. Run once per root and diff the JSONs:
any difference means the port drifted the official path.
"""
import json
import sys

root, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
# Neutralize editable-install meta path finders so `import turbovla` resolves to `root`.
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

assert turbovla.__file__.startswith(root), f"turbovla resolved to {turbovla.__file__}, expected under {root}"

RES = "/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained"
cfg = TurboVLAConfig(
    text=TextEncoderConfig(
        model_name_or_path=f"{RES}/bert-base-uncased",
        local_files_only=True,
        attention_implementation=None,
        max_length=64,
    ),
    vision=VisionEncoderConfig(
        model_name_or_path=f"{RES}/dinov3-vitb16-pretrain-lvd1689m",
        image_size=224,
        num_views=2,
        attention_implementation=None,
        compute_precision="fp32",
        local_files_only=True,
    ),
    interaction=InteractionConfig(attention_backend="sdpa", compute_precision="fp32"),
    action=ActionHeadConfig(
        action_dim=7,
        state_dim=7,
        horizon=12,
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
shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

with torch.no_grad():
    torch.manual_seed(1)
    pixel_values = torch.rand(2, 2, 3, 224, 224)
    # Two instructions with different token lengths exercise padding + special-token masks.
    out = model(
        ["push the plate forward", "pick up the red mug and place it on the stove"],
        {"dinov3": pixel_values},
        torch.rand(2, 7),
    )

payload = {
    "root": root,
    "n_params": len(shapes),
    "param_total": int(sum(v.numel() for v in model.state_dict().values())),
    "shapes": shapes,
    "out_shape": list(out.shape),
    "out": out.tolist(),
}
with open(out_path, "w") as fh:
    json.dump(payload, fh)
print(f"{root}: params={len(shapes)} total={payload['param_total']} out={list(out.shape)} |out|sum={out.abs().sum().item():.6f}")
