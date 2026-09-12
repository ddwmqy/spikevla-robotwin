"""P0A gate #2b: full wrapper construction from an arm config (CPU-capable).

End-to-end construction check: yaml → merge_framework_config → TurboVLAFramework →
core model → image processor → predict_action. Catches config-validation bugs (e.g. a
wrong attn_implementation for a spike encoder) that the direct-config gates bypass, and
verifies the env-var contract of the yaml.

Usage (env vars must be set as in the yaml; add load_pretrained=false if the init
checkpoint is not available yet):
    python scripts/gate2b_wrapper_build.py [config.yaml] [--set key=value ...]
"""
import os
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path[:0] = [
    root,
    f"{root}/third_party/starvla_runtime",
    f"{root}/third_party/vla_adapter",
]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework  # noqa: E402

argv = sys.argv[1:]
positional = [a for a in argv if not a.startswith("--")]
config_path = positional[0] if positional else f"{root}/experiments/robotwin/configs/clean50_a.yaml"
sets = [a.lstrip("-") for a in argv if a.startswith("--")]
print(f"config: {config_path}")

# env-var contract for whichever arm is being built (assert lazily per encoder type)
with open(config_path) as fh:
    cfg = OmegaConf.create(yaml.safe_load(fh))
if sets:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sets))  # same typing rules as the trainer
fw_cfg = cfg.framework

needed = ["ROBOTWIN_DATA_ROOT"]
if str(fw_cfg.text.get("encoder_type", "bert")) == "smoothspike_bert":
    needed.append("SMOOTHSPIKE_MODEL_PATH")
if str(fw_cfg.vision.get("encoder_type", "dinov3")) == "sdtv3_19m":
    needed += ["SDTV3_WEIGHTS_PATH", "SDTV3_PROCESSOR_PATH"]
else:
    needed.append("DINOV3_MODEL_PATH")
for var in needed:
    assert os.environ.get(var), f"gate #2b requires env var {var} (same contract as training launch)"

torch.manual_seed(0)
fw = TurboVLAFramework(config=cfg)
core = fw.model.config
print(f"framework built: {type(fw.model).__name__}; processor {type(fw.image_processor).__name__} @ {fw.image_processor.size}")
print(f"text={core.text.encoder_type} T={core.text.timesteps} mask={core.text_mask_version} "
      f"vision={core.vision.encoder_type} interaction={core.interaction.cross_attention_type}")
print(f"views={fw.num_views} horizon={fw.action_horizon}")

fw.eval()
with torch.no_grad():
    images = [Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8)) for _ in range(3)]
    example = [{"image": images, "lang": "put the bowl on the plate", "state": [0.0] * 14}]
    out = fw.predict_action(example)
actions = np.asarray(out["normalized_actions"])
assert actions.shape == (1, 50, 14) and np.isfinite(actions).all()
print(f"predict_action: {actions.shape} finite — OK")
print("GATE #2b PASS")
