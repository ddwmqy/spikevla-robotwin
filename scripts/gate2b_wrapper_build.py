"""P0A gate #2b: full A-arm wrapper construction from clean50_a.yaml (CPU-capable).

End-to-end construction check: yaml → merge_framework_config → TurboVLAFramework →
core A model (SmoothSpike text + SDT-V3 vision + Spike2Max fusion) → image processor
→ predict_action. Catches config-validation bugs (e.g. wrong attn_implementation)
that direct-config gates #1/#2 bypass, and verifies the env-var contract of the yaml.
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

for var in ("SMOOTHSPIKE_MODEL_PATH", "SDTV3_WEIGHTS_PATH", "SDTV3_PROCESSOR_PATH", "ROBOTWIN_DATA_ROOT"):
    assert os.environ.get(var), f"gate #2b requires env var {var} (same contract as training launch)"

with open(f"{root}/experiments/robotwin/configs/clean50_a.yaml") as fh:
    cfg = OmegaConf.create(yaml.safe_load(fh))

torch.manual_seed(0)
fw = TurboVLAFramework(config=cfg)
core = fw.model.config
assert core.text.encoder_type == "smoothspike_bert" and core.text.timesteps == 4
assert core.text_mask_version == "corrected" and core.text.attention_implementation == "eager"
assert core.vision.encoder_type == "sdtv3_19m" and core.vision.num_views == 3
assert core.interaction.cross_attention_type == "spike_sdsa" and core.interaction.cross_timesteps == 4
print(f"framework built: {type(fw.model).__name__}; processor {type(fw.image_processor).__name__} @ {fw.image_processor.size}")
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
