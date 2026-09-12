"""§5.2 guard check for the B arm: the initialization load must NOT touch the spike text backbone.

B trains the official architecture with the text encoder swapped for SmoothSpike (frozen), and
is initialized from an official checkpoint that also carries plain-BERT tensors. With
`load_bert: true` those tensors would silently overwrite the spike weights (the wrapper's
loader skips only shape-mismatched keys). This script builds the B arm *through the real
wrapper* (so the actual `_load_initialization` code path runs) and asserts that every
SmoothSpike text tensor is still bitwise identical to the fused checkpoint.

Usage (compute server, after the arm env vars are exported as for training):
    python scripts/check_b_init_hash.py                     # expects load_bert: false -> PASS
    python scripts/check_b_init_hash.py --expect-overwrite   # load_bert: true -> documents the hazard

The repo's 55k official checkpoint works as the init source for this check because it carries
both `bert.*` and `feat_map.*` keys; the shapes need not match GroundingDINO's for the
*mechanism* to be exercised.
"""
import argparse
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path[:0] = [root, f"{root}/third_party/starvla_runtime", f"{root}/third_party/vla_adapter"]

import torch  # noqa: E402
import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=f"{root}/experiments/robotwin/configs/clean50_b.yaml")
parser.add_argument("--load-bert", choices=["false", "true"], default=None)
parser.add_argument("--expect-overwrite", action="store_true", help="assert the *hazard* instead (documents it)")
parser.add_argument(
    "--synthetic-init",
    action="store_true",
    help="fabricate an init checkpoint in the init-source naming (bert.* / feat_map.* / "
    "transformer.encoder.*) from the model's own perturbed weights, so the guard can be "
    "exercised without waiting for the real initialization checkpoint to download",
)
args = parser.parse_args()

with open(args.config) as fh:
    cfg = OmegaConf.create(yaml.safe_load(fh))
if args.load_bert is not None:
    cfg.framework.initialization.load_bert = args.load_bert == "true"
print(f"config={args.config}")
print(f"load_bert={bool(cfg.framework.initialization.load_bert)} "
      f"load_text_projection={bool(cfg.framework.initialization.load_text_projection)}")

if args.synthetic_init:
    import tempfile
    from pathlib import Path

    probe_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    probe_cfg.framework.initialization.load_pretrained = False
    probe_cfg.framework.initialization.pretrained_ckpt = ""
    torch.manual_seed(123)
    probe = TurboVLAFramework(config=probe_cfg)

    prefixes = [
        ("text_encoder.bert.", "bert."),
        ("text_encoder.text_projection.", "feat_map."),
        ("vision_language_interaction.text_layers.", "transformer.encoder.text_layers."),
        ("vision_language_interaction.fusion_layers.", "transformer.encoder.fusion_layers."),
    ]
    synthetic = {}
    for key, value in probe.model.state_dict().items():
        for target_prefix, source_prefix in prefixes:
            if key.startswith(target_prefix):
                # perturb so that any load is detectable, while keeping the shapes valid
                synthetic[source_prefix + key[len(target_prefix):]] = (
                    value.float() + torch.randn_like(value.float()) * 1e-3
                ).to(value.dtype)
                break
    assert synthetic, "failed to synthesise an init checkpoint"
    tmp_path = Path(tempfile.mkdtemp()) / "synthetic_init.pth"
    torch.save({"model": synthetic, "args": argparse.Namespace(note="synthetic")}, tmp_path)
    cfg.framework.initialization.pretrained_ckpt = str(tmp_path)
    print(f"synthetic init: {len(synthetic)} tensors -> {tmp_path}")

torch.manual_seed(0)
fw = TurboVLAFramework(config=cfg)

fused_path = cfg.framework.text.bert_path
fused = load_file(f"{fused_path}/model.safetensors", device="cpu")
state = fw.model.state_dict()

mismatched, matched = [], 0
for key, src in fused.items():
    if not key.startswith("bert."):
        continue  # MLM head is discarded by the loader by design
    target_key = f"text_encoder.bert.{key[len('bert.'):]}"
    if target_key not in state:
        mismatched.append(target_key)
        continue
    if torch.equal(state[target_key], src):
        matched += 1
    else:
        delta = (state[target_key].float() - src.float()).abs().max().item()
        mismatched.append(f"{target_key} (max|Δ|={delta:.3e})")

print(f"spike text backbone tensors bitwise-equal: {matched}")
print(f"overwritten / missing: {len(mismatched)}")
for name in mismatched[:8]:
    print(f"  - {name}")

if args.expect_overwrite:
    ok = len(mismatched) > 0
    print("HAZARD CONFIRMED: an init checkpoint's BERT tensors DO reach the spike encoder" if ok
          else "UNEXPECTED: nothing was overwritten — is the init checkpoint carrying bert.* keys?")
else:
    ok = not mismatched
    print("§5.2 GUARD PASS — the init load left the spike text backbone untouched" if ok
          else "§5.2 GUARD FAIL — the spike text backbone was overwritten by the init checkpoint")
sys.exit(0 if ok else 1)
