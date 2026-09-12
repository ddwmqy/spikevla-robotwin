"""§5.3 fixed-input-batch precision matrix (levels 1-2 of the check flow).

Level 1 (数值检查 A): on the initialized arm, compare TEXT features, VISUAL features, FUSION
outputs and ACTIONS across precision cells against a high-precision reference, on one fixed
batch of real inputs. Level 2: report the action-error metrics in the normalized domain —
joint MAE / max error and gripper sign-agreement at the 0.49 threshold (the double-caliber
gripper metric needs the ensemble/threshold split of the deployment path and is therefore
part of the deployment check, not this one).

Every cell runs in its own process: precision switches mutate module dtypes and global torch
state, so in-process sweeps would contaminate each other.

Three stages (all take the same --out directory):
    python scripts/precision_matrix.py prepare --arm b --out results/PrecisionMatrix/b_<date>
    python scripts/precision_matrix.py cell    --arm b --out ... --cell ref
    python scripts/precision_matrix.py compare --out ...

`prepare` samples the fixed batch from the real clean50 dataset once and stores it, so every
cell sees bit-identical inputs. The model is built in eval mode with a fixed seed; for the B/C1
arms the real initialization checkpoint is loaded (load happens in fp32, identically in every
cell), so the compared weights are exactly the arm's real starting point.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

root = "/data/260010028/dwh_vla/v4_code"
sys.path[:0] = [root, f"{root}/third_party/starvla_runtime", f"{root}/third_party/vla_adapter"]

ARMS = {
    "c1": f"{root}/experiments/robotwin/configs/clean50.yaml",
    "b": f"{root}/experiments/robotwin/configs/clean50_b.yaml",
    "a": f"{root}/experiments/robotwin/configs/clean50_a.yaml",
}

# Grid cells: (name, matmul precision, interaction precision, vision precision, attention backend).
# `None` for matmul means "leave torch's default" (recorded in the report either way).
#
# Attention backend: the official config uses flash_attention_2, but flash-attn rejects fp32
# inputs outright — so an fp32 reference is *impossible* with the official backend. The matrix
# therefore runs on sdpa (uniform across cells, so the precision comparison is clean), plus one
# `official_flash` cell that reproduces the real training configuration; the gap between
# `official` and `official_flash` isolates the backend's own numerical effect.
CELLS = [
    ("ref", "highest", "fp32", "fp32", "sdpa"),
    ("official", None, "bf16_autocast", "bf16_autocast", "sdpa"),
    ("official_flash", None, "bf16_autocast", "bf16_autocast", "flash_attention_2"),
    ("ann_bf16_vis_fp32", None, "bf16_autocast", "fp32", "sdpa"),
    ("ann_bf16_vis_bf16", None, "bf16_autocast", "bf16", "sdpa"),
    ("fp32_matmul_high", "high", "fp32", "fp32", "sdpa"),
    ("fp32_matmul_medium", "medium", "fp32", "fp32", "sdpa"),
    ("bf16_matmul_highest", "highest", "bf16_autocast", "bf16_autocast", "sdpa"),
]
DEFAULT_CELL = "ref"
N_SAMPLES = 2


def _build_model(arm: str, cell: str, overrides: list[str], device: str):
    import torch
    import yaml
    from omegaconf import OmegaConf

    from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework

    name, matmul, inter_p, vis_p, attn = next(c for c in CELLS if c[0] == cell)
    if matmul is not None:
        torch.set_float32_matmul_precision(matmul)

    with open(ARMS[arm]) as fh:
        cfg = OmegaConf.create(yaml.safe_load(fh))
    cfg.framework.interaction.compute_precision = inter_p
    cfg.framework.vision.compute_precision = vis_p
    cfg.framework.vision.attn_implementation = attn
    # SmoothSpike requires eager attention (enforced by the config); other text encoders follow
    # the cell backend so the text path is not silently held at a different precision regime.
    if str(cfg.framework.text.get("encoder_type", "bert")) == "bert":
        cfg.framework.text.attn_implementation = attn
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    torch.manual_seed(0)
    fw = TurboVLAFramework(config=cfg)
    fw.model.eval().to(device)
    return fw, {"cell": name, "matmul_precision": torch.get_float32_matmul_precision(),
                "interaction_compute_precision": inter_p, "vision_compute_precision": vis_p,
                "attention_backend": attn,
                "text_attention_backend": str(cfg.framework.text.get("attn_implementation"))}


def stage_prepare(args) -> None:
    """Fix one real input batch from the dataset (shared by every cell)."""
    import torch
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    with open(ARMS[args.arm]) as fh:
        cfg = OmegaConf.create(yaml.safe_load(fh))
    ds = get_vla_dataset(cfg.datasets.vla_data, mode="train")
    torch.manual_seed(7)
    indices = torch.randint(0, len(ds), (N_SAMPLES,)).tolist()
    samples = [ds[i] for i in indices]

    import numpy as np

    def to_tensor(im):
        arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    images = torch.stack(
        [torch.stack([to_tensor(im) for im in s["image"]]) for s in samples]
    )  # [B, V, 3, H, W]
    payload = {
        "instructions": [str(s["lang"]) for s in samples],
        "pixel_values": images,
        "states": torch.as_tensor(np.asarray([s["state"] for s in samples])).squeeze(1).float(),
        "dataset_indices": indices,
    }
    torch.save(payload, out / "inputs.pt")
    print(f"fixed batch: {tuple(images.shape)} from dataset indices {indices} -> {out / 'inputs.pt'}")


def stage_cell(args) -> None:
    import torch

    out = Path(args.out)
    device = torch.device(args.device)
    payload = torch.load(out / "inputs.pt", map_location="cpu", weights_only=False)
    fw, meta = _build_model(args.arm, args.cell, args.set, args.device)

    instructions = payload["instructions"]
    pixels = payload["pixel_values"].to(device)
    states = payload["states"].to(device)
    with torch.no_grad():
        text_raw, _, _ = fw.model.text_encoder(instructions, device=device)
        actions, feats = fw.model(
            instructions, {"pixel_values": pixels}, states, return_distillation_features=True
        )
        condition = fw.model.encode_condition(instructions, {"pixel_values": pixels})

    torch.save(
        {
            "text_raw": text_raw.float().cpu(),
            "text_proj": feats["text"].float().cpu(),
            "visual_proj": feats["visual"].float().cpu(),
            "condition": condition.float().cpu(),
            "actions": actions.float().cpu(),
            "meta": meta,
        },
        out / f"cell_{args.cell}.pt",
    )
    print(f"cell {args.cell}: text_raw{tuple(text_raw.shape)} visual{tuple(feats['visual'].shape)} "
          f"condition{tuple(condition.shape)} actions{tuple(actions.shape)}")


def stage_compare(args) -> None:
    import torch

    out = Path(args.out)
    cell_files = {p.stem.replace("cell_", ""): p for p in sorted(out.glob("cell_*.pt"))}
    ref_name = args.ref
    ref = torch.load(cell_files[ref_name], map_location="cpu", weights_only=False)
    meta = {"arm": args.arm, "reference_cell": ref_name, "input": str(out / "inputs.pt"), "cells": {}}

    rows = []
    for name, path in cell_files.items():
        cell = torch.load(path, map_location="cpu", weights_only=False)
        entry = {"precision": cell["meta"]}
        for key in ("text_raw", "text_proj", "visual_proj", "condition", "actions"):
            d = (cell[key] - ref[key]).abs()
            entry[key] = {"max_abs": d.max().item(), "mae": d.mean().item()}
        act, ref_act = cell["actions"], ref["actions"]
        # joints = first 12 dims (min-max normalized), grippers = last 2 (binary decision)
        jd = (act[..., :12] - ref_act[..., :12]).abs()
        entry["joint"] = {"mae": jd.mean().item(), "max_abs": jd.max().item()}
        g_cell, g_ref = act[..., 12:], ref_act[..., 12:]
        entry["gripper_raw_agreement_049"] = (g_cell >= 0.49).eq(g_ref >= 0.49).float().mean().item()
        entry["gripper_final_agreement_050"] = (
            (g_cell >= 0.5).eq(g_ref >= 0.5).float().mean().item()
        )
        meta["cells"][name] = entry
        rows.append((name, entry))

    (out / "report.json").write_text(json.dumps(meta, indent=2) + "\n")
    hdr = f"{'cell':22s} {'text_raw':>10s} {'text_proj':>10s} {'visual':>10s} {'fusion':>10s} {'action':>10s} {'jointMAE':>9s} {'grip.49':>8s} {'grip.50':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, e in rows:
        print(f"{name:22s} {e['text_raw']['max_abs']:10.5f} {e['text_proj']['max_abs']:10.5f} "
              f"{e['visual_proj']['max_abs']:10.5f} {e['condition']['max_abs']:10.5f} "
              f"{e['actions']['max_abs']:10.5f} {e['joint']['mae']:9.5f} "
              f"{e['gripper_raw_agreement_049']:8.4f} {e['gripper_final_agreement_050']:8.4f}")
    print(f"\n(all values are deltas vs the `{ref_name}` reference, element-wise max|Δ| unless noted)")
    print(f"report: {out / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "cell", "compare", "sweep"])
    parser.add_argument("--arm", choices=sorted(ARMS), default="b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell", default=DEFAULT_CELL)
    parser.add_argument("--ref", default="ref")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", action="append", default=[], help="extra override (with or without --)")
    args = parser.parse_args()
    args.set = [s if s.startswith("--") else f"--{s}" for s in args.set]

    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "cell":
        stage_cell(args)
    elif args.stage == "compare":
        stage_compare(args)
    else:  # sweep: prepare + one subprocess per cell + compare
        stage_prepare(args)
        for name, *_ in CELLS:
            cmd = [sys.executable, __file__, "cell", "--arm", args.arm, "--out", args.out,
                   "--cell", name, "--device", args.device, *args.set]
            print(f"--- cell {name} ---", flush=True)
            subprocess.run(cmd, check=True)
        stage_compare(args)


if __name__ == "__main__":
    main()
