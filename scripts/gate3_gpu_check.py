"""P0A gate #3: real-GPU forward for the A arm (compute-server side).

Runs the A-arm spike model on CUDA with a fixed input batch in fp32 and in the
training precision (bf16 autocast), and prints a compact digest (shapes, finiteness,
moment statistics, tensor checksum) to paste back into ASSETS.md. CPU-side gates #0-#2
must already have passed; this script adds the GPU numerics the pod cannot produce.

Usage (inside the turbovla-robotwin env, on a GPU machine):
    python scripts/gate3_gpu_check.py            # fp32 + bf16 autocast
    python scripts/gate3_gpu_check.py --fp32-only
"""
import argparse
import hashlib
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path.insert(0, root)
sys.meta_path[:] = [m for m in sys.meta_path if "editable" not in type(m).__name__.lower()]

import torch  # noqa: E402

from turbovla.models import TurboVLAConfig, build_turbovla  # noqa: E402
from turbovla.models.configuration import (  # noqa: E402
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)

RES = "/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained"
INSTRUCTIONS = [
    "put both the moka pot and the yellow mug on the stove",
    "place the alphabet soup in the basket",
]


def make_config() -> TurboVLAConfig:
    return TurboVLAConfig(
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


def fixed_batch(device: torch.device):
    g = torch.Generator().manual_seed(1)
    return (
        INSTRUCTIONS,
        {"pixel_values": torch.rand(2, 3, 3, 224, 224, generator=g).to(device)},
        torch.rand(2, 14, generator=g).to(device),
    )


def digest(out: torch.Tensor) -> dict:
    flat = out.detach().float().cpu().flatten()
    return {
        "shape": list(out.shape),
        "finite": bool(torch.isfinite(out).all()),
        "mean": round(flat.mean().item(), 8),
        "std": round(flat.std().item(), 8),
        "sha1": hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-only", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "gate #3 requires a CUDA device"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)
    model = build_turbovla(make_config()).to(device)
    model.eval()

    results = {}
    with torch.no_grad():
        torch.manual_seed(1)
        out = model(*fixed_batch(device))
        results["fp32"] = digest(out)
        print(f"fp32  {results['fp32']}")

        if not args.fp32_only:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(1)
                out_bf16 = model(*fixed_batch(device))
            results["bf16_autocast"] = digest(out_bf16)
            print(f"bf16  {results['bf16_autocast']}")
            gap = (out_bf16.detach().float() - out.detach().float()).abs().max().item()
            print(f"fp32↔bf16 max|Δ| = {gap:.6f}")
            results["fp32_bf16_maxdiff"] = gap

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak CUDA memory: {peak:.2f} GiB")
    results["peak_gib"] = round(peak, 2)

    assert all(r["finite"] for r in results.values() if isinstance(r, dict)), "non-finite outputs on GPU"
    print("GATE #3 PASS")


if __name__ == "__main__":
    main()
