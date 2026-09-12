"""§5.4 GPU verification: counting relationships on the real trainer, real data, real GPU.

Instrumentation around the *unmodified* training loop: wraps `_log_metrics`,
`eval_action_model` and `_save_checkpoint` to record (micro-batch, optimizer step, LR, loss),
then asserts the counting relationships after the run. Nothing in the training logic is
changed — only observations are added.

Verifies (by actual parameter-update count):
  * `completed_steps` advances once per `gradient_accumulation_steps` micro-batches;
  * LR advances only when `completed_steps` advances (the §5.4 fix) — every micro-batch
    inside one accumulation window logs the same LR;
  * the LR schedule peaks exactly at the warmup boundary in *optimizer steps*:
    `num_warmup_steps` (default 1000), not micro-batches;
  * eval/save fire exactly once per interval multiple.

Usage (compute server, inside the turbovla-robotwin env):
    # official-recipe run, crosses the 1000-step warmup boundary
    python scripts/robotwin/verify_54_counting.py \
        --config_yaml experiments/robotwin/configs/clean50.yaml \
        --max-steps 1100 --warmup 1000 --eval-interval 25 --save-interval 200

    # accumulation path (accum=4): crosses a 100-step warmup boundary cheaply
    python scripts/robotwin/verify_54_counting.py \
        --config_yaml experiments/robotwin/configs/clean50.yaml \
        --accum 4 --max-steps 480 --warmup 100 --eval-interval 25 --save-interval 200 \
        --run-id-suffix _accum4

Requires the same env vars as training (BERT_MODEL_PATH / DINOV3_MODEL_PATH /
TURBOVLA_INIT_CKPT / ROBOTWIN_DATA_ROOT, ...). The trace JSONL and the verdict are written
into the run directory as `54_verify_trace.jsonl` / `54_verify_report.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

root = "/data/260010028/dwh_vla/v4_code"
sys.path[:0] = [root, f"{root}/third_party/starvla_runtime", f"{root}/third_party/vla_adapter"]

from starVLA.training import train_starvla as base_train  # noqa: E402
from starVLA.training.train_robotwin_clean_act_pi05_recipe import (  # noqa: E402
    EMAVLATrainer,
    main as recipe_main,
)

TRACE: list[dict] = []
COUNTS = {"eval": 0, "save": 0, "log": 0}

_orig_log_metrics = base_train.VLATrainer._log_metrics
_orig_eval = base_train.VLATrainer.eval_action_model
_orig_save = EMAVLATrainer._save_checkpoint


def _log_metrics(self, metrics):
    COUNTS["log"] += 1
    TRACE.append(
        {
            "call": COUNTS["log"],
            "step": int(self.completed_steps),
            "lr": float(self.lr_scheduler.get_last_lr()[0]),
            "loss": float(metrics.get("action_dit_loss", float("nan"))),
        }
    )
    return _orig_log_metrics(self, metrics)


def _eval_action_model(self, step_metrics=None):
    COUNTS["eval"] += 1
    return _orig_eval(self, step_metrics)


def _save_checkpoint(self):
    COUNTS["save"] += 1
    return _orig_save(self)


def install():
    base_train.VLATrainer._log_metrics = _log_metrics
    base_train.VLATrainer.eval_action_model = _eval_action_model
    EMAVLATrainer._save_checkpoint = _save_checkpoint


def verify(run_dir: Path, args) -> dict:
    steps = [r["step"] for r in TRACE]
    lrs = [r["lr"] for r in TRACE]
    distinct_steps = sorted(set(steps))
    failures = []

    # 1. LR is constant within an accumulation window and advances only with completed_steps.
    by_step: dict[int, set[float]] = {}
    for r in TRACE:
        by_step.setdefault(r["step"], set()).add(r["lr"])
    multi_lr = {s: sorted(v) for s, v in by_step.items() if len(v) > 1}
    if multi_lr:
        failures.append(f"LR changed within a single optimizer step: {multi_lr}")

    # 2. Micro-batches per optimizer step == accumulation factor.
    if not args.accum_is_one:
        per_step = {s: steps.count(s) for s in distinct_steps}
        expected = args.accum
        bad = {s: c for s, c in per_step.items() if c != expected}
        if bad:
            failures.append(f"micro-batches per step != accum={expected}: {bad}")

    # 3. Warmup peaks exactly at the boundary, in optimizer steps.
    lr_by_step = {s: next(r["lr"] for r in TRACE if r["step"] == s) for s in distinct_steps}
    peak_step = max(lr_by_step, key=lambda s: lr_by_step[s])
    peak_lr = lr_by_step[peak_step]
    warm = args.warmup
    if warm in lr_by_step:
        if lr_by_step[warm] != peak_lr:
            failures.append(f"LR at warmup step {warm} ({lr_by_step[warm]}) is not the peak ({peak_lr})")
        if warm - 1 in lr_by_step and lr_by_step[warm - 1] >= peak_lr:
            failures.append(f"LR did not increase into the warmup boundary: {lr_by_step[warm - 1]} -> {lr_by_step[warm]}")
    elif peak_step < warm:
        failures.append(f"run ended at step {peak_step} before the warmup boundary {warm}")
    if peak_step > warm:
        failures.append(f"LR kept rising past the warmup boundary: peak at step {peak_step}, expected {warm}")

    # 4. Gate firing counts.
    expected_eval = len([s for s in distinct_steps if s % args.eval_interval == 0])
    expected_save = len([s for s in distinct_steps if s > 0 and s % args.save_interval == 0])
    if COUNTS["eval"] != expected_eval:
        failures.append(f"eval fired {COUNTS['eval']}x, expected {expected_eval}")
    if COUNTS["save"] != expected_save:
        failures.append(f"save fired {COUNTS['save']}x, expected {expected_save}")

    # 5. Checkpoints on disk match the save gate.
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("steps_*_model.safetensors"))
    ckpt_steps = sorted({int(n.split("_")[1]) for n in ckpts})
    if ckpt_steps != [s for s in distinct_steps if s > 0 and s % args.save_interval == 0]:
        failures.append(f"checkpoint steps {ckpt_steps} do not match save gate multiples")

    report = {
        "micro_batches": len(TRACE),
        "completed_steps": distinct_steps[-1] if distinct_steps else 0,
        "accum": args.accum,
        "warmup": warm,
        "peak_lr_step": peak_step,
        "peak_lr": peak_lr,
        "counts": dict(COUNTS),
        "expected": {"eval": expected_eval, "save": expected_save},
        "checkpoint_steps": ckpt_steps,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="experiments/robotwin/configs/clean50.yaml")
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1100)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--run-id", default=None, help="defaults to clean50_54verify_accum<N>_warm<W>")
    args = parser.parse_args()
    args.accum_is_one = args.accum == 1
    run_id = args.run_id or f"clean50_54verify_accum{args.accum}_warm{args.warmup}"

    install()
    overrides = [
        f"run_id={run_id}",
        f"trainer.max_train_steps={args.max_steps}",
        f"trainer.num_warmup_steps={args.warmup}",
        f"trainer.eval_interval={args.eval_interval}",
        f"trainer.save_interval={args.save_interval}",
        "trainer.logging_frequency=1",
        f"trainer.gradient_accumulation_steps={args.accum}",
    ]
    sys.argv = [sys.argv[0], "--config_yaml", args.config_yaml, *overrides]

    recipe_main()

    # Mirror the trainer's own path resolution: run_root_dir / run_id (cwd-relative).
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config_yaml)
    run_dir = Path(str(OmegaConf.select(cfg, "run_root_dir", default="results/Checkpoints"))) / run_id
    run_dir = run_dir if run_dir.is_absolute() else Path(root) / run_dir

    (run_dir / "54_verify_trace.jsonl").write_text("\n".join(json.dumps(r) for r in TRACE) + "\n")
    report = verify(run_dir, args)
    (run_dir / "54_verify_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"§5.4 VERIFY {report['verdict']} — report: {run_dir / '54_verify_report.json'}")
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
