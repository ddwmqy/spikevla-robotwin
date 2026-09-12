"""§5.4 verification (CPU): optimizer-step ↔ scheduler ↔ EMA ↔ save/eval gate counting.

Drives the real `EMAVLATrainer.train()` control flow and the real `_train_step`
implementations against a tiny dummy module, on CPU, with no DeepSpeed. The I/O-heavy
methods (checkpointing, eval, logging, data loading) are stubbed so that only the
counting relationships are under test.

Expected, by actual parameter-update count:
  - completed_steps          == micro_batches / accumulator
  - scheduler.last_epoch     == completed_steps        (the §5.4 fix)
  - EMA updates              == completed_steps        (already guarded upstream)
  - eval fires               == #multiples of eval_interval reached, each exactly once
  - save fires               == #multiples of save_interval > 0, each exactly once
At accumulator == 1 (the official RoboTwin config) the fix is a no-op, so all counters
must reduce to the pre-fix semantics: one scheduler step and one gate check per micro-batch.
"""
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

root = "/data/260010028/dwh_vla/v4_code"
sys.path[:0] = [root, f"{root}/third_party/starvla_runtime", f"{root}/third_party/vla_adapter"]

# Stub wandb: the pod's v3 env has a protobuf/wandb version skew unrelated to this test,
# and the trainer only touches wandb behind `self.wandb_enabled` (stubbed off below).
_wandb_stub = types.ModuleType("wandb")
_wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)  # accelerate probes find_spec
_wandb_stub.init = MagicMock()
_wandb_stub.log = MagicMock()
sys.modules["wandb"] = _wandb_stub

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from starVLA.training import train_starvla as base_train  # noqa: E402
from starVLA.training.train_robotwin_clean_act_pi05_recipe import EMAVLATrainer  # noqa: E402

# Silence the progress bar so the counting output stays readable.
base_train.tqdm = lambda **kwargs: types.SimpleNamespace(update=lambda *a: None, set_postfix=lambda *a: None)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))

    def forward(self, batch):
        return {"action_loss": (self.weight**2).sum() + 1.0}


def run(accum: int, max_steps: int = 20, eval_interval: int = 5, save_interval: int = 10):
    accelerator = Accelerator(gradient_accumulation_steps=accum, cpu=True)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / 10))
    model, optimizer = accelerator.prepare(model, optimizer)

    trainer = object.__new__(EMAVLATrainer)
    trainer.accelerator = accelerator
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "max_train_steps": max_steps,
                "eval_interval": eval_interval,
                "save_interval": save_interval,
                "logging_frequency": 10**9,  # silence logging path
                "gradient_clipping": None,
                "ema_decay": 0.999,
                "ema_device": "cpu",
            },
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
        }
    )
    trainer.completed_steps = 0
    trainer.vla_train_dataloader = list(range(10**6))  # only len() is used (logging)
    trainer.ema_decay = 0.999
    raw = getattr(model, "module", model)  # DDP-wrapped only under distributed
    trainer.ema_state = {"weight": raw.weight.detach().float().clone()}

    counters = {"ema": 0, "eval": 0, "save": 0, "micro": 0}

    # Count around the real EMA update (real math, real sync_gradients guard in _train_step).
    real_update_ema = EMAVLATrainer._update_ema

    def counting_update_ema():
        counters["ema"] += 1
        real_update_ema(trainer)

    trainer._update_ema = counting_update_ema

    # _train_step is deliberately NOT stubbed: the real EMAVLATrainer._train_step
    # (base body with the §5.4 scheduler guard + EMA hook) is what is under test.
    trainer._log_training_config = lambda: None
    trainer._create_data_iterators = lambda: None
    trainer._log_metrics = lambda metrics: None
    trainer._finalize_training = lambda: None

    def _get_next_batch():
        counters["micro"] += 1
        return None

    trainer._get_next_batch = _get_next_batch

    def _eval(metrics=None):
        counters["eval"] += 1
        return metrics if metrics is not None else {}

    trainer.eval_action_model = _eval

    def _save():
        counters["save"] += 1

    trainer._save_checkpoint = _save

    trainer.train()

    return {
        "accum": accum,
        "completed_steps": trainer.completed_steps,
        "scheduler_last_epoch": scheduler.last_epoch,
        **counters,
    }


def main() -> None:
    for accum in (1, 4):
        r = run(accum)
        print(r)
        micro, steps = r["micro"], r["completed_steps"]
        assert micro == 20 * accum, f"accum={accum}: micro-batches {micro}"
        assert steps == 20, f"accum={accum}: completed_steps {steps}"
        assert r["scheduler_last_epoch"] == steps, f"accum={accum}: scheduler {r['scheduler_last_epoch']} != {steps}"
        assert r["ema"] == steps, f"accum={accum}: EMA {r['ema']} != {steps}"
        # 20 micro-steps: multiples of 5 → 5,10,15,20 (4 fires); of 10 → 10,20 (2 fires)
        assert r["eval"] == 4, f"accum={accum}: eval {r['eval']} != 4"
        assert r["save"] == 2, f"accum={accum}: save {r['save']} != 2"
    print("GATE #5 PASS — optimizer-step ↔ scheduler ↔ EMA ↔ gates counted consistently at accum=1 and 4")


if __name__ == "__main__":
    main()
