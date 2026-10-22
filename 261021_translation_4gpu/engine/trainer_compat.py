"""Checkpoint compatibility for the pinned single-GPU training workers.

Transformers 4.40.1 loads single-GPU optimizer state on CPU. PyTorch 2.0.1
does not move the ``step`` tensor back to the parameter device on load, even
for fused AdamW, whose CUDA kernel requires that device to match. That Trainer
version also omits Accelerate's FP16 GradScaler from checkpoint save/load.
Keep the optimizer and training settings intact; repair only resumed state.
"""
from pathlib import Path
import os
import tempfile

import torch
from transformers import Trainer, Seq2SeqTrainer


def repair_fused_adamw_step_devices(optimizer):
    """Move misplaced fused AdamW step counters without resetting any state."""
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        return 0
    moved = 0
    for group in optimizer.param_groups:
        if not group.get("fused", False):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if (parameter.device.type == "cuda" and torch.is_tensor(step)
                    and step.device != parameter.device):
                state["step"] = step.to(device=parameter.device)
                moved += 1
    return moved


class _ResumeCompatibilityMixin:
    def _active_grad_scaler(self):
        scaler = getattr(self.accelerator, "scaler", None)
        return scaler if scaler is not None and scaler.is_enabled() else None

    def _save_optimizer_and_scheduler(self, output_dir):
        super()._save_optimizer_and_scheduler(output_dir)
        scaler = self._active_grad_scaler()
        if not self.args.should_save or scaler is None:
            return
        directory = Path(output_dir)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=directory,
                                             prefix=".scaler-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(scaler.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "scaler.pt")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        self._resume_fused_steps_moved = 0
        self._resume_scaler_restored = False
        if checkpoint is None:
            return
        self._resume_fused_steps_moved = repair_fused_adamw_step_devices(self.optimizer)
        if self._resume_fused_steps_moved:
            print(f"[RESUME] Moved {self._resume_fused_steps_moved} fused AdamW step counters "
                  "to their parameter devices; values preserved", flush=True)
        scaler = self._active_grad_scaler()
        if scaler is not None:
            path = Path(checkpoint) / "scaler.pt"
            if not path.is_file():
                raise FileNotFoundError(f"FP16 resume requires saved GradScaler state: {path}")
            scaler.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self._resume_scaler_restored = True
            print("[RESUME] Restored FP16 GradScaler state", flush=True)


class ResumeSafeTrainer(_ResumeCompatibilityMixin, Trainer):
    """CLM Trainer with fused AdamW and FP16 checkpoint compatibility."""


class ResumeSafeSeq2SeqTrainer(_ResumeCompatibilityMixin, Seq2SeqTrainer):
    """Translation Trainer using the same checkpoint compatibility path."""
