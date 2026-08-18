#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command runner for 260818 full-region-classification experiment
with 4 tokenizers × 5 seeds on a single A6000 (or similar) GPU.

Execution example:
  cd /path/to/repo/260818_4tokenizer_5seed_region_classification
  python run_260818_region_one_shot.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_seed_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def get_supported_options(script_path: Path) -> set[str]:
    """
    Parse argparse option names from script help text.
    """
    try:
        out = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (out.stdout or "") + (out.stderr or "")
    except Exception:
        return set()
    opts = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", text)
    return set(opt.lower() for opt in opts)


def query_process_gpu(pid: int, gpu_id: int) -> int | None:
    """
    Return used GPU memory for a process in MiB. Returns None if process not found.
    """
    cmd = [
        "nvidia-smi",
        f"-i",
        str(gpu_id),
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=3
        )
        if out.returncode != 0 or not out.stdout:
            return None
        for line in out.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if not cols or len(cols) < 2:
                continue
            if cols[0] == str(pid):
                mem = re.sub(r"\s*MiB?$", "", cols[1], flags=re.IGNORECASE).strip()
                try:
                    return int(float(mem))
                except ValueError:
                    return None
        return None
    except Exception:
        return None


def query_gpu_util(gpu_id: int) -> int | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_id),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=3
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(float(out.stdout.strip().splitlines()[0].strip()))
    except Exception:
        return None


class GPUMonitor(threading.Thread):
    def __init__(self, pid: int, gpu_id: int, log_path: Path, interval: float = 5.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.gpu_id = gpu_id
        self.log_path = log_path
        self.interval = interval
        self._stop = threading.Event()
        self.peak_mem = 0

    def run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "pid", "process_vram_MiB", "gpu_util_percent"])
            while not self._stop.is_set():
                mem = query_process_gpu(self.pid, self.gpu_id)
                util = query_gpu_util(self.gpu_id)
                if mem is not None:
                    self.peak_mem = max(self.peak_mem, mem)
                writer.writerow([ts(), self.pid, mem if mem is not None else "", util if util is not None else ""])
                f.flush()
                self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def read_tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-max_lines:]
    return "".join(lines)


def run_command(
    cmd: List[str], log_path: Path, gpu_log_path: Path, gpu_id: int, stream_logs: bool
) -> Tuple[int, str, int]:
    """
    Run subprocess and return (returncode, tail_log, peak_vram_mib).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    monitor = GPUMonitor(proc.pid, gpu_id, gpu_log_path)
    monitor.start()

    last_lines = []
    max_keep = 200
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[{ts()}] CMD: {' '.join(cmd)}\n")
        lf.flush()
        for line in proc.stdout:
            lf.write(line)
            if stream_logs:
                print(line.rstrip())
            last_lines.append(line)
            if len(last_lines) > max_keep:
                last_lines = last_lines[-max_keep:]

    rc = proc.wait()
    monitor.stop()
    monitor.join(timeout=3)

    return rc, "".join(last_lines), monitor.peak_mem


def build_cmd(script_path: Path, args: Dict[str, object], supported: set[str]) -> List[str]:
    cmd = [sys.executable, str(script_path)]
    for k, v in args.items():
        if isinstance(v, bool):
            candidates = [f"--{k}", f"--{k.replace('_', '-')}"]
            name = next((c for c in candidates if c.lower() in supported), None)
            if v and name is not None:
                cmd.append(name)
        else:
            candidates = [f"--{k}", f"--{k.replace('_', '-')}"]
            name = next((c for c in candidates if c.lower() in supported), None)
            if name is None:
                continue
            cmd.extend([name, str(v)])
    return cmd


def run_stage(
    stage_name: str,
    script_path: Path,
    supported_options: set[str],
    cmd_base: Dict[str, object],
    log_dir: Path,
    gpu_id: int,
    stream_logs: bool,
) -> Tuple[bool, int, int]:
    """
    Run one stage. Auto-retry only on OOM by halving batch sizes.
    """
    batch_key = "train_batch_size"
    eval_key = "eval_batch_size"
    train_bs = int(cmd_base[batch_key])
    eval_bs = int(cmd_base[eval_key])

    max_retry = 4
    for attempt in range(max_retry):
        cmd = build_cmd(script_path, cmd_base, supported_options)
        log_path = log_dir / f"{stage_name}_a{attempt+1}.log"
        gpu_log = log_dir / f"{stage_name}_a{attempt+1}_gpu.csv"
        print(f"\n[{ts()}] {stage_name.upper()} start | attempt={attempt+1} | cmd={cmd_base['tokenizer_mode']} | train_bs={train_bs}, eval_bs={eval_bs}")
        rc, tail, peak_mem = run_command(cmd, log_path, gpu_log, gpu_id, stream_logs)
        if rc == 0:
            return True, attempt + 1, peak_mem

        oom = "out of memory" in tail.lower()
        if oom and attempt < max_retry - 1:
            # aggressive -> safety fallback
            train_bs = max(128, train_bs // 2)
            eval_bs = max(256, eval_bs // 2)
            cmd_base[batch_key] = train_bs
            cmd_base[eval_key] = eval_bs
            print(f"[{ts()}] OOM detected. retry with reduced batch: train={train_bs}, eval={eval_bs}")
            continue

        print(f"[{ts()}] {stage_name.upper()} failed. See {log_path}\n--- tail ---\n{tail}")
        return False, attempt + 1, peak_mem

    return False, max_retry, 0


def build_stage_args(
    tokenizer: str,
    seed: int,
    stage: str,
    output_dir: Path,
    use_existing_tsv: bool,
    train_tsv: str,
    eval_tsv: str,
    common: Dict[str, object],
) -> Dict[str, object]:
    if stage == "mlm":
        cmd = {
            "tokenizer_mode": tokenizer,
            "seed": seed,
            "max_length": common["max_length"],
            "train_batch_size": common["mlm_train_batch"],
            "eval_batch_size": common["mlm_eval_batch"],
            "preprocessing_num_workers": common["preprocessing_num_workers"],
            "dataloader_num_workers": common["dataloader_num_workers"],
            "fp16": True,
            "num_train_epochs": common["mlm_epochs"],
            "learning_rate": common["mlm_lr"],
            "warmup_ratio": common["mlm_warmup_ratio"],
            "weight_decay": common["weight_decay"],
            "gradient_accumulation_steps": common["gradient_accumulation_steps"],
            "output_dir": output_dir / "mlm",
            "overwrite_output_dir": True,
        }
    else:
        cmd = {
            "tokenizer_mode": tokenizer,
            "seed": seed,
            "max_length": common["max_length"],
            "train_batch_size": common["cls_train_batch"],
            "eval_batch_size": common["cls_eval_batch"],
            "preprocessing_num_workers": common["preprocessing_num_workers"],
            "dataloader_num_workers": common["dataloader_num_workers"],
            "fp16": True,
            "num_train_epochs": common["cls_epochs"],
            "learning_rate": common["cls_lr"],
            "warmup_ratio": common["cls_warmup_ratio"],
            "weight_decay": common["weight_decay"],
            "gradient_accumulation_steps": common["gradient_accumulation_steps"],
            "mlm_model_dir": str(output_dir / "mlm" / "final_model"),
            "output_dir": output_dir / "classifier",
            "overwrite_output_dir": True,
            "use_existing_tsv": use_existing_tsv,
            "train_tsv": train_tsv,
            "eval_tsv": eval_tsv,
        }
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code_root",
        default="",
        help="Directory containing step1_continue_mlm_pretrain.py and step2_finetune_region_classifier.py. If omitted, infer from script location.",
    )
    parser.add_argument(
        "--experiment_root",
        default="",
        help="Where logs, checkpoints, summaries will be saved. If omitted: <repo_root>/260818_4tokenizer_5seed_region_classification_a6000",
    )
    parser.add_argument(
        "--tokenizers",
        default="dialect,klue,kobert,mbert",
        help="Comma-separated tokenizer modes",
    )
    parser.add_argument(
        "--seeds",
        default="13,21,42,87,100",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--train_tsv",
        default="",
        help="Train TSV for region classification. If omitted, use <repo_root>/260630_test_1/region_classification_data/dialect_region_train.tsv",
    )
    parser.add_argument(
        "--eval_tsv",
        default="",
        help="Evaluation TSV for region classification. If omitted, use <repo_root>/260630_test_1/region_classification_data/dialect_region_eval.tsv",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--stream_logs", action="store_true", default=True)
    parser.add_argument("--no_stream_logs", dest="stream_logs", action="store_false")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    # Resolve roots in an OS-agnostic way
    if args.code_root:
        code_root = Path(args.code_root).resolve()
    else:
        # current directory first
        code_root = Path(__file__).resolve().parent

    # If launched from 260818_* directory, step1/step2 are typically in 260807_* sibling.
    if not (code_root / "step1_continue_mlm_pretrain.py").exists():
        fallback = code_root.parent / "260807_4tokenizer_5seed_region_classification"
        if (
            (fallback / "step1_continue_mlm_pretrain.py").exists()
            and (fallback / "step2_finetune_region_classifier.py").exists()
        ):
            code_root = fallback

    # infer repo root (sibling dirs like 260630_test_1 expected)
    repo_root = code_root.parent if code_root.name.startswith("2608") else code_root

    if args.experiment_root:
        exp_root = Path(args.experiment_root).resolve()
    else:
        exp_root = repo_root / "260818_4tokenizer_5seed_region_classification_a6000"

    if args.train_tsv:
        train_tsv = str(Path(args.train_tsv).resolve())
    else:
        train_tsv = str(
            (repo_root / "260630_test_1" / "region_classification_data" / "dialect_region_train.tsv").resolve()
        )

    if args.eval_tsv:
        eval_tsv = str(Path(args.eval_tsv).resolve())
    else:
        eval_tsv = str(
            (repo_root / "260630_test_1" / "region_classification_data" / "dialect_region_eval.tsv").resolve()
        )

    exp_root.mkdir(parents=True, exist_ok=True)

    tokenizers = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    seeds = parse_seed_list(args.seeds)

    # A6000 aggressive config
    common: Dict[str, object] = {
        "max_length": 128,
        "mlm_epochs": 3,
        "mlm_train_batch": 1024,
        "mlm_eval_batch": 2048,
        "mlm_lr": 5e-5,
        "mlm_warmup_ratio": 0.06,
        "cls_epochs": 3,
        "cls_train_batch": 2048,
        "cls_eval_batch": 4096,
        "cls_lr": 2e-5,
        "cls_warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 16,
        "gradient_accumulation_steps": 1,
    }

    # Keep process-to-process environment isolated in our script
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    mlm_script = code_root / "step1_continue_mlm_pretrain.py"
    cls_script = code_root / "step2_finetune_region_classifier.py"

    for p in (mlm_script, cls_script):
        if not p.exists():
            print(f"ERROR: missing script: {p}")
            return 2

    supported_mlm = get_supported_options(mlm_script)
    supported_cls = get_supported_options(cls_script)
    if not supported_mlm or not supported_cls:
        print(f"[{ts()}] WARN: help parsing failed for one script. Proceeding with filtered args may be limited.")

    summary_path = exp_root / "results" / "run_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []

    total_runs = len(tokenizers) * len(seeds)
    run_no = 0

    for tok in tokenizers:
        for seed in seeds:
            run_no += 1
            seed_dir = exp_root / tok / f"seed_{seed}"
            logs_dir = seed_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            mlm_out = seed_dir
            final_mlm = mlm_out / "mlm" / "final_model"
            final_cls = mlm_out / "classifier" / "final_model"

            print(f"\n[{ts()}] ===== RUN {run_no}/{total_runs} | tokenizer={tok} seed={seed} =====")
            if args.skip_existing and final_cls.exists():
                print(f"[{ts()}] skip existing final classifier: {final_cls}")
                continue

            # MLM stage
            if args.skip_existing and final_mlm.exists():
                print(f"[{ts()}] skip existing MLM checkpoint: {final_mlm}")
            else:
                mlm_args = build_stage_args(tok, seed, "mlm", mlm_out, False, "", "", common)
                mlm_cmd = mlm_args
                mlm_success, mlm_retry, mlm_peak = run_stage(
                    "mlm",
                    mlm_script,
                    supported_mlm,
                    mlm_cmd,
                    logs_dir,
                    args.gpu_id,
                    args.stream_logs,
                )
                summary_rows.append(
                    {
                        "tokenizer": tok,
                        "seed": str(seed),
                        "stage": "mlm",
                        "success": str(mlm_success),
                        "attempts": str(mlm_retry),
                        "peak_mem_mib": str(mlm_peak),
                        "output": str(mlm_out / "mlm"),
                    }
                )
                if not mlm_success:
                    print(f"[{ts()}] MLM failed for {tok} seed={seed}; stop this seed.")
                    continue

            # Classification stage
            cls_args = build_stage_args(
                tok, seed, "cls", mlm_out, True, train_tsv, eval_tsv, common
            )
            cls_success, cls_retry, cls_peak = run_stage(
                "classifier",
                cls_script,
                supported_cls,
                cls_args,
                logs_dir,
                args.gpu_id,
                args.stream_logs,
            )
            summary_rows.append(
                {
                    "tokenizer": tok,
                    "seed": str(seed),
                    "stage": "classifier",
                    "success": str(cls_success),
                    "attempts": str(cls_retry),
                    "peak_mem_mib": str(cls_peak),
                    "output": str(mlm_out / "classifier"),
                }
            )
            if not cls_success:
                print(f"[{ts()}] Classifier failed for {tok} seed={seed}.")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tokenizer", "seed", "stage", "success", "attempts", "peak_mem_mib", "output"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[{ts()}] Done. summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
