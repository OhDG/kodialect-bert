#!/usr/bin/env python3
"""Fresh decoder, then independent tokenizer/seed runs. Controller uses only stdlib.

One GPU owns each tokenizer's cache and runs its seeds sequentially. There is no
DDP and no sharing of trained decoder instances between translation runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
TOKENIZERS = ("dialect", "klue", "kobert", "mbert")
SEEDS = (13, 21, 42, 87, 100)
DECODER_STAGE = "02_pretrain_shared_decoder"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assignment(gpus):
    if (not 1 <= len(gpus) <= 4 or any(not re.fullmatch(r"0|[1-9][0-9]*", x) for x in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("Select 1-4 distinct nonnegative GPU indices.")
    lanes = {gpu: [] for gpu in gpus}
    for index, tokenizer in enumerate(TOKENIZERS):
        lanes[gpus[index % len(gpus)]].append(tokenizer)
    return lanes


def make_config(root, run_id, gpus, profile):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must contain only letters, digits, hyphen and underscore.")
    for section in ("decoder", "translation"):
        config = profile[section]
        if config["train_batch_size"] * config["gradient_accumulation_steps"] != 256:
            raise ValueError("The controlled experiment requires effective batch 256.")
    sources = [root / "run_parallel.py", root / "requirements.txt", *sorted((root / "engine").glob("*.py"))]
    manifests = sorted((root / "provenance").glob("inputs_*_manifest.json"))
    return {
        "schema": 1, "run_id": run_id, "root": str(root), "python": sys.executable,
        "gpus": gpus, "tokenizers": list(TOKENIZERS), "seeds": list(SEEDS),
        "assignments": assignment(gpus), "profile": profile,
        "code_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in sources},
        "input_manifest_sha256": {p.name: sha256(p) for p in manifests},
    }


def child_environment(gpu, profile):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS=str(profile["cpu_threads"]), MKL_NUM_THREADS=str(profile["cpu_threads"]),
               RAYON_NUM_THREADS=str(profile["cpu_threads"]))
    return env


def add_options(command, options):
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    return command


def stage_paths(run_dir, tokenizer=None, seed=None):
    if tokenizer is None:
        return DECODER_STAGE, run_dir / "outputs" / "shared_standard_decoder"
    return f"03_translate_{tokenizer}_seed_{seed}", run_dir / "outputs" / "translation" / tokenizer / f"seed_{seed}"


def build_command(root, run_dir, profile, tokenizer=None, seed=None):
    _, output = stage_paths(run_dir, tokenizer, seed)
    common = {key: profile[key] for key in (
        "dataloader_num_workers", "dataloader_prefetch_factor", "preprocessing_num_workers")}
    if tokenizer is None:
        command = [sys.executable, str(root / "engine" / "step2_pretrain_standard_decoder_clm.py")]
        options = {
            "target_tokenizer_name": root / "models/mlm/klue/final_model",
            "train_corpus": root / "data/corpus/standard_train_corpus.txt",
            "validation_corpus": root / "data/corpus/standard_validation_corpus.txt",
            "dataset_cache_dir": run_dir / "cache/decoder", "output_dir": output,
            **common, **profile["decoder"],
        }
    else:
        command = [sys.executable, str(root / "engine" / "step3_train_dialect_to_standard.py")]
        options = {
            "tokenizer_name": tokenizer, "seed": seed,
            "source_mlm_model_dir": root / f"models/mlm/{tokenizer}/final_model",
            "shared_decoder_model_dir": run_dir / "outputs/shared_standard_decoder/final_model",
            "dataset_cache_dir": run_dir / f"cache/{tokenizer}/raw",
            "tokenized_cache_dir": run_dir / f"cache/{tokenizer}/tokenized",
            "output_dir": output,
            **{f"{s}_tsv": root / f"data/translation/translation_{s}.tsv" for s in ("train", "validation", "test")},
            **common, **profile["translation"],
        }
    return add_options(command, options)


def weights_present(directory):
    directory = Path(directory)
    if not (directory / "config.json").is_file():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = directory / name
        if path.is_file() and path.stat().st_size > 0:
            return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            try:
                shards = set(read_json(index)["weight_map"].values())
                if shards and all(Path(s).name == s and (directory / s).is_file()
                                  and (directory / s).stat().st_size > 0 for s in shards):
                    return True
            except (ValueError, KeyError, TypeError):
                pass
    return False


def latest_checkpoint(output, require_scaler=False):
    valid = []
    for path in output.glob("checkpoint-*"):
        if not re.fullmatch(r"checkpoint-\d+", path.name) or not weights_present(path):
            continue
        if not all((path / name).is_file() and (path / name).stat().st_size > 0
                   for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")):
            continue
        if require_scaler and not ((path / "scaler.pt").is_file() and (path / "scaler.pt").stat().st_size > 0):
            continue
        try:
            step = int(path.name.split("-")[-1])
            if read_json(path / "trainer_state.json")["global_step"] != step:
                continue
            valid.append((step, path))
        except (ValueError, KeyError, TypeError):
            continue
    return max(valid, default=(0, None))[1]


def output_complete(output, tokenizer=None, seed=None, expected_test=None):
    try:
        name = "experiment_metadata.json" if tokenizer else "decoder_pretraining_metadata.json"
        metadata = read_json(output / name)
        if metadata.get("completed") is not True or not weights_present(output / "final_model"):
            return False
        if not tokenizer:
            saved = output / "final_model"
            return ((output / "validation_results.json").is_file() and (saved / "tokenizer_config.json").is_file()
                    and any(p.is_file() and p.stat().st_size > 0 for pattern in ("*.txt", "*.model", "tokenizer.json")
                            for p in saved.glob(pattern)))
        if metadata.get("tokenizer_name") != tokenizer or metadata.get("seed") != seed:
            return False
        for side in ("source_tokenizer", "target_tokenizer"):
            saved = output / "final_model" / side
            if not (saved / "tokenizer_config.json").is_file() or not any(
                    p.is_file() and p.stat().st_size > 0 for pattern in ("*.txt", "*.model", "*.spiece", "tokenizer.json")
                    for p in saved.glob(pattern)):
                return False
        for split, count in (("validation", 50000), ("test", expected_test)):
            report = read_json(output / f"{split}_generation_report.json")
            if count is not None and report["overall"]["count"] != count:
                return False
            path = output / f"{split}_predictions.tsv.gz"
            if not path.is_file() or path.stat().st_size == 0:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class ExclusiveRun:
    def __init__(self, root):
        self.path = root / "runs/.runner.lock"
        self.handle = None

    def __enter__(self):
        if os.name != "posix":
            raise RuntimeError("Training targets Linux. Use --plan or unit tests on Windows.")
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError("A runner is already active in this bundle. Inspect it before starting another.")
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started={now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *unused):
        self.handle.close()


def preflight(root, gpus, minimum_free):
    from tools.verify_inputs import verify_root
    print("[CHECK] Verifying all data/model SHA256 values...", flush=True)
    inputs = verify_root(root)
    free = shutil.disk_usage(root).free / 2**30
    print(f"[CHECK] {root}: {free:.1f} GiB free; requested guard {minimum_free:.1f} GiB", flush=True)
    if free < minimum_free:
        raise RuntimeError("Insufficient free disk for the configured guard. Choose a larger /data filesystem; see README.")
    # This imports dependencies only in the selected Python environment subprocess.
    command = [sys.executable, str(root / "tools/smoke_test.py"), "--check-environment",
               "--output-dir", str(root / "smoke/preflight_environment")]
    subprocess.run(command, check=True, cwd=root)
    environment = read_json(root / "smoke/preflight_environment/environment.json")
    probe = "import torch,json; print(json.dumps([{'index':i,'name':torch.cuda.get_device_name(i),'bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]))"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    hardware = json.loads(subprocess.check_output([sys.executable, "-c", probe], env=env, text=True))
    for gpu in gpus:
        if int(gpu) >= len(hardware) or hardware[int(gpu)]["bytes"] < 23 * 2**30:
            raise RuntimeError(f"GPU {gpu} unavailable or under 23 GiB. Run nvidia-smi and inspect the profile.")
    host_probe = "import psutil,json; print(json.dumps({'physical_cores':psutil.cpu_count(logical=False),'logical_cpus':psutil.cpu_count(),'available_ram_gib':psutil.virtual_memory().available/2**30,'total_ram_gib':psutil.virtual_memory().total/2**30}))"
    host = json.loads(subprocess.check_output([sys.executable, "-c", host_probe], text=True))
    print(f"[CHECK] Host: {host}", flush=True)
    if host["available_ram_gib"] < 24 * len(gpus):
        print("[WARNING] Less than 24 GiB available host RAM per worker; fewer simultaneous GPUs may be needed.", flush=True)
    return {"inputs": inputs, "gpus": hardware, "host": host, "environment": environment,
            "free_disk_gib": free, "checked_at": now()}


def gpu_summary(logs, stage):
    memory, sm = [], []
    for path in sorted(logs.glob(stage + "_attempt*_gpu.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key, target in (("process_gpu_memory_mib", memory), ("process_sm_utilization_percent", sm)):
                    try:
                        value = float(row[key])
                        if value >= 0:
                            target.append(value)
                    except (KeyError, ValueError, TypeError):
                        pass
    active = [x for x in sm if x >= 5]
    return {"peak_process_gpu_memory_mib": max(memory, default=None),
            "average_active_process_sm_percent": statistics.mean(active) if active else None,
            "samples": len(memory), "measurement_scope": "experiment Python process only (exact PID)"}


class Runner:
    def __init__(self, root, config, resume, preflight_report):
        self.root, self.config = root, config
        self.profile = config["profile"]
        self.directory = root / "runs" / config["run_id"]
        self.signature = fingerprint(config)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.children = {}
        self.expected_test = preflight_report["inputs"]["pair_counts"]["test"]
        if self.directory.exists():
            if not resume:
                raise RuntimeError("Run directory exists. Use a new run-id for fresh training or --resume for this run.")
            if read_json(self.directory / "run_config.json") != config:
                raise RuntimeError("Resume configuration, code or inputs changed. Restore them or use a fresh run-id.")
            self.state = read_json(self.directory / "state.json")
            for record in self.state["stages"].values():
                for attempt in record.get("attempts", []):
                    if attempt.get("status") == "running":
                        attempt["status"] = "interrupted_without_end_record"
                        record["timing_incomplete"] = True
        else:
            if resume:
                raise RuntimeError("Cannot resume: run directory does not exist.")
            self.directory.mkdir(parents=True)
            atomic_json(self.directory / "run_config.json", config)
            self.state = {"started_at": now(), "run_signature": self.signature, "stages": {}, "status": "prepared"}
        self.logs = self.directory / "logs"
        self.logs.mkdir(exist_ok=True)
        atomic_json(self.directory / "preflight.json", preflight_report)
        self.save()

    def save(self):
        with self.mutex:
            self.state["updated_at"] = now()
            atomic_json(self.directory / "state.json", self.state)

    def terminate_children(self):
        self.stop.set()
        with self.mutex:
            children = list(self.children.values())
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def run_stage(self, gpu, tokenizer=None, seed=None):
        if self.stop.is_set():
            return
        stage, output = stage_paths(self.directory, tokenizer, seed)
        with self.mutex:
            previous = self.state["stages"].get(stage, {})
        if output_complete(output, tokenizer, seed, self.expected_test):
            print(f"[SKIP complete] {stage}", flush=True)
            with self.mutex:
                self.state["stages"][stage] = {**previous, "status": "completed", "gpu": gpu,
                    "tokenizer": tokenizer, "seed": seed, "revalidated_at": now()}
                self.save()
            return
        checkpoint = latest_checkpoint(output, require_scaler=True) if output.exists() else None
        if output.exists() and any(output.iterdir()) and checkpoint is None:
            raise RuntimeError(f"Incomplete output has no complete resumable checkpoint: {output}. Preserve it and use a new run-id.")
        command = build_command(self.root, self.directory, self.profile, tokenizer, seed)
        if checkpoint:
            command.extend(["--resume_from_checkpoint", str(checkpoint)])
        attempts = list(previous.get("attempts", []))
        attempt_number = len(attempts) + 1
        log_path = self.logs / f"{stage}_attempt{attempt_number}.log"
        attempt = {"started_at": now(), "command": command, "log": str(log_path), "status": "running"}
        attempts.append(attempt)
        record = {"status": "running", "started_at": attempt["started_at"], "gpu": gpu,
                  "tokenizer": tokenizer, "seed": seed, "attempts": attempts,
                  "timing_incomplete": previous.get("timing_incomplete", False),
                  "prior_elapsed_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts[:-1])}
        with self.mutex:
            self.state["stages"][stage] = record
            self.save()
        print(f"[START gpu={gpu}] {stage}  log={log_path}", flush=True)
        start = time.monotonic()
        process = monitor = None
        try:
            # Recheck before spawn after another worker failed.
            if self.stop.is_set():
                raise RuntimeError("Run stopped before starting this stage.")
            with log_path.open("x", encoding="utf-8") as log:
                with self.mutex:
                    if self.stop.is_set():
                        raise RuntimeError("Run stopped.")
                    process = subprocess.Popen(command, cwd=self.root, env=child_environment(gpu, self.profile),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.children[stage] = process
                monitor_command = [sys.executable, str(self.root / "engine/process_gpu_monitor.py"),
                    "--pid", str(process.pid), "--gpu_id", gpu,
                    "--output", str(self.logs / f"{stage}_attempt{attempt_number}_gpu.csv"),
                    "--interval", str(self.profile["monitor_interval"])]
                monitor = subprocess.Popen(monitor_command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if self.stop.wait(5):
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                if process.returncode != 0:
                    raise RuntimeError(f"{stage} exited {process.returncode}; inspect {log_path}")
            if not output_complete(output, tokenizer, seed, self.expected_test):
                raise RuntimeError(f"{stage} exited without complete model/evaluation artifacts: {output}")
            attempt["status"] = record["status"] = "completed"
            print(f"[DONE gpu={gpu}] {stage}", flush=True)
        except BaseException as exc:
            attempt["status"] = record["status"] = "failed"
            record["error"] = str(exc)
            self.terminate_children()
            raise
        finally:
            # Also reap a worker if monitor startup or output validation failed.
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if monitor and monitor.poll() is None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
            with self.mutex:
                self.children.pop(stage, None)
                attempt["elapsed_seconds"] = time.monotonic() - start
                attempt["ended_at"] = record["ended_at"] = now()
                record["elapsed_seconds"] = sum(a.get("elapsed_seconds", 0) for a in attempts)
                resource = gpu_summary(self.logs, stage)
                resource["wall_seconds"] = record["elapsed_seconds"]
                resource["attempt_count"] = len(attempts)
                resource["timing_incomplete"] = record["timing_incomplete"]
                atomic_json(self.logs / f"{stage}_gpu_summary.json", resource)
                self.save()

    def summarize(self):
        command = [sys.executable, str(self.root / "engine/step4_summarize_results.py"),
                   "--outputs_root", str(self.directory / "outputs"), "--logs_dir", str(self.logs),
                   "--result_dir", str(self.directory / "results")]
        with self.mutex:
            if self.stop.is_set():
                raise RuntimeError("Interrupted before summary.")
            process = subprocess.Popen(command, cwd=self.root, start_new_session=True)
            self.children["summary"] = process
        try:
            while process.poll() is None:
                if self.stop.wait(2):
                    self.terminate_children()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            if self.stop.is_set() or process.returncode != 0:
                raise RuntimeError("Summary interrupted or failed; completed experiment artifacts are retained.")
        finally:
            with self.mutex:
                self.children.pop("summary", None)
        runtimes = {name: {key: record.get(key) for key in
                          ("elapsed_seconds", "timing_incomplete", "gpu", "tokenizer", "seed")}
                    for name, record in self.state["stages"].items()}
        atomic_json(self.directory / "results/pipeline_stage_times.json", runtimes)
        # Engine training/test metrics refer to the last Trainer invocation on resume.
        # Separate end-to-end stage timing retains all recorded attempts.
        lines = ["", "## Pipeline wall time (all recorded attempts)", "",
                 "Includes loading, training, generation, CPU metrics and writes. Training/test timing above comes from the final invocation; resumed runs must not be treated as uninterrupted timing benchmarks.", "",
                 "| Stage | GPU | Hours | Timing complete |", "| --- | ---: | ---: | --- |"]
        for name, record in runtimes.items():
            elapsed = record["elapsed_seconds"]
            hours = f"{elapsed / 3600:.3f}" if elapsed is not None else "N/A"
            lines.append(f"| {name} | {record['gpu']} | {hours} | {'no' if record['timing_incomplete'] or elapsed is None else 'yes'} |")
        with (self.directory / "results/final_results.md").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")

    def lane(self, gpu, tokenizers):
        for tokenizer in tokenizers:
            for seed in SEEDS:
                if self.stop.is_set():
                    return
                self.run_stage(gpu, tokenizer, seed)

    def run(self):
        def interrupted(signum, frame):
            print(f"[STOP] signal {signum}; stopping this runner's workers.", flush=True)
            self.terminate_children()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        self.state["status"] = "running"
        self.state["launcher_pid"] = os.getpid()
        self.save()
        try:
            self.run_stage(self.config["gpus"][0])
            if self.stop.is_set():
                raise RuntimeError("Stopped before translation.")
            with ThreadPoolExecutor(max_workers=len(self.config["gpus"])) as executor:
                futures = [executor.submit(self.lane, gpu, toks) for gpu, toks in self.config["assignments"].items()]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        self.terminate_children()
                        raise
            if self.stop.is_set():
                raise RuntimeError("Interrupted before summary.")
            self.summarize()
            self.state["status"] = "completed"
            self.state["ended_at"] = now()
        except BaseException:
            self.state["status"] = "failed_or_interrupted"
            self.terminate_children()
            raise
        finally:
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="4090_fresh_01")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/rtx4090.json")
    parser.add_argument("--min-free-gib", type=float, default=None,
                        help="Disk guard: fresh/check 100 GiB, resume 20 GiB; not a maximum-size estimate")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = make_config(ROOT, args.run_id, args.gpus, read_json(args.profile))
        directory = ROOT / "runs" / args.run_id
        if args.status:
            state = read_json(directory / "state.json")
            print(f"{args.run_id}: {state['status']} (updated {state['updated_at']})")
            completed = 0
            for name, record in state["stages"].items():
                print(f"  GPU {record['gpu']}: {name}: {record['status']}")
                completed += bool(record.get("tokenizer") and record["status"] == "completed")
            print(f"Translation completed: {completed}/20")
            return 0
        if args.plan:
            print(json.dumps({"assignments": config["assignments"], "profile": config["profile"],
                  "decoder_command": build_command(ROOT, directory, config["profile"]),
                  "example_translation_command": build_command(ROOT, directory, config["profile"], "dialect", 13),
                  "fresh_decoder": True, "reuse_imported_mlm_encoders": True}, indent=2))
            return 0
        minimum = args.min_free_gib if args.min_free_gib is not None else (20 if args.resume else 100)
        if minimum < 0:
            raise ValueError("min-free-gib cannot be negative")
        with ExclusiveRun(ROOT):
            report = preflight(ROOT, args.gpus, minimum)
            if args.check:
                print("[OK] Inputs, environment, GPU availability and disk guard passed. Run GPU smoke next.")
                return 0
            Runner(ROOT, config, args.resume, report).run()
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
