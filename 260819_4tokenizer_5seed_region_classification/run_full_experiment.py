import argparse
import codecs
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from experiment_common import REGION_LABELS, TOKENIZER_SPECS, latest_complete_checkpoint, save_json


ROOT = Path(__file__).resolve().parent
DATA_SCRIPT = ROOT / "step1_prepare_data_80_10_10.py"
TOKENIZER_SCRIPT = ROOT / "step2_train_dialect_tokenizer.py"
MLM_SCRIPT = ROOT / "step3_pretrain_small_bert_mlm.py"
CLASSIFIER_SCRIPT = ROOT / "step4_finetune_region_classifier.py"
SUMMARY_SCRIPT = ROOT / "step5_summarize_results.py"
GPU_MONITOR_SCRIPT = ROOT / "gpu_monitor.py"

DEFAULT_TOKENIZERS = ["dialect", "klue", "kobert", "mbert"]
DEFAULT_SEEDS = [13, 21, 42, 87, 100]
DEFAULT_SOURCE_MANIFEST = "../260630_test_1/corpus_split_manifest.csv"
# Reuse the already-trained MLM encoders 260811 (translation) already fine-tunes
# against, instead of retraining MLM from scratch. Pass --reuse_mlm_from "" to
# disable this and train all four tokenizers from scratch as 260807 originally did.
DEFAULT_REUSE_MLM_FROM = "../260807_4tokenizer_5seed_region_classification"


def create_smoke_fixture() -> Path:
    fixture_dir = ROOT / "smoke_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixture_dir / "corpus_split_manifest.csv"
    rows = []
    for region_index, region in enumerate(REGION_LABELS):
        for file_index in range(3):
            json_path = fixture_dir / f"region_{region_index}_file_{file_index}.json"
            utterances = [
                {
                    "dialect_form": (
                        f"{region} smoke sample {file_index} sentence {sentence_index}. "
                        f"region marker {region_index}."
                    )
                }
                for sentence_index in range(32)
            ]
            with json_path.open("w", encoding="utf-8") as f:
                json.dump({"utterance": utterances}, f, ensure_ascii=False)
            rows.append(
                {
                    "region": region,
                    "source_group": "synthetic_smoke",
                    "source_type": "base",
                    "path": str(json_path.resolve()),
                    "num_sentences": len(utterances),
                }
            )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["region", "source_group", "source_type", "path", "num_sentences"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SMOKE] Synthetic source manifest: {manifest_path}")
    return manifest_path


def data_root(args: argparse.Namespace) -> str:
    return "./data_smoke" if args.smoke else "./data"


def dialect_tokenizer_root(args: argparse.Namespace) -> str:
    return "./dialect_bert_tokenizer_smoke" if args.smoke else "./dialect_bert_tokenizer"


def cache_root(args: argparse.Namespace) -> Path:
    if args.smoke and os.name == "nt":
        return Path(ROOT.anchor) / "kd260819_cache"
    return ROOT / ("cache_smoke" if args.smoke else "cache")


def display_command(command: List[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def write_console(output: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_output = output.encode(encoding, errors="replace").decode(encoding)
    sys.stdout.write(safe_output)
    sys.stdout.flush()


def summarize_gpu_csv(csv_path: Path, wall_seconds: float) -> Dict[str, object]:
    process_vram_samples: List[float] = []
    process_utilization_samples: List[float] = []
    sample_count = 0
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sample_count += 1
                try:
                    if row.get("process_vram_mib", "").strip():
                        process_vram_samples.append(float(row["process_vram_mib"]))
                    if row.get("process_sm_utilization_percent", "").strip():
                        process_utilization_samples.append(
                            float(row["process_sm_utilization_percent"])
                        )
                except (TypeError, ValueError):
                    pass
    positive_vram_samples = [value for value in process_vram_samples if value > 0.0]
    active_utilization = [value for value in process_utilization_samples if value >= 5.0]
    return {
        "measurement_scope": "training_root_pid_and_descendants_only",
        "samples": sample_count,
        "wall_seconds": wall_seconds,
        "nvml_process_vram_supported": bool(positive_vram_samples),
        "peak_nvml_process_vram_mib": (
            max(positive_vram_samples) if positive_vram_samples else None
        ),
        "process_sm_utilization_supported": bool(process_utilization_samples),
        "average_process_sm_utilization_percent": (
            statistics.mean(process_utilization_samples)
            if process_utilization_samples
            else None
        ),
        "average_active_process_sm_utilization_percent": (
            statistics.mean(active_utilization) if active_utilization else None
        ),
        "peak_process_sm_utilization_percent": (
            max(process_utilization_samples) if process_utilization_samples else None
        ),
        "note": (
            "Process-level NVML metrics via pynvml. On Linux nvmlDeviceGet* process "
            "queries typically report real per-process VRAM/SM usage."
        ),
    }


def run_stage(
    stage_name: str,
    command: List[str],
    logs_dir: Path,
    monitor_gpu: bool,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage_name}.log"
    gpu_csv_path = logs_dir / f"{stage_name}_gpu.csv"
    gpu_summary_path = logs_dir / f"{stage_name}_gpu_summary.json"
    print("\n" + "=" * 100, flush=True)
    print(f"STAGE: {stage_name}", flush=True)
    print(display_command(command), flush=True)
    print("=" * 100 + "\n", flush=True)

    monitor_process: Optional[subprocess.Popen] = None
    process: Optional[subprocess.Popen] = None
    start = time.perf_counter()
    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_file:
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=child_env,
            )
            if monitor_gpu:
                monitor_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(GPU_MONITOR_SCRIPT),
                        "--output",
                        str(gpu_csv_path),
                        "--root_pid",
                        str(process.pid),
                        "--interval",
                        "1.0",
                    ],
                    cwd=ROOT,
                )
            assert process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                output = decoder.decode(chunk)
                write_console(output)
                log_file.write(output)
                log_file.flush()
            remaining = decoder.decode(b"", final=True)
            if remaining:
                write_console(remaining)
                log_file.write(remaining)
                log_file.flush()
            return_code = process.wait()
    finally:
        wall_seconds = time.perf_counter() - start
        if monitor_process is not None:
            monitor_process.terminate()
            try:
                monitor_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor_process.kill()
        if monitor_gpu:
            save_json(gpu_summary_path, summarize_gpu_csv(gpu_csv_path, wall_seconds))

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def preflight(args: argparse.Namespace) -> None:
    missing = []
    for package in (
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "sentencepiece",
        "tqdm",
        "psutil",
        "pynvml",
    ):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}. "
            "Run: python -m pip install -r requirements_260819.txt"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the full experiment.")
    device = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[PREFLIGHT] GPU: {device} ({memory_gb:.1f} GB)")
    if memory_gb < 40 and not args.smoke:
        raise RuntimeError(
            "The default profile expects an RTX A6000-class 48 GB GPU. "
            "Use --smoke for a small check or adjust batch sizes in the runner."
        )
    if not Path(args.source_manifest).is_file() and not (ROOT / args.source_manifest).is_file():
        raise FileNotFoundError(f"Source manifest not found: {args.source_manifest}")


def completed_mlm(output_dir: Path) -> bool:
    metadata = output_dir / "mlm_pretraining_metadata.json"
    final_model = output_dir / "final_model" / "config.json"
    return metadata.is_file() and final_model.is_file()


def reusable_mlm_encoders_missing(reuse_root: Path, tokenizers: List[str]) -> List[str]:
    return [
        tokenizer_name
        for tokenizer_name in tokenizers
        if not (reuse_root / "outputs" / "mlm" / tokenizer_name / "final_model" / "config.json").is_file()
        or not (reuse_root / "outputs" / "mlm" / tokenizer_name / "mlm_pretraining_metadata.json").is_file()
    ]


def reuse_mlm_encoder(tokenizer_name: str, reuse_root: Path, mlm_dir: Path, logs_dir: Path) -> None:
    """Copy an already-trained MLM encoder instead of retraining it.

    Only final_model/ and the pretraining metadata are copied — not any
    intermediate checkpoint-* directory, which would waste disk and time.
    The source's GPU log is copied too (if present) so step5 can still report
    an MLM row, but it is measured with the source experiment's own monitor
    (whole-device nvidia-smi for 260807), not this run's process-only one.
    """
    source_dir = reuse_root / "outputs" / "mlm" / tokenizer_name
    mlm_dir.mkdir(parents=True, exist_ok=True)

    dest_final_model = mlm_dir / "final_model"
    if dest_final_model.exists():
        shutil.rmtree(dest_final_model)
    shutil.copytree(source_dir / "final_model", dest_final_model)
    shutil.copy2(
        source_dir / "mlm_pretraining_metadata.json",
        mlm_dir / "mlm_pretraining_metadata.json",
    )
    print(f"[REUSE] Copied trained MLM encoder for {tokenizer_name} from {source_dir}")

    source_gpu_summary = reuse_root / "logs" / f"03_mlm_{tokenizer_name}_gpu_summary.json"
    source_gpu_csv = reuse_root / "logs" / f"03_mlm_{tokenizer_name}_gpu.csv"
    if source_gpu_summary.is_file():
        shutil.copy2(source_gpu_summary, logs_dir / f"03_mlm_{tokenizer_name}_gpu_summary.json")
        if source_gpu_csv.is_file():
            shutil.copy2(source_gpu_csv, logs_dir / f"03_mlm_{tokenizer_name}_gpu.csv")
        print(
            f"[REUSE] Copied {tokenizer_name}'s original whole-device GPU summary "
            "(inherited, not re-measured by this run's process-only monitor)."
        )
    else:
        print(
            f"[REUSE] No GPU summary found at {source_gpu_summary}; "
            f"the MLM efficiency row for {tokenizer_name} will be incomplete."
        )


def completed_classifier(output_dir: Path) -> bool:
    return (
        (output_dir / "experiment_metadata.json").is_file()
        and (output_dir / "test_classification_report.json").is_file()
        and (output_dir / "final_model" / "config.json").is_file()
    )


def mlm_batch_profile(tokenizer_name: str, smoke: bool, args: argparse.Namespace) -> Dict[str, int]:
    if smoke:
        return {"train": 8, "eval": 16, "accumulation": 1}
    if tokenizer_name == "mbert":
        return {
            "train": args.mlm_train_batch_mbert,
            "eval": args.mlm_eval_batch_mbert,
            "accumulation": args.mlm_grad_accum_mbert,
        }
    return {
        "train": args.mlm_train_batch_others,
        "eval": args.mlm_eval_batch_others,
        "accumulation": args.mlm_grad_accum_others,
    }


def data_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(DATA_SCRIPT),
        "--source_manifest",
        args.source_manifest,
        "--output_dir",
        data_root(args),
        "--seed",
        "42",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def tokenizer_command(args: argparse.Namespace) -> List[str]:
    data_dir = data_root(args)
    command = [
        sys.executable,
        str(TOKENIZER_SCRIPT),
        "--corpus",
        f"{data_dir}/corpus/dialect_train_corpus.txt",
        "--output_dir",
        dialect_tokenizer_root(args),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.smoke:
        command.extend(
            [
                "--vocab_size",
                "100",
                "--min_frequency",
                "1",
                "--limit_alphabet",
                "100",
            ]
        )
    return command


def mlm_command(args: argparse.Namespace, tokenizer_name: str, output_dir: Path) -> List[str]:
    batch = mlm_batch_profile(tokenizer_name, args.smoke, args)
    data_dir = data_root(args)
    command = [
        sys.executable,
        str(MLM_SCRIPT),
        "--tokenizer_name",
        tokenizer_name,
        "--output_dir",
        str(output_dir),
        "--dialect_tokenizer_dir",
        dialect_tokenizer_root(args),
        "--train_corpus",
        f"{data_dir}/corpus/dialect_train_corpus.txt",
        "--validation_corpus",
        f"{data_dir}/corpus/dialect_validation_corpus.txt",
        "--dataset_cache_dir",
        str(cache_root(args) / "huggingface_datasets"),
        "--train_batch_size",
        str(batch["train"]),
        "--eval_batch_size",
        str(batch["eval"]),
        "--gradient_accumulation_steps",
        str(batch["accumulation"]),
        "--dataloader_num_workers",
        str(0 if args.smoke else args.dataloader_num_workers),
        "--preprocessing_num_workers",
        str(1 if args.smoke else args.preprocessing_num_workers),
        "--tokenize_batch_size",
        str(128 if args.smoke else args.tokenize_batch_size),
        "--seed",
        "42",
    ]
    if args.overwrite:
        command.append("--overwrite_output_dir")
    elif not completed_mlm(output_dir):
        checkpoint = latest_complete_checkpoint(output_dir)
        if checkpoint is not None:
            command.extend(["--resume_from_checkpoint", str(checkpoint)])
    if args.smoke:
        command.extend(
            [
                "--num_train_epochs",
                "1",
                "--max_train_samples",
                "2048",
                "--max_validation_samples",
                "512",
                "--logging_steps",
                "10",
            ]
        )
    return command


def classifier_command(
    args: argparse.Namespace,
    tokenizer_name: str,
    seed: int,
    mlm_dir: Path,
    output_dir: Path,
) -> List[str]:
    data_dir = data_root(args)
    experiment_cache = cache_root(args)
    command = [
        sys.executable,
        str(CLASSIFIER_SCRIPT),
        "--mlm_model_dir",
        str(mlm_dir / "final_model"),
        "--tokenized_cache_dir",
        str(experiment_cache / "classification_tokenized" / tokenizer_name),
        "--dataset_cache_dir",
        str(experiment_cache / "huggingface_datasets"),
        "--output_dir",
        str(output_dir),
        "--train_tsv",
        f"{data_dir}/region_classification/dialect_region_train.tsv",
        "--validation_tsv",
        f"{data_dir}/region_classification/dialect_region_validation.tsv",
        "--test_tsv",
        f"{data_dir}/region_classification/dialect_region_test.tsv",
        "--train_batch_size",
        "16" if args.smoke else str(args.classifier_train_batch),
        "--eval_batch_size",
        "32" if args.smoke else str(args.classifier_eval_batch),
        "--dataloader_num_workers",
        str(0 if args.smoke else args.dataloader_num_workers),
        "--preprocessing_num_workers",
        str(1 if args.smoke else args.preprocessing_num_workers),
        "--gradient_accumulation_steps",
        "1" if args.smoke else str(args.classifier_gradient_accumulation_steps),
        "--tokenize_batch_size",
        str(128 if args.smoke else args.tokenize_batch_size),
        "--seed",
        str(seed),
    ]
    if args.overwrite:
        command.append("--overwrite_output_dir")
    elif not completed_classifier(output_dir):
        checkpoint = latest_complete_checkpoint(output_dir)
        if checkpoint is not None:
            command.extend(["--resume_from_checkpoint", str(checkpoint)])
    if args.smoke:
        command.extend(
            [
                "--num_train_epochs",
                "1",
                "--max_train_samples",
                "2048",
                "--max_validation_samples",
                "512",
                "--max_test_samples",
                "512",
                "--logging_steps",
                "10",
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    if args.smoke and args.source_manifest == DEFAULT_SOURCE_MANIFEST:
        args.source_manifest = str(create_smoke_fixture())
    preflight(args)
    logs_dir = ROOT / ("logs_smoke" if args.smoke else "logs")
    outputs_root = ROOT / ("outputs_smoke" if args.smoke else "outputs")
    results_dir = ROOT / ("results_smoke" if args.smoke else "results")
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    reuse_root: Optional[Path] = None
    if args.reuse_mlm_from and not args.smoke:
        reuse_root = Path(args.reuse_mlm_from)
        if not reuse_root.is_absolute():
            reuse_root = (ROOT / reuse_root).resolve()
        missing = reusable_mlm_encoders_missing(reuse_root, args.tokenizers)
        if missing:
            raise FileNotFoundError(
                f"--reuse_mlm_from={reuse_root} is missing trained encoders for: {missing}. "
                'Fix the path, or pass --reuse_mlm_from "" to train all four from scratch instead.'
            )
        print(f"[REUSE] Reusing already-trained MLM encoders from: {reuse_root}")

    run_stage("01_prepare_data", data_command(args), logs_dir, monitor_gpu=False)
    if reuse_root is None:
        run_stage("02_train_dialect_tokenizer", tokenizer_command(args), logs_dir, monitor_gpu=False)

    for tokenizer_name in args.tokenizers:
        mlm_dir = outputs_root / "mlm" / tokenizer_name
        if completed_mlm(mlm_dir) and not args.overwrite:
            print(f"[SKIP] Completed MLM: {tokenizer_name}")
        elif reuse_root is not None:
            reuse_mlm_encoder(tokenizer_name, reuse_root, mlm_dir, logs_dir)
        else:
            run_stage(
                f"03_mlm_{tokenizer_name}",
                mlm_command(args, tokenizer_name, mlm_dir),
                logs_dir,
                monitor_gpu=True,
            )

        for seed in args.seeds:
            classifier_dir = outputs_root / "classifiers" / tokenizer_name / f"seed_{seed}"
            if completed_classifier(classifier_dir) and not args.overwrite:
                print(f"[SKIP] Completed classifier: {tokenizer_name}, seed={seed}")
                continue
            run_stage(
                f"04_classifier_{tokenizer_name}_seed_{seed}",
                classifier_command(args, tokenizer_name, seed, mlm_dir, classifier_dir),
                logs_dir,
                monitor_gpu=True,
            )

    summary_command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--output_root",
        str(outputs_root),
        "--result_dir",
        str(results_dir),
        "--logs_dir",
        str(logs_dir),
        "--tokenizers",
        *args.tokenizers,
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    run_stage("05_summarize_results", summary_command, logs_dir, monitor_gpu=False)
    save_json(
        results_dir / "pipeline_completed.json",
        {
            "completed": True,
            "tokenizers": args.tokenizers,
            "seeds": args.seeds,
            "smoke": args.smoke,
        },
    )
    print("\n[OK] Full experiment completed.")
    print(f"Final report: {results_dir / 'final_results.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full A6000 four-tokenizer, five-seed experiment with one command."
    )
    parser.add_argument("--source_manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--reuse_mlm_from",
        default=DEFAULT_REUSE_MLM_FROM,
        help=(
            "Directory containing an already-completed outputs/mlm/<tokenizer>/final_model "
            "for all requested tokenizers; skips MLM retraining and copies these instead. "
            'Pass --reuse_mlm_from "" to disable and train all four from scratch (ignored '
            "under --smoke, which always trains its own tiny synthetic encoders)."
        ),
    )
    parser.add_argument("--tokenizers", nargs="+", choices=sorted(TOKENIZER_SPECS), default=DEFAULT_TOKENIZERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--tokenize_batch_size", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mlm_train_batch_others", type=int, default=256)
    parser.add_argument("--mlm_eval_batch_others", type=int, default=512)
    parser.add_argument("--mlm_grad_accum_others", type=int, default=1)
    parser.add_argument("--mlm_train_batch_mbert", type=int, default=128)
    parser.add_argument("--mlm_eval_batch_mbert", type=int, default=256)
    parser.add_argument("--mlm_grad_accum_mbert", type=int, default=2)
    parser.add_argument("--classifier_train_batch", type=int, default=256)
    parser.add_argument("--classifier_eval_batch", type=int, default=2048)
    parser.add_argument("--classifier_gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
