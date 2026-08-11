import argparse
import codecs
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from experiment_common import (
    REGION_LABELS,
    SEEDS,
    TOKENIZER_NAMES,
    latest_complete_checkpoint,
    save_json,
)


ROOT = Path(__file__).resolve().parent
DATA_SCRIPT = ROOT / "step1_prepare_parallel_data.py"
DECODER_SCRIPT = ROOT / "step2_pretrain_standard_decoder_clm.py"
TRANSLATION_SCRIPT = ROOT / "step3_train_dialect_to_standard.py"
SUMMARY_SCRIPT = ROOT / "step4_summarize_results.py"
GPU_MONITOR_SCRIPT = ROOT / "process_gpu_monitor.py"
DEFAULT_SPLIT_MANIFEST = (
    "../260807_4tokenizer_5seed_region_classification/"
    "data/corpus_split_manifest_80_10_10.csv"
)
DEFAULT_SOURCE_MLM_ROOT = (
    "../260807_4tokenizer_5seed_region_classification/outputs/mlm"
)


def create_smoke_fixture() -> Path:
    fixture_dir = ROOT / "smoke_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixture_dir / "corpus_split_manifest_80_10_10.csv"
    rows = []
    split_names = ["train", "validation", "test"]
    for region_index, region in enumerate(REGION_LABELS):
        for file_index, split in enumerate(split_names):
            json_path = fixture_dir / f"region_{region_index}_{split}.json"
            utterances = []
            for sentence_index in range(32):
                dialect = (
                    f"{region} smoke sample {file_index} sentence {sentence_index}. "
                    f"region marker {region_index}."
                )
                if sentence_index % 4 == 0:
                    standard = dialect
                else:
                    standard = (
                        f"{region} standard sample {file_index} sentence {sentence_index}. "
                        f"region marker {region_index}."
                    )
                utterances.append(
                    {"dialect_form": dialect, "standard_form": standard}
                )
            with json_path.open("w", encoding="utf-8") as f:
                json.dump({"utterance": utterances}, f, ensure_ascii=False)
            rows.append(
                {
                    "region": region,
                    "source_group": "synthetic_smoke",
                    "source_type": "base",
                    "path": str(json_path.resolve()),
                    "split": split,
                }
            )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["region", "source_group", "source_type", "path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SMOKE] Synthetic parallel-data manifest: {manifest_path}")
    return manifest_path


def display_command(command: List[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def write_console(output: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_output = output.encode(encoding, errors="replace").decode(encoding)
    sys.stdout.write(safe_output)
    sys.stdout.flush()


def numeric(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "").strip()
    if value in {"", "None", "N/A", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))]


def summarize_process_gpu_csv(csv_path: Path, wall_seconds: float) -> Dict[str, object]:
    rows = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    process_memory = [
        value
        for row in rows
        if (value := numeric(row, "process_gpu_memory_mib")) is not None
    ]
    process_sm = [
        value
        for row in rows
        if (value := numeric(row, "process_sm_utilization_percent")) is not None
    ]
    active_process_sm = [value for value in process_sm if value >= 5.0]
    device_util = [
        value
        for row in rows
        if (value := numeric(row, "device_gpu_utilization_percent")) is not None
    ]
    device_memory = [
        value
        for row in rows
        if (value := numeric(row, "device_memory_used_mib")) is not None
    ]
    power = [
        value
        for row in rows
        if (value := numeric(row, "device_power_draw_w")) is not None
    ]
    temperatures = [
        value
        for row in rows
        if (value := numeric(row, "device_temperature_c")) is not None
    ]
    summary: Dict[str, object] = {
        "samples": len(rows),
        "wall_seconds": wall_seconds,
        "process_metric_scope": "target_pid_only",
        "device_metric_scope": "whole_gpu_including_other_processes",
        "process_memory_samples": len(process_memory),
        "process_sm_samples": len(process_sm),
    }
    if process_memory:
        summary.update(
            {
                "peak_process_gpu_memory_mib": max(process_memory),
                "average_process_gpu_memory_mib": statistics.mean(process_memory),
            }
        )
    if process_sm:
        summary.update(
            {
                "average_process_sm_percent": statistics.mean(process_sm),
                "p95_process_sm_percent": percentile(process_sm, 0.95),
                "peak_process_sm_percent": max(process_sm),
                "average_active_process_sm_percent": statistics.mean(active_process_sm)
                if active_process_sm
                else 0.0,
                "process_active_fraction": len(active_process_sm) / len(process_sm),
            }
        )
    if device_util:
        summary.update(
            {
                "average_whole_device_gpu_utilization_percent": statistics.mean(device_util),
                "peak_whole_device_memory_used_mib": max(device_memory)
                if device_memory
                else None,
            }
        )
    if power:
        average_power = statistics.mean(power)
        summary.update(
            {
                "average_whole_device_power_w": average_power,
                "estimated_whole_device_energy_wh": average_power * wall_seconds / 3600.0,
            }
        )
    if temperatures:
        summary["maximum_whole_device_temperature_c"] = max(temperatures)
    return summary


def run_stage(stage_name: str, command: List[str], logs_dir: Path, monitor_gpu: bool) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage_name}.log"
    gpu_csv_path = logs_dir / f"{stage_name}_gpu.csv"
    gpu_summary_path = logs_dir / f"{stage_name}_gpu_summary.json"
    print("\n" + "=" * 100, flush=True)
    print(f"STAGE: {stage_name}", flush=True)
    print(display_command(command), flush=True)
    print("=" * 100 + "\n", flush=True)

    start = time.perf_counter()
    return_code = -1
    process = None
    monitor_process = None
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_file:
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
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
                        "--pid",
                        str(process.pid),
                        "--output",
                        str(gpu_csv_path),
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
        if process is not None and process.poll() is None:
            process.terminate()
        if monitor_process is not None:
            monitor_process.terminate()
            try:
                monitor_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor_process.kill()
        if monitor_gpu:
            save_json(
                gpu_summary_path,
                summarize_process_gpu_csv(gpu_csv_path, wall_seconds),
            )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def preflight(args: argparse.Namespace, split_manifest: Path, source_mlm_root: Path) -> None:
    missing = []
    for package in (
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "sentencepiece",
        "tqdm",
        "sacrebleu",
        "rapidfuzz",
    ):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}. "
            "Run: python -m pip install -r requirements_260811.txt"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")
    device = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[PREFLIGHT] GPU: {device} ({memory_gb:.1f} GB)")
    if memory_gb < 40 and not args.smoke:
        raise RuntimeError(
            "The full profile expects an RTX A6000-class 48 GB GPU. "
            "Use --smoke locally or adjust the runner batch profile."
        )
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")
    for tokenizer in TOKENIZER_NAMES:
        model_dir = source_mlm_root / tokenizer / "final_model"
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Source MLM model not found: {model_dir}")


def completed_decoder(output_dir: Path) -> bool:
    return (
        (output_dir / "decoder_pretraining_metadata.json").is_file()
        and (output_dir / "final_model" / "config.json").is_file()
    )


def completed_translation(output_dir: Path) -> bool:
    return (
        (output_dir / "experiment_metadata.json").is_file()
        and (output_dir / "test_generation_report.json").is_file()
        and (output_dir / "final_model" / "config.json").is_file()
    )


def add_boolean_flag(command: List[str], name: str, value: bool) -> None:
    command.append(f"--{name}" if value else f"--no-{name}")


def decoder_command(args, data_root, cache_root, outputs_root):
    output_dir = outputs_root / "shared_standard_decoder"
    command = [
        sys.executable,
        str(DECODER_SCRIPT),
        "--train_corpus",
        str(data_root / "corpus" / "standard_train_corpus.txt"),
        "--validation_corpus",
        str(data_root / "corpus" / "standard_validation_corpus.txt"),
        "--dataset_cache_dir",
        str(cache_root / "datasets_decoder"),
        "--output_dir",
        str(output_dir),
    ]
    if args.smoke:
        command.extend(
            [
                "--num_train_epochs",
                "1",
                "--train_batch_size",
                "8",
                "--eval_batch_size",
                "16",
                "--dataloader_num_workers",
                "0",
                "--preprocessing_num_workers",
                "1",
                "--max_train_samples",
                "160",
                "--max_validation_samples",
                "160",
                "--logging_steps",
                "5",
            ]
        )
    if args.overwrite:
        command.append("--overwrite_output_dir")
    else:
        checkpoint = latest_complete_checkpoint(output_dir)
        if checkpoint:
            command.extend(["--resume_from_checkpoint", str(checkpoint)])
    return command


def translation_command(
    args,
    tokenizer,
    seed,
    data_root,
    cache_root,
    outputs_root,
    source_mlm_root,
):
    output_dir = outputs_root / "translation" / tokenizer / f"seed_{seed}"
    command = [
        sys.executable,
        str(TRANSLATION_SCRIPT),
        "--tokenizer_name",
        tokenizer,
        "--seed",
        str(seed),
        "--source_mlm_model_dir",
        str(source_mlm_root / tokenizer / "final_model"),
        "--shared_decoder_model_dir",
        str(outputs_root / "shared_standard_decoder" / "final_model"),
        "--train_tsv",
        str(data_root / "translation" / "translation_train.tsv"),
        "--validation_tsv",
        str(data_root / "translation" / "translation_validation.tsv"),
        "--test_tsv",
        str(data_root / "translation" / "translation_test.tsv"),
        "--dataset_cache_dir",
        str(cache_root / "datasets_translation"),
        "--tokenized_cache_dir",
        str(cache_root / "tokenized_translation" / tokenizer),
        "--output_dir",
        str(output_dir),
    ]
    if args.smoke:
        command.extend(
            [
                "--num_train_epochs",
                "1",
                "--train_batch_size",
                "8",
                "--eval_batch_size",
                "8",
                "--gradient_accumulation_steps",
                "1",
                "--generation_num_beams",
                "1",
                "--dataloader_num_workers",
                "0",
                "--preprocessing_num_workers",
                "1",
                "--validation_generation_samples",
                "160",
                "--max_train_samples",
                "160",
                "--max_test_samples",
                "160",
                "--logging_steps",
                "5",
            ]
        )
    if args.overwrite:
        command.append("--overwrite_output_dir")
    else:
        checkpoint = latest_complete_checkpoint(output_dir)
        if checkpoint:
            command.extend(["--resume_from_checkpoint", str(checkpoint)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete four-tokenizer, five-seed translation experiment."
    )
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--source_mlm_root", default=DEFAULT_SOURCE_MLM_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        split_manifest = create_smoke_fixture()
        source_mlm_root = (
            ROOT.parent
            / "260807_4tokenizer_5seed_region_classification"
            / "outputs_smoke"
            / "mlm"
        )
    else:
        split_manifest = Path(args.split_manifest)
        source_mlm_root = Path(args.source_mlm_root)
    split_manifest = split_manifest.resolve()
    source_mlm_root = source_mlm_root.resolve()
    preflight(args, split_manifest, source_mlm_root)

    suffix = "_smoke" if args.smoke else ""
    data_root = ROOT / f"data{suffix}"
    if args.smoke and os.name == "nt":
        cache_root = Path(ROOT.anchor) / "kd260811_cache"
    else:
        cache_root = ROOT / f"cache{suffix}"
    outputs_root = ROOT / f"outputs{suffix}"
    logs_dir = ROOT / f"logs{suffix}"
    result_dir = ROOT / f"results{suffix}"

    data_metadata = data_root / "preparation_metadata.json"
    if not data_metadata.is_file() or args.overwrite:
        data_command = [
            sys.executable,
            str(DATA_SCRIPT),
            "--split_manifest",
            str(split_manifest),
            "--output_dir",
            str(data_root),
        ]
        if args.overwrite:
            data_command.append("--overwrite")
        run_stage("01_prepare_parallel_data", data_command, logs_dir, monitor_gpu=False)
    else:
        print(f"[SKIP] Prepared parallel data: {data_root}")

    decoder_dir = outputs_root / "shared_standard_decoder"
    if not completed_decoder(decoder_dir) or args.overwrite:
        run_stage(
            "02_pretrain_shared_decoder",
            decoder_command(args, data_root, cache_root, outputs_root),
            logs_dir,
            monitor_gpu=True,
        )
    else:
        print(f"[SKIP] Shared decoder: {decoder_dir}")

    for tokenizer in TOKENIZER_NAMES:
        for seed in SEEDS:
            output_dir = outputs_root / "translation" / tokenizer / f"seed_{seed}"
            if completed_translation(output_dir) and not args.overwrite:
                print(f"[SKIP] Completed translation run: {tokenizer}, seed={seed}")
                continue
            run_stage(
                f"03_translate_{tokenizer}_seed_{seed}",
                translation_command(
                    args,
                    tokenizer,
                    seed,
                    data_root,
                    cache_root,
                    outputs_root,
                    source_mlm_root,
                ),
                logs_dir,
                monitor_gpu=True,
            )

    summary_command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--outputs_root",
        str(outputs_root),
        "--logs_dir",
        str(logs_dir),
        "--result_dir",
        str(result_dir),
    ]
    run_stage("04_summarize_results", summary_command, logs_dir, monitor_gpu=False)
    print("\n=== COMPLETE ===")
    print(f"Final results: {result_dir / 'final_results.md'}")


if __name__ == "__main__":
    main()
