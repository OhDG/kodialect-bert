import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List

from experiment_common import REGION_LABELS, SEEDS, TOKENIZER_NAMES, load_json, save_json


METRICS = [
    "chrf_plus_plus",
    "sacrebleu",
    "cer",
    "exact_match",
    "normalized_exact_match",
]


def mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def optional_mean_std(values):
    available = [float(value) for value in values if value is not None]
    return mean_std(available) if available else None


def format_mean_std(summary: Dict[str, float], digits: int = 4) -> str:
    return f"{summary['mean']:.{digits}f} +/- {summary['std']:.{digits}f}"


def format_optional(summary, digits: int = 2, suffix: str = "") -> str:
    if summary is None:
        return "N/A"
    return format_mean_std(summary, digits) + suffix


def load_runs(outputs_root: Path, logs_dir: Path):
    runs = {}
    for tokenizer in TOKENIZER_NAMES:
        runs[tokenizer] = {}
        for seed in SEEDS:
            run_dir = outputs_root / "translation" / tokenizer / f"seed_{seed}"
            report_path = run_dir / "test_generation_report.json"
            metadata_path = run_dir / "experiment_metadata.json"
            if not report_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Incomplete translation result: {run_dir}")
            gpu_path = logs_dir / f"03_translate_{tokenizer}_seed_{seed}_gpu_summary.json"
            runs[tokenizer][seed] = {
                "report": load_json(report_path),
                "metadata": load_json(metadata_path),
                "gpu": load_json(gpu_path) if gpu_path.is_file() else {},
            }
    return runs


def summarize(args: argparse.Namespace) -> None:
    outputs_root = Path(args.outputs_root)
    logs_dir = Path(args.logs_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(outputs_root, logs_dir)

    summary = {
        "test": {},
        "changed_only_test": {},
        "paired_dialect_minus_baseline": {},
        "per_region_chrf_plus_plus": {},
        "runtime_and_process_gpu": {},
    }
    for tokenizer in TOKENIZER_NAMES:
        summary["test"][tokenizer] = {}
        summary["changed_only_test"][tokenizer] = {}
        for metric in METRICS:
            summary["test"][tokenizer][metric] = mean_std(
                [
                    float(runs[tokenizer][seed]["report"]["overall"][metric])
                    for seed in SEEDS
                ]
            )
            summary["changed_only_test"][tokenizer][metric] = mean_std(
                [
                    float(runs[tokenizer][seed]["report"]["changed_only"][metric])
                    for seed in SEEDS
                ]
            )

        summary["per_region_chrf_plus_plus"][tokenizer] = {
            region: mean_std(
                [
                    float(
                        runs[tokenizer][seed]["report"]["per_region"][region][
                            "chrf_plus_plus"
                        ]
                    )
                    for seed in SEEDS
                ]
            )
            for region in REGION_LABELS
        }
        runtimes = [
            float(
                runs[tokenizer][seed]["metadata"]["train_process_metrics"][
                    "wall_seconds"
                ]
            )
            for seed in SEEDS
        ]
        test_runtimes = [
            float(runs[tokenizer][seed]["report"]["runtime"]["wall_seconds"])
            for seed in SEEDS
        ]
        torch_allocated = [
            float(
                runs[tokenizer][seed]["metadata"]["train_process_metrics"].get(
                    "torch_peak_allocated_gb", 0.0
                )
            )
            for seed in SEEDS
        ]
        torch_reserved = [
            float(
                runs[tokenizer][seed]["metadata"]["train_process_metrics"].get(
                    "torch_peak_reserved_gb", 0.0
                )
            )
            for seed in SEEDS
        ]
        process_vram = [
            runs[tokenizer][seed]["gpu"].get("peak_process_gpu_memory_mib")
            for seed in SEEDS
        ]
        process_sm = [
            runs[tokenizer][seed]["gpu"].get("average_active_process_sm_percent")
            for seed in SEEDS
        ]
        stage_runtimes = [
            runs[tokenizer][seed]["gpu"].get("wall_seconds") for seed in SEEDS
        ]
        summary["runtime_and_process_gpu"][tokenizer] = {
            "train_wall_hours": mean_std([value / 3600.0 for value in runtimes]),
            "total_stage_wall_hours": optional_mean_std(
                [None if value is None else float(value) / 3600.0 for value in stage_runtimes]
            ),
            "test_wall_minutes": mean_std([value / 60.0 for value in test_runtimes]),
            "torch_peak_allocated_gb": mean_std(torch_allocated),
            "torch_peak_reserved_gb": mean_std(torch_reserved),
            "peak_process_gpu_memory_gb": optional_mean_std(
                [None if value is None else float(value) / 1024.0 for value in process_vram]
            ),
            "average_active_process_sm_percent": optional_mean_std(process_sm),
        }

    for baseline in TOKENIZER_NAMES[1:]:
        summary["paired_dialect_minus_baseline"][baseline] = {}
        for metric in METRICS:
            differences = [
                float(runs["dialect"][seed]["report"]["overall"][metric])
                - float(runs[baseline][seed]["report"]["overall"][metric])
                for seed in SEEDS
            ]
            summary["paired_dialect_minus_baseline"][baseline][metric] = mean_std(
                differences
            )

    first_report = runs["dialect"][SEEDS[0]]["report"]
    summary["identity_input_baseline"] = first_report["identity_input_baseline"]
    summary["test_count"] = first_report["overall"]["count"]
    summary["changed_fraction"] = first_report["changed_fraction"]

    decoder_metadata_path = outputs_root / "shared_standard_decoder" / "decoder_pretraining_metadata.json"
    if decoder_metadata_path.is_file():
        summary["shared_decoder"] = load_json(decoder_metadata_path)
    decoder_gpu_path = logs_dir / "02_pretrain_shared_decoder_gpu_summary.json"
    if decoder_gpu_path.is_file():
        summary["shared_decoder_process_gpu"] = load_json(decoder_gpu_path)

    save_json(result_dir / "final_results.json", summary)

    with (result_dir / "test_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", *[f"{metric}_mean" for metric in METRICS], *[f"{metric}_std" for metric in METRICS]])
        for tokenizer in TOKENIZER_NAMES:
            writer.writerow(
                [tokenizer]
                + [summary["test"][tokenizer][metric]["mean"] for metric in METRICS]
                + [summary["test"][tokenizer][metric]["std"] for metric in METRICS]
            )

    lines = [
        "# Four-tokenizer, five-seed dialect-to-standard results",
        "",
        "The best checkpoint for each run was selected by Validation chrF++, then evaluated once on the independent Test split.",
        "",
        "## Independent Test summary (mean +/- standard deviation)",
        "",
        "| Tokenizer | chrF++ | SacreBLEU | CER (lower) | Exact match | Normalized exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tokenizer in TOKENIZER_NAMES:
        values = summary["test"][tokenizer]
        lines.append(
            f"| {tokenizer} | {format_mean_std(values['chrf_plus_plus'])} | "
            f"{format_mean_std(values['sacrebleu'])} | {format_mean_std(values['cer'])} | "
            f"{format_mean_std(values['exact_match'])} | "
            f"{format_mean_std(values['normalized_exact_match'])} |"
        )

    lines.extend(
        [
            "",
            "## Changed-only Test summary",
            "",
            "| Tokenizer | chrF++ | SacreBLEU | CER (lower) | Exact match | Normalized exact |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tokenizer in TOKENIZER_NAMES:
        values = summary["changed_only_test"][tokenizer]
        lines.append(
            f"| {tokenizer} | {format_mean_std(values['chrf_plus_plus'])} | "
            f"{format_mean_std(values['sacrebleu'])} | {format_mean_std(values['cer'])} | "
            f"{format_mean_std(values['exact_match'])} | "
            f"{format_mean_std(values['normalized_exact_match'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired improvement: dialect minus baseline",
            "",
            "Negative CER means that the dialect tokenizer has fewer character errors.",
            "",
            "| Baseline | chrF++ | SacreBLEU | CER | Exact match | Normalized exact |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline, values in summary["paired_dialect_minus_baseline"].items():
        lines.append(
            f"| {baseline} | {format_mean_std(values['chrf_plus_plus'])} | "
            f"{format_mean_std(values['sacrebleu'])} | {format_mean_std(values['cer'])} | "
            f"{format_mean_std(values['exact_match'])} | "
            f"{format_mean_std(values['normalized_exact_match'])} |"
        )

    lines.extend(
        [
            "",
            "## Runtime and target-process GPU metrics",
            "",
            "Process GPU values exclude other users when NVIDIA per-process telemetry is available. Torch peaks are always measured inside the training process.",
            "",
            "| Tokenizer | Train time (h) | Total stage (h) | Test time (min) | Torch allocated (GB) | Torch reserved (GB) | Process VRAM (GB) | Active process SM |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tokenizer, values in summary["runtime_and_process_gpu"].items():
        lines.append(
            f"| {tokenizer} | {format_mean_std(values['train_wall_hours'], 2)} | "
            f"{format_optional(values['total_stage_wall_hours'], 2)} | "
            f"{format_mean_std(values['test_wall_minutes'], 2)} | "
            f"{format_mean_std(values['torch_peak_allocated_gb'], 2)} | "
            f"{format_mean_std(values['torch_peak_reserved_gb'], 2)} | "
            f"{format_optional(values['peak_process_gpu_memory_gb'], 2)} | "
            f"{format_optional(values['average_active_process_sm_percent'], 1, '%')} |"
        )

    if "shared_decoder" in summary:
        decoder = summary["shared_decoder"]
        process = decoder.get("process_metrics", {})
        gpu = summary.get("shared_decoder_process_gpu", {})
        lines.extend(
            [
                "",
                "### Shared standard-form decoder",
                "",
                f"- Training time: {float(process.get('wall_seconds', 0.0)) / 3600.0:.2f} h",
                f"- Torch peak allocated/reserved: {float(process.get('torch_peak_allocated_gb', 0.0)):.2f}/{float(process.get('torch_peak_reserved_gb', 0.0)):.2f} GB",
                f"- Target-process peak VRAM: {float(gpu['peak_process_gpu_memory_mib']) / 1024.0:.2f} GB"
                if gpu.get("peak_process_gpu_memory_mib") is not None
                else "- Target-process peak VRAM: N/A",
                f"- Target-process active SM: {float(gpu['average_active_process_sm_percent']):.1f}%"
                if gpu.get("average_active_process_sm_percent") is not None
                else "- Target-process active SM: N/A",
                "",
                "The four source MLM encoders were reused from the 260807 controlled tokenizer experiment; their earlier whole-device GPU measurements are not mixed into these process-only results.",
            ]
        )

    lines.extend(["", "## Per-region chrF++", ""])
    for tokenizer in TOKENIZER_NAMES:
        lines.extend(
            [
                f"### {tokenizer}",
                "",
                "| Region | chrF++ mean | chrF++ std |",
                "| --- | ---: | ---: |",
            ]
        )
        for region, values in summary["per_region_chrf_plus_plus"][tokenizer].items():
            lines.append(f"| {region} | {values['mean']:.4f} | {values['std']:.4f} |")
        lines.append("")

    (result_dir / "final_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Final result summary: {result_dir / 'final_results.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize five-seed translation results.")
    parser.add_argument("--outputs_root", default="./outputs")
    parser.add_argument("--logs_dir", default="./logs")
    parser.add_argument("--result_dir", default="./results")
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
