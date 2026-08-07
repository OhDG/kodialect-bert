import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

from experiment_common import REGION_LABELS, save_json


METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def collect_results(args: argparse.Namespace) -> Dict[str, object]:
    output_root = Path(args.output_root)
    logs_dir = Path(args.logs_dir)
    results: Dict[str, object] = {"seeds": args.seeds, "tokenizers": {}}
    for tokenizer_name in args.tokenizers:
        mlm_metadata = load_json(output_root / "mlm" / tokenizer_name / "mlm_pretraining_metadata.json")
        mlm_gpu_path = logs_dir / f"03_mlm_{tokenizer_name}_gpu_summary.json"
        mlm_gpu = load_json(mlm_gpu_path) if mlm_gpu_path.is_file() else {}
        seed_results = []
        for seed in args.seeds:
            run_dir = output_root / "classifiers" / tokenizer_name / f"seed_{seed}"
            report_path = run_dir / "test_classification_report.json"
            metadata_path = run_dir / "experiment_metadata.json"
            if not report_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing completed result: {run_dir}")
            report = load_json(report_path)
            metadata = load_json(metadata_path)
            gpu_path = logs_dir / f"04_classifier_{tokenizer_name}_seed_{seed}_gpu_summary.json"
            gpu = load_json(gpu_path) if gpu_path.is_file() else {}
            seed_results.append(
                {
                    "seed": seed,
                    "metrics": {metric: float(report[metric]) for metric in METRICS},
                    "per_label": report["per_label"],
                    "best_model_checkpoint": metadata.get("best_model_checkpoint"),
                    "best_validation_macro_f1": metadata.get("best_validation_macro_f1"),
                    "train_runtime": metadata.get("train_metrics", {}).get("train_runtime"),
                    "test_runtime": report.get("runtime", {}).get("wall_runtime_seconds"),
                    "gpu": gpu,
                }
            )

        summary = {
            metric: mean_std([item["metrics"][metric] for item in seed_results])
            for metric in METRICS
        }
        per_label_summary = {
            region: {
                metric: mean_std(
                    [float(item["per_label"][region][metric]) for item in seed_results]
                )
                for metric in ("precision", "recall", "f1")
            }
            for region in REGION_LABELS
        }
        results["tokenizers"][tokenizer_name] = {
            "vocab_size": mlm_metadata.get("vocab_size"),
            "parameter_count": mlm_metadata.get("parameter_count"),
            "mlm_train_runtime": mlm_metadata.get("train_metrics", {}).get("train_runtime"),
            "mlm_gpu": mlm_gpu,
            "runs": seed_results,
            "summary": summary,
            "per_label_summary": per_label_summary,
            "efficiency": {
                "classification_train_runtime": mean_std(
                    [float(item["train_runtime"]) for item in seed_results]
                ),
                "test_runtime": mean_std([float(item["test_runtime"]) for item in seed_results]),
                "classification_average_active_gpu_utilization_percent": mean_std(
                    [
                        float(item["gpu"].get("average_active_gpu_utilization_percent", 0.0))
                        for item in seed_results
                    ]
                ),
                "classification_peak_memory_used_mib": max(
                    float(item["gpu"].get("peak_memory_used_mib", 0.0)) for item in seed_results
                ),
                "classification_average_power_w": mean_std(
                    [float(item["gpu"].get("average_power_w", 0.0)) for item in seed_results]
                ),
            },
        }

    dialect_runs = {
        item["seed"]: item for item in results["tokenizers"]["dialect"]["runs"]
    }
    paired = {}
    for baseline in args.tokenizers:
        if baseline == "dialect":
            continue
        baseline_runs = {
            item["seed"]: item for item in results["tokenizers"][baseline]["runs"]
        }
        paired[baseline] = {
            metric: mean_std(
                [
                    dialect_runs[seed]["metrics"][metric]
                    - baseline_runs[seed]["metrics"][metric]
                    for seed in args.seeds
                ]
            )
            for metric in METRICS
        }
    results["dialect_minus_baseline"] = paired
    return results


def write_csv_files(results: Dict[str, object], output_dir: Path) -> None:
    with (output_dir / "overall_by_seed.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", "seed", *METRICS, "train_runtime", "test_runtime"])
        for tokenizer_name, tokenizer_result in results["tokenizers"].items():
            for run in tokenizer_result["runs"]:
                writer.writerow(
                    [
                        tokenizer_name,
                        run["seed"],
                        *[run["metrics"][metric] for metric in METRICS],
                        run["train_runtime"],
                        run["test_runtime"],
                    ]
                )

    with (output_dir / "overall_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", *[f"{metric}_mean" for metric in METRICS], *[f"{metric}_std" for metric in METRICS]])
        for tokenizer_name, tokenizer_result in results["tokenizers"].items():
            writer.writerow(
                [tokenizer_name]
                + [tokenizer_result["summary"][metric]["mean"] for metric in METRICS]
                + [tokenizer_result["summary"][metric]["std"] for metric in METRICS]
            )

    with (output_dir / "efficiency_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tokenizer",
                "vocab_size",
                "parameter_count",
                "mlm_train_runtime_seconds",
                "mlm_average_active_gpu_utilization_percent",
                "mlm_peak_memory_used_mib",
                "classification_train_runtime_mean_seconds",
                "classification_train_runtime_std_seconds",
                "classification_average_active_gpu_utilization_percent",
                "classification_peak_memory_used_mib",
                "test_runtime_mean_seconds",
            ]
        )
        for tokenizer_name, tokenizer_result in results["tokenizers"].items():
            efficiency = tokenizer_result["efficiency"]
            writer.writerow(
                [
                    tokenizer_name,
                    tokenizer_result["vocab_size"],
                    tokenizer_result["parameter_count"],
                    tokenizer_result["mlm_train_runtime"],
                    tokenizer_result["mlm_gpu"].get("average_active_gpu_utilization_percent"),
                    tokenizer_result["mlm_gpu"].get("peak_memory_used_mib"),
                    efficiency["classification_train_runtime"]["mean"],
                    efficiency["classification_train_runtime"]["std"],
                    efficiency["classification_average_active_gpu_utilization_percent"]["mean"],
                    efficiency["classification_peak_memory_used_mib"],
                    efficiency["test_runtime"]["mean"],
                ]
            )


def write_markdown(results: Dict[str, object], output_path: Path) -> None:
    lines = [
        "# Four-tokenizer, five-seed region classification results",
        "",
        "## Independent Test summary (mean +/- standard deviation)",
        "",
        "| Tokenizer | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tokenizer_name, tokenizer_result in results["tokenizers"].items():
        cells = []
        for metric in METRICS:
            value = tokenizer_result["summary"][metric]
            cells.append(f"{value['mean']:.6f} +/- {value['std']:.6f}")
        lines.append(f"| {tokenizer_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Paired improvement: dialect minus baseline",
            "",
            "| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline, baseline_result in results["dialect_minus_baseline"].items():
        cells = []
        for metric in METRICS:
            value = baseline_result[metric]
            cells.append(f"{value['mean']:+.6f} +/- {value['std']:.6f}")
        lines.append(f"| {baseline} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Runtime and GPU efficiency",
            "",
            "| Tokenizer | Vocab | Params | MLM time | MLM active GPU | MLM peak VRAM | Classifier time | Classifier active GPU | Classifier peak VRAM | Test time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tokenizer_name, tokenizer_result in results["tokenizers"].items():
        efficiency = tokenizer_result["efficiency"]
        mlm_gpu = tokenizer_result["mlm_gpu"]
        lines.append(
            f"| {tokenizer_name} | {tokenizer_result['vocab_size']:,} | "
            f"{tokenizer_result['parameter_count']:,} | "
            f"{tokenizer_result['mlm_train_runtime'] / 3600:.2f} h | "
            f"{mlm_gpu.get('average_active_gpu_utilization_percent', 0.0):.1f}% | "
            f"{mlm_gpu.get('peak_memory_used_mib', 0.0) / 1024:.2f} GB | "
            f"{efficiency['classification_train_runtime']['mean'] / 3600:.2f} +/- "
            f"{efficiency['classification_train_runtime']['std'] / 3600:.2f} h | "
            f"{efficiency['classification_average_active_gpu_utilization_percent']['mean']:.1f}% | "
            f"{efficiency['classification_peak_memory_used_mib'] / 1024:.2f} GB | "
            f"{efficiency['test_runtime']['mean'] / 60:.2f} min |"
        )

    lines.extend(["", "## Per-region F1", ""])
    for tokenizer_name, tokenizer_result in results["tokenizers"].items():
        lines.append(f"### {tokenizer_name}")
        lines.append("")
        lines.append("| Region | F1 mean | F1 std |")
        lines.append("| --- | ---: | ---: |")
        for region in REGION_LABELS:
            value = tokenizer_result["per_label_summary"][region]["f1"]
            lines.append(f"| {region} | {value['mean']:.6f} | {value['std']:.6f} |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(args)
    save_json(output_dir / "final_results.json", results)
    write_csv_files(results, output_dir)
    write_markdown(results, output_dir / "final_results.md")
    print(f"[OK] Final results: {output_dir / 'final_results.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate four-tokenizer, five-seed Test results.")
    parser.add_argument("--output_root", default="./outputs")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--logs_dir", default="./logs")
    parser.add_argument("--tokenizers", nargs="+", default=["dialect", "klue", "kobert", "mbert"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42, 87, 100])
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
