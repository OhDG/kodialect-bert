import json
import argparse
from pathlib import Path
from typing import Dict


METRIC_KEYS = ["loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_classifier_result(classifier_dir: Path) -> Dict[str, float]:
    metrics = load_json(classifier_dir / "eval_metrics_simple.json")
    result = {key: float(metrics[key]) for key in METRIC_KEYS if key in metrics}
    return result


def format_value(value) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dialect vs KLUE tokenizer accuracy-focused classifier results.")
    parser.add_argument("--dialect_classifier_dir", type=str, default="./dialect_small_bert_region_classifier_epoch3_accuracy")
    parser.add_argument("--klue_classifier_dir", type=str, default="./klue_small_bert_region_classifier_epoch3_accuracy")
    parser.add_argument("--output_md", type=str, default="./tokenizer_accuracy_comparison.md")
    parser.add_argument("--output_json", type=str, default="./tokenizer_accuracy_comparison.json")
    args = parser.parse_args()

    dialect = load_classifier_result(Path(args.dialect_classifier_dir))
    klue = load_classifier_result(Path(args.klue_classifier_dir))
    diff = {
        key: (dialect[key] - klue[key]) if dialect.get(key) is not None and klue.get(key) is not None else None
        for key in sorted(set(dialect) | set(klue))
    }

    rows = [
        ("Classifier eval loss", "loss"),
        ("Accuracy", "accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]

    lines = [
        "| Metric | Dialect tokenizer | KLUE tokenizer | Dialect - KLUE |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        lines.append(
            f"| {label} | {format_value(dialect.get(key))} | {format_value(klue.get(key))} | {format_value(diff.get(key))} |"
        )

    output = "\n".join(lines) + "\n"
    print(output)

    Path(args.output_md).write_text(output, encoding="utf-8")
    Path(args.output_json).write_text(
        json.dumps({"dialect": dialect, "klue": klue, "diff": diff}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
