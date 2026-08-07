import argparse
import json
from pathlib import Path
from typing import Dict, Optional


METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_last_log_value(state_path: Path, key: str) -> Optional[float]:
    if not state_path.exists():
        return None

    state = load_json(state_path)
    for item in reversed(state.get("log_history", [])):
        if key in item:
            return item[key]
    return None


def load_classifier_result(classifier_dir: Path) -> Dict[str, Optional[float]]:
    metrics = load_json(classifier_dir / "eval_metrics_simple.json")
    result = {key: float(metrics[key]) for key in METRIC_KEYS}
    result["classification_eval_loss"] = find_last_log_value(classifier_dir / "trainer_state.json", "eval_loss")
    result["classification_train_runtime"] = find_last_log_value(classifier_dir / "trainer_state.json", "train_runtime")
    return result


def load_mlm_result(mlm_dir: Path) -> Dict[str, Optional[float]]:
    return {
        "mlm_eval_loss": find_last_log_value(mlm_dir / "trainer_state.json", "eval_loss"),
        "mlm_train_runtime": find_last_log_value(mlm_dir / "trainer_state.json", "train_runtime"),
    }


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dialect vs KLUE tokenizer experiment results.")
    parser.add_argument("--dialect_mlm_dir", type=str, default="./dialect_small_bert_mlm_epoch3_continued")
    parser.add_argument("--dialect_classifier_dir", type=str, default="./dialect_small_bert_region_classifier_epoch2_weighted")
    parser.add_argument("--klue_mlm_dir", type=str, default="./klue_small_bert_mlm_epoch3_continued")
    parser.add_argument("--klue_classifier_dir", type=str, default="./klue_small_bert_region_classifier_epoch2_weighted")
    parser.add_argument("--output_md", type=str, default="./tokenizer_comparison.md")
    parser.add_argument("--output_json", type=str, default="./tokenizer_comparison.json")
    args = parser.parse_args()

    dialect = {
        **load_mlm_result(Path(args.dialect_mlm_dir)),
        **load_classifier_result(Path(args.dialect_classifier_dir)),
    }
    klue = {
        **load_mlm_result(Path(args.klue_mlm_dir)),
        **load_classifier_result(Path(args.klue_classifier_dir)),
    }
    diff = {
        key: (dialect[key] - klue[key]) if dialect.get(key) is not None and klue.get(key) is not None else None
        for key in sorted(set(dialect) | set(klue))
    }

    rows = [
        ("MLM eval loss", "mlm_eval_loss"),
        ("Classifier eval loss", "classification_eval_loss"),
        ("Accuracy", "accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("MLM train runtime sec", "mlm_train_runtime"),
        ("Classifier train runtime sec", "classification_train_runtime"),
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
