import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


REGION_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LABEL2ID = {label: idx for idx, label in enumerate(REGION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

DEFAULT_MLM_MODEL_DIR = "./small_bert_mlm/final_model"
DEFAULT_TRAIN_TSV = "../shared/region_classification_data/dialect_region_train.tsv"
DEFAULT_EVAL_TSV = "../shared/region_classification_data/dialect_region_eval.tsv"
DEFAULT_OUTPUT_DIR = "./small_bert_region_classifier_accuracy"


def resolve_existing_path(primary: str, *fallbacks: str) -> Path:
    for raw_path in (primary, *fallbacks):
        path = Path(raw_path)
        if path.exists():
            return path
    candidates = "\n".join(str(Path(p)) for p in (primary, *fallbacks))
    raise FileNotFoundError(f"Could not find required path. Checked:\n{candidates}")


def load_tokenizer_and_model(args: argparse.Namespace):
    model_dir = resolve_existing_path(args.mlm_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    config = AutoConfig.from_pretrained(
        str(model_dir),
        num_labels=len(REGION_LABELS),
        id2label={idx: label for idx, label in ID2LABEL.items()},
        label2id={label: idx for label, idx in LABEL2ID.items()},
        problem_type="single_label_classification",
    )
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = {idx: label for idx, label in ID2LABEL.items()}
    model.config.label2id = {label: idx for label, idx in LABEL2ID.items()}

    return tokenizer, model


def require_datasets_package():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets first: pip install datasets") from e
    return load_dataset


def compute_class_weight_info(label_values, mode: str) -> Dict[str, object]:
    labels = [int(label) for label in label_values]
    counts = np.bincount(labels, minlength=len(REGION_LABELS)).astype(np.float64)
    total = float(counts.sum())

    if mode == "none":
        weights = None
    else:
        safe_counts = np.maximum(counts, 1.0)
        weights = total / (len(REGION_LABELS) * safe_counts)
        if mode == "sqrt_balanced":
            weights = np.sqrt(weights)
        weights = weights / weights.mean()

    return {
        "mode": mode,
        "counts": {REGION_LABELS[idx]: int(counts[idx]) for idx in range(len(REGION_LABELS))},
        "weights": None
        if weights is None
        else {REGION_LABELS[idx]: float(weights[idx]) for idx in range(len(REGION_LABELS))},
    }


def load_and_tokenize_datasets(args: argparse.Namespace, tokenizer):
    load_dataset = require_datasets_package()

    train_tsv = resolve_existing_path(
        args.train_tsv,
        "../shared/region_classification_data/dialect_region_train.tsv",
        "./region_classification_data/dialect_region_train.tsv",
        "../260630_test_1/region_classification_data/dialect_region_train.tsv",
    )
    eval_tsv = resolve_existing_path(
        args.eval_tsv,
        "../shared/region_classification_data/dialect_region_eval.tsv",
        "./region_classification_data/dialect_region_eval.tsv",
        "../260630_test_1/region_classification_data/dialect_region_eval.tsv",
    )

    print(f"[INFO] Train TSV: {train_tsv}")
    print(f"[INFO] Eval TSV:  {eval_tsv}")

    raw_datasets = load_dataset(
        "csv",
        data_files={"train": str(train_tsv), "validation": str(eval_tsv)},
        delimiter="\t",
    )

    if args.max_train_samples is not None:
        train_size = min(args.max_train_samples, len(raw_datasets["train"]))
        raw_datasets["train"] = raw_datasets["train"].select(range(train_size))

    if args.max_eval_samples is not None:
        eval_size = min(args.max_eval_samples, len(raw_datasets["validation"]))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(eval_size))

    class_weight_info = compute_class_weight_info(raw_datasets["train"]["label"], args.class_weighting)

    def preprocess(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokenized["labels"] = [int(label) for label in batch["label"]]
        return tokenized

    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing classification data",
    )
    return tokenized_datasets, class_weight_info


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> Tuple[Dict[str, float], Dict[str, object]]:
    num_labels = len(REGION_LABELS)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    for true_label, pred_label in zip(labels, preds):
        if 0 <= int(true_label) < num_labels and 0 <= int(pred_label) < num_labels:
            confusion[int(true_label), int(pred_label)] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0

    per_label = {}
    precision_values = []
    recall_values = []
    f1_values = []
    weighted_f1_sum = 0.0

    for idx, region in ID2LABEL.items():
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support

        per_label[region] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(weighted_f1_sum / total) if total else 0.0,
    }
    report = {
        **metrics,
        "labels": REGION_LABELS,
        "per_label": per_label,
        "confusion_matrix": confusion.tolist(),
    }
    return metrics, report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics, _ = classification_metrics(np.asarray(labels), np.asarray(preds))
    return metrics


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[np.ndarray] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=None if self.class_weights is None else self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def save_eval_report(trainer: Trainer, eval_dataset, output_dir: Path) -> None:
    prediction_output = trainer.predict(eval_dataset)
    preds = np.argmax(prediction_output.predictions, axis=-1)
    labels = np.asarray(prediction_output.label_ids)
    metrics, report = classification_metrics(labels, preds)
    prediction_metrics = {
        key.removeprefix("test_"): float(value)
        for key, value in prediction_output.metrics.items()
        if isinstance(value, (int, float, np.floating))
    }
    if "loss" in prediction_metrics:
        metrics["loss"] = prediction_metrics["loss"]
        report["loss"] = prediction_metrics["loss"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with (output_dir / "eval_metrics_simple.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (output_dir / "eval_confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *REGION_LABELS])
        for idx, region in ID2LABEL.items():
            writer.writerow([region, *report["confusion_matrix"][idx]])

    print("\n=== Final eval metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Report saved: {output_dir / 'eval_classification_report.json'}")


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    import inspect

    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())

    load_best_model_at_end = args.load_best_model_at_end and args.eval_strategy != "no"

    kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": True,
        "report_to": "none",
    }

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = args.eval_strategy
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = args.eval_strategy

    if "save_strategy" in supported:
        kwargs["save_strategy"] = args.save_strategy

    if args.eval_strategy == "steps":
        kwargs["eval_steps"] = args.eval_steps
    if args.save_strategy == "steps":
        kwargs["save_steps"] = args.save_steps

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**filtered_kwargs)


def build_trainer(
    model,
    training_args,
    tokenized_datasets,
    tokenizer,
    data_collator,
    class_weight_info: Dict[str, object],
) -> Trainer:
    import inspect

    class_weights = None
    if class_weight_info["weights"] is not None:
        class_weights = np.asarray(
            [class_weight_info["weights"][region] for region in REGION_LABELS],
            dtype=np.float32,
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer_cls = WeightedClassificationTrainer if class_weights is not None else Trainer
    if class_weights is not None:
        trainer_kwargs["class_weights"] = class_weights
    return trainer_cls(**trainer_kwargs)


def train_classifier(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer, model = load_tokenizer_and_model(args)
    tokenized_datasets, class_weight_info = load_and_tokenize_datasets(args, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(args)
    trainer = build_trainer(model, training_args, tokenized_datasets, tokenizer, data_collator, class_weight_info)

    print("\n--- Region classification fine-tuning start ---")
    print(f"mlm_model_dir: {args.mlm_model_dir}")
    print(f"vocab_size: {len(tokenizer):,}")
    print(f"max_length: {args.max_length}")
    print(f"labels: {LABEL2ID}")
    print(f"class_weighting: {class_weight_info}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"label2id": LABEL2ID, "id2label": ID2LABEL, "class_weight_info": class_weight_info},
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_eval_report(trainer, tokenized_datasets["validation"], output_dir)
    print(f"\n[OK] Classifier saved: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small BERT MLM-pretrained model for dialect region classification with accuracy-focused settings."
    )
    parser.add_argument("--mlm_model_dir", type=str, default=DEFAULT_MLM_MODEL_DIR)
    parser.add_argument("--train_tsv", type=str, default=DEFAULT_TRAIN_TSV)
    parser.add_argument("--eval_tsv", type=str, default=DEFAULT_EVAL_TSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument(
        "--class_weighting",
        type=str,
        choices=["none", "sqrt_balanced", "balanced"],
        default="none",
        help="Use weighted cross entropy to reduce region imbalance. For accuracy-focused runs, keep the default none.",
    )

    parser.add_argument("--eval_strategy", type=str, choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_classifier(parse_args())
