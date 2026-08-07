import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from experiment_common import (
    ID2LABEL,
    LABEL2ID,
    REGION_LABELS,
    classification_metrics,
    configure_cuda,
    configure_reproducibility,
    enable_trusted_local_resume,
    patch_legacy_tokenizer_save_vocabulary,
    reset_classification_modules,
    save_json,
    save_tokenizer_compatible,
)


def tokenizer_fingerprint(tokenizer) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def expected_cache_metadata(args: argparse.Namespace, tokenizer) -> Dict[str, object]:
    return {
        "max_length": args.max_length,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "vocab_size": len(tokenizer),
        "files": {
            "train": file_signature(Path(args.train_tsv)),
            "validation": file_signature(Path(args.validation_tsv)),
            "test": file_signature(Path(args.test_tsv)),
        },
        "sample_limits": {
            "train": args.max_train_samples,
            "validation": args.max_validation_samples,
            "test": args.max_test_samples,
        },
    }


def load_or_create_tokenized_dataset(args: argparse.Namespace, tokenizer):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("Install datasets before fine-tuning.") from exc

    cache_dir = Path(args.tokenized_cache_dir)
    metadata_path = cache_dir / "cache_metadata.json"
    expected = expected_cache_metadata(args, tokenizer)
    if (cache_dir / "dataset_dict.json").is_file() and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
        if actual == expected:
            print(f"[CACHE] Loading tokenized classification dataset: {cache_dir}")
            return load_from_disk(str(cache_dir))

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    raw = load_dataset(
        "csv",
        data_files={
            "train": args.train_tsv,
            "validation": args.validation_tsv,
            "test": args.test_tsv,
        },
        delimiter="\t",
        cache_dir=args.dataset_cache_dir,
    )
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }
    for split, limit in limits.items():
        if limit is not None:
            raw[split] = raw[split].select(range(min(limit, len(raw[split]))))

    def preprocess(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokenized["labels"] = [int(label) for label in batch["label"]]
        tokenized["length"] = [len(input_ids) for input_ids in tokenized["input_ids"]]
        return tokenized

    tokenized = raw.map(
        preprocess,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train/validation/test classification data",
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_dir))
    save_json(metadata_path, expected)
    print(f"[CACHE] Tokenized classification dataset saved: {cache_dir}")
    return tokenized


def compute_class_weights(labels, mode: str) -> Dict[str, object]:
    counts = np.bincount([int(label) for label in labels], minlength=len(REGION_LABELS)).astype(np.float64)
    total = float(counts.sum())
    weights = None
    if mode != "none":
        weights = total / (len(REGION_LABELS) * np.maximum(counts, 1.0))
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


def load_model(args: argparse.Namespace):
    model_dir = Path(args.mlm_model_dir)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"MLM final model not found: {model_dir}")
    # The locally saved KoBERT tokenizer carries its reviewed custom tokenizer code.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
    patch_legacy_tokenizer_save_vocabulary(tokenizer)
    config = AutoConfig.from_pretrained(
        str(model_dir),
        num_labels=len(REGION_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )
    config.pad_token_id = tokenizer.pad_token_id
    configure_reproducibility(args.seed)
    model_kwargs = {"config": config, "ignore_mismatched_sizes": True}
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), **model_kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in model_kwargs:
            raise
        print(f"[WARN] Attention implementation fallback to auto/eager: {exc}")
        model_kwargs.pop("attn_implementation")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), **model_kwargs)

    reset_classification_modules(model, args.seed)
    return tokenizer, model


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    metrics, _ = classification_metrics(np.asarray(labels), np.asarray(predictions))
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
        loss_function = torch.nn.CrossEntropyLoss(
            weight=None if self.class_weights is None else self.class_weights.to(logits.device)
        )
        loss = loss_function(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": args.optim,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "group_by_length": False,
        "length_column_name": "length",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": True,
        "report_to": "none",
    }
    if args.dataloader_num_workers > 0:
        kwargs["dataloader_persistent_workers"] = True
        kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    if "evaluation_strategy" not in supported and "eval_strategy" in supported:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def build_trainer(model, args, training_args, datasets, tokenizer, class_weight_info):
    class_weights = None
    if class_weight_info["weights"] is not None:
        class_weights = np.asarray(
            [class_weight_info["weights"][region] for region in REGION_LABELS],
            dtype=np.float32,
        )
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        "compute_metrics": compute_metrics,
    }
    parameters = set(inspect.signature(Trainer.__init__).parameters)
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    trainer_class = WeightedClassificationTrainer if class_weights is not None else Trainer
    if class_weights is not None:
        kwargs["class_weights"] = class_weights
    return trainer_class(**kwargs)


def save_prediction_report(trainer, dataset, split: str, output_dir: Path) -> Dict[str, object]:
    start = time.perf_counter()
    prediction_output = trainer.predict(dataset, metric_key_prefix=split)
    wall_seconds = time.perf_counter() - start
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    labels = np.asarray(prediction_output.label_ids)
    metrics, report = classification_metrics(labels, predictions)
    loss_key = f"{split}_loss"
    if loss_key in prediction_output.metrics:
        metrics["loss"] = float(prediction_output.metrics[loss_key])
        report["loss"] = float(prediction_output.metrics[loss_key])
    metrics["wall_runtime_seconds"] = wall_seconds
    metrics["samples_per_second_wall"] = len(labels) / wall_seconds if wall_seconds else 0.0
    report["runtime"] = metrics.copy()

    save_json(output_dir / f"{split}_classification_report.json", report)
    save_json(output_dir / f"{split}_metrics.json", metrics)
    np.savez_compressed(output_dir / f"{split}_predictions.npz", labels=labels, predictions=predictions)
    with (output_dir / f"{split}_confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *REGION_LABELS])
        for idx, region in ID2LABEL.items():
            writer.writerow([region, *report["confusion_matrix"][idx]])
    return report


def fine_tune(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    enable_trusted_local_resume(args.resume_from_checkpoint, args.output_dir)
    cuda_info = configure_cuda(args.tf32)
    configure_reproducibility(args.seed)
    for path in (Path(args.train_tsv), Path(args.validation_tsv), Path(args.test_tsv)):
        if not path.is_file():
            raise FileNotFoundError(f"Classification TSV not found: {path}")

    tokenizer, model = load_model(args)
    datasets = load_or_create_tokenized_dataset(args, tokenizer)
    class_weight_info = compute_class_weights(datasets["train"]["labels"], args.class_weighting)
    training_args = build_training_arguments(args)
    trainer = build_trainer(model, args, training_args, datasets, tokenizer, class_weight_info)

    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    print("\n--- Five-region classification fine-tuning ---")
    print(f"tokenizer/model: {args.mlm_model_dir}")
    print(f"seed: {args.seed}")
    print(f"vocab_size: {len(tokenizer):,}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"micro/effective train batch: {args.train_batch_size}/{effective_batch}")
    print(f"class weights: {class_weight_info}")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    save_tokenizer_compatible(tokenizer, final_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    validation_report = save_prediction_report(trainer, datasets["validation"], "validation", output_dir)
    test_report = save_prediction_report(trainer, datasets["test"], "test", output_dir)
    trainer_state = trainer.state
    metadata = {
        "completed": True,
        "seed": args.seed,
        "mlm_model_dir": args.mlm_model_dir,
        "vocab_size": len(tokenizer),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_train_batch_size": effective_batch,
        "class_weight_info": class_weight_info,
        "best_model_checkpoint": trainer_state.best_model_checkpoint,
        "best_validation_macro_f1": trainer_state.best_metric,
        "train_metrics": train_result.metrics,
        "validation_metrics": {
            key: validation_report[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        },
        "test_metrics": {
            key: test_report[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        },
        "cuda": cuda_info,
        "arguments": vars(args),
    }
    save_json(output_dir / "experiment_metadata.json", metadata)
    print("\n=== Final independent Test metrics ===")
    for key, value in metadata["test_metrics"].items():
        print(f"{key}: {value:.6f}")
    print(f"[OK] Best-validation model saved: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and independently test one tokenizer/seed model.")
    parser.add_argument("--mlm_model_dir", required=True)
    parser.add_argument("--train_tsv", default="./data/region_classification/dialect_region_train.tsv")
    parser.add_argument(
        "--validation_tsv", default="./data/region_classification/dialect_region_validation.tsv"
    )
    parser.add_argument("--test_tsv", default="./data/region_classification/dialect_region_test.tsv")
    parser.add_argument("--dataset_cache_dir", default="./cache/huggingface_datasets")
    parser.add_argument("--tokenized_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument(
        "--class_weighting", choices=["none", "sqrt_balanced", "balanced"], default="balanced"
    )

    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa"], default="eager")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--tokenize_batch_size", type=int, default=8000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    fine_tune(parse_args())
