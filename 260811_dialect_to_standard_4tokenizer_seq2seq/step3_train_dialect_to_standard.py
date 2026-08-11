import argparse
import csv
import gzip
import inspect
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import (
    AutoModel,
    BertLMHeadModel,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from experiment_common import (
    REGION_LABELS,
    configure_cuda,
    configure_reproducibility,
    enable_trusted_local_resume,
    file_signature,
    finish_process_measurement,
    generation_metrics,
    load_local_tokenizer,
    save_json,
    save_tokenizer_compatible,
    start_process_measurement,
    subset_generation_metrics,
    tokenizer_fingerprint,
)


class DualTokenizerSeq2SeqCollator:
    def __init__(self, source_tokenizer, target_tokenizer, model, pad_to_multiple_of=8):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.model = model
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        source_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.source_tokenizer.pad(
            source_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = [list(feature["labels"]) for feature in features]
        maximum = max(len(label) for label in labels)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            maximum = ((maximum + multiple - 1) // multiple) * multiple
        label_padding = -100
        if self.target_tokenizer.padding_side == "right":
            labels = [label + [label_padding] * (maximum - len(label)) for label in labels]
        else:
            labels = [[label_padding] * (maximum - len(label)) + label for label in labels]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch["labels"]
            )
            batch["decoder_attention_mask"] = batch["decoder_input_ids"].ne(
                self.target_tokenizer.pad_token_id
            )
        return batch


def stratified_indices(regions: Sequence[str], limit: int, seed: int) -> List[int]:
    if limit >= len(regions):
        return list(range(len(regions)))
    grouped: Dict[str, List[int]] = {region: [] for region in REGION_LABELS}
    for index, region in enumerate(regions):
        grouped.setdefault(region, []).append(index)

    exact = {
        region: len(indices) * limit / len(regions)
        for region, indices in grouped.items()
        if indices
    }
    quotas = {region: int(value) for region, value in exact.items()}
    remaining = limit - sum(quotas.values())
    ranked = sorted(exact, key=lambda region: exact[region] - quotas[region], reverse=True)
    for region in ranked[:remaining]:
        quotas[region] += 1

    rng = random.Random(seed)
    selected = []
    for region, indices in grouped.items():
        indices = list(indices)
        rng.shuffle(indices)
        selected.extend(indices[: quotas.get(region, 0)])
    rng.shuffle(selected)
    return selected


def expected_cache_metadata(args, source_tokenizer, target_tokenizer):
    return {
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "source_tokenizer_fingerprint": tokenizer_fingerprint(source_tokenizer),
        "target_tokenizer_fingerprint": tokenizer_fingerprint(target_tokenizer),
        "files": {
            "train": file_signature(Path(args.train_tsv)),
            "validation": file_signature(Path(args.validation_tsv)),
            "test": file_signature(Path(args.test_tsv)),
        },
        "sample_limits": {
            "train": args.max_train_samples,
            "validation_generation": args.validation_generation_samples,
            "test": args.max_test_samples,
        },
        "validation_selection_seed": args.validation_selection_seed,
    }


def write_metadata_sidecar(path: Path, dataset) -> None:
    fields = ["source_text", "target_text", "region", "file_id", "is_changed"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in dataset:
            writer.writerow({field: row[field] for field in fields})


def read_metadata_sidecar(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["is_changed"] = str(row["is_changed"]).lower() in {"1", "true", "yes"}
    return rows


def load_or_create_tokenized_dataset(args, source_tokenizer, target_tokenizer):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("Install datasets before translation fine-tuning.") from exc

    cache_dir = Path(args.tokenized_cache_dir)
    metadata_path = cache_dir / "cache_metadata.json"
    validation_sidecar = cache_dir / "validation_metadata.tsv"
    test_sidecar = cache_dir / "test_metadata.tsv"
    expected = expected_cache_metadata(args, source_tokenizer, target_tokenizer)
    if (
        (cache_dir / "dataset_dict.json").is_file()
        and metadata_path.is_file()
        and validation_sidecar.is_file()
        and test_sidecar.is_file()
    ):
        with metadata_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
        if actual == expected:
            print(f"[CACHE] Loading tokenized translation data: {cache_dir}")
            return (
                load_from_disk(str(cache_dir)),
                read_metadata_sidecar(validation_sidecar),
                read_metadata_sidecar(test_sidecar),
            )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    Path(args.dataset_cache_dir).mkdir(parents=True, exist_ok=True)
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
    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(
            range(min(args.max_train_samples, len(raw["train"])))
        )
    if (
        args.validation_generation_samples is not None
        and args.validation_generation_samples < len(raw["validation"])
    ):
        indices = stratified_indices(
            raw["validation"]["region"],
            args.validation_generation_samples,
            args.validation_selection_seed,
        )
        raw["validation"] = raw["validation"].select(indices)
    if args.max_test_samples is not None:
        raw["test"] = raw["test"].select(
            range(min(args.max_test_samples, len(raw["test"])))
        )

    validation_rows = raw["validation"]
    test_rows = raw["test"]
    target_eos = target_tokenizer.sep_token_id
    if target_eos is None:
        raise ValueError("Target tokenizer must define sep_token_id as EOS.")

    def preprocess(batch):
        model_inputs = source_tokenizer(
            batch["source_text"],
            truncation=True,
            max_length=args.max_source_length,
            padding=False,
        )
        target_ids = target_tokenizer(
            batch["target_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_target_length - 1,
            padding=False,
        )["input_ids"]
        model_inputs["labels"] = [ids + [target_eos] for ids in target_ids]
        model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]
        return model_inputs

    tokenized = raw.map(
        preprocess,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing dialect-to-standard translation data",
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_dir))
    save_json(metadata_path, expected)
    write_metadata_sidecar(validation_sidecar, validation_rows)
    write_metadata_sidecar(test_sidecar, test_rows)
    print(f"[CACHE] Tokenized translation data saved: {cache_dir}")
    return (
        tokenized,
        read_metadata_sidecar(validation_sidecar),
        read_metadata_sidecar(test_sidecar),
    )


def load_model(args, source_tokenizer, target_tokenizer):
    source_dir = Path(args.source_mlm_model_dir)
    decoder_dir = Path(args.shared_decoder_model_dir)
    for path, name in ((source_dir, "source MLM encoder"), (decoder_dir, "shared decoder")):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    configure_reproducibility(args.seed)
    encoder = AutoModel.from_pretrained(
        str(source_dir), trust_remote_code=True, add_pooling_layer=False
    )
    decoder = BertLMHeadModel.from_pretrained(str(decoder_dir))
    decoder.config.is_decoder = True
    decoder.config.add_cross_attention = True
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id = target_tokenizer.cls_token_id
    model.config.eos_token_id = target_tokenizer.sep_token_id
    model.config.pad_token_id = target_tokenizer.pad_token_id
    model.config.vocab_size = decoder.config.vocab_size
    model.config.tie_encoder_decoder = False
    model.encoder.config.pad_token_id = source_tokenizer.pad_token_id
    model.decoder.config.pad_token_id = target_tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = target_tokenizer.cls_token_id
    model.generation_config.eos_token_id = target_tokenizer.sep_token_id
    model.generation_config.pad_token_id = target_tokenizer.pad_token_id
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.num_beams = args.generation_num_beams
    return model


def build_training_arguments(args) -> Seq2SeqTrainingArguments:
    supported = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
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
        "metric_for_best_model": "chrf_plus_plus",
        "greater_is_better": True,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.generation_num_beams,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "fp16_full_eval": args.fp16,
        "optim": args.optim,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "group_by_length": True,
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
    return Seq2SeqTrainingArguments(
        **{key: value for key, value in kwargs.items() if key in supported}
    )


def decode_predictions(predictions, labels, target_tokenizer):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.asarray(labels).copy()
    labels[labels == -100] = target_tokenizer.pad_token_id
    prediction_text = target_tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    reference_text = target_tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return [text.strip() for text in prediction_text], [text.strip() for text in reference_text]


def build_compute_metrics(target_tokenizer):
    def compute_metrics(eval_prediction):
        predictions, references = decode_predictions(
            eval_prediction.predictions,
            eval_prediction.label_ids,
            target_tokenizer,
        )
        return generation_metrics(predictions, references)

    return compute_metrics


def build_trainer(model, args, datasets, source_tokenizer, target_tokenizer):
    kwargs = {
        "model": model,
        "args": build_training_arguments(args),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": DualTokenizerSeq2SeqCollator(
            source_tokenizer, target_tokenizer, model, pad_to_multiple_of=8
        ),
        "compute_metrics": build_compute_metrics(target_tokenizer),
    }
    parameters = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = target_tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = target_tokenizer
    return Seq2SeqTrainer(**kwargs)


def save_generation_report(
    trainer,
    dataset,
    metadata_rows,
    split,
    output_dir,
    target_tokenizer,
    generation_max_length,
    generation_num_beams,
):
    measurement = start_process_measurement()
    prediction_output = trainer.predict(
        dataset,
        metric_key_prefix=split,
        max_length=generation_max_length,
        num_beams=generation_num_beams,
    )
    process_metrics = finish_process_measurement(measurement)
    predictions, _decoded_references = decode_predictions(
        prediction_output.predictions,
        prediction_output.label_ids,
        target_tokenizer,
    )
    if len(predictions) != len(metadata_rows):
        raise RuntimeError(
            f"Prediction/metadata mismatch: {len(predictions)} != {len(metadata_rows)}"
        )
    changed = [bool(row["is_changed"]) for row in metadata_rows]
    regions = [str(row["region"]) for row in metadata_rows]
    references = [str(row["target_text"]) for row in metadata_rows]
    report = subset_generation_metrics(predictions, references, changed, regions)
    report["runtime"] = process_metrics
    report["trainer_metrics"] = {
        key: float(value)
        for key, value in prediction_output.metrics.items()
        if isinstance(value, (int, float))
    }
    identity_predictions = [str(row["source_text"]) for row in metadata_rows]
    report["identity_input_baseline"] = subset_generation_metrics(
        identity_predictions, references, changed, regions
    )
    save_json(output_dir / f"{split}_generation_report.json", report)

    samples_path = output_dir / f"{split}_predictions.tsv.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["region", "is_changed", "source_text", "reference", "prediction"]
        )
        for row, reference, prediction in zip(metadata_rows, references, predictions):
            writer.writerow(
                [
                    row["region"],
                    int(bool(row["is_changed"])),
                    row["source_text"],
                    reference,
                    prediction,
                ]
            )
    return report


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    enable_trusted_local_resume(args.resume_from_checkpoint, args.output_dir)
    cuda_info = configure_cuda(args.tf32)
    configure_reproducibility(args.seed)
    for path in (Path(args.train_tsv), Path(args.validation_tsv), Path(args.test_tsv)):
        if not path.is_file():
            raise FileNotFoundError(f"Translation TSV not found: {path}")

    source_tokenizer = load_local_tokenizer(
        Path(args.source_mlm_model_dir), args.max_source_length
    )
    target_tokenizer = load_local_tokenizer(
        Path(args.shared_decoder_model_dir), args.max_target_length
    )
    datasets, validation_rows, test_rows = load_or_create_tokenized_dataset(
        args, source_tokenizer, target_tokenizer
    )
    model = load_model(args, source_tokenizer, target_tokenizer)
    trainer = build_trainer(
        model, args, datasets, source_tokenizer, target_tokenizer
    )

    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    print("\n--- Dialect-to-standard encoder-decoder fine-tuning ---")
    print(f"tokenizer: {args.tokenizer_name}, seed: {args.seed}")
    print(f"source encoder: {args.source_mlm_model_dir}")
    print(f"shared decoder: {args.shared_decoder_model_dir}")
    print(f"source/target vocab: {len(source_tokenizer):,}/{len(target_tokenizer):,}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(
        f"micro/effective train batch: {args.train_batch_size}/{effective_batch}, "
        f"eval batch: {args.eval_batch_size}"
    )
    print(
        f"validation generation samples: {len(datasets['validation']):,}, "
        f"independent test samples: {len(datasets['test']):,}"
    )

    train_measurement = start_process_measurement()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_process_metrics = finish_process_measurement(train_measurement)

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    save_tokenizer_compatible(source_tokenizer, final_dir / "source_tokenizer")
    save_tokenizer_compatible(target_tokenizer, final_dir / "target_tokenizer")
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    validation_report = save_generation_report(
        trainer,
        datasets["validation"],
        validation_rows,
        "validation",
        output_dir,
        target_tokenizer,
        args.generation_max_length,
        args.generation_num_beams,
    )
    test_report = save_generation_report(
        trainer,
        datasets["test"],
        test_rows,
        "test",
        output_dir,
        target_tokenizer,
        args.generation_max_length,
        args.generation_num_beams,
    )

    metadata = {
        "completed": True,
        "tokenizer_name": args.tokenizer_name,
        "seed": args.seed,
        "source_mlm_model_dir": args.source_mlm_model_dir,
        "shared_decoder_model_dir": args.shared_decoder_model_dir,
        "source_vocab_size": len(source_tokenizer),
        "target_vocab_size": len(target_tokenizer),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_train_batch_size": effective_batch,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_chrf_plus_plus": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "train_process_metrics": train_process_metrics,
        "validation_metrics": validation_report["overall"],
        "test_metrics": test_report["overall"],
        "cuda": cuda_info,
        "arguments": vars(args),
    }
    save_json(output_dir / "experiment_metadata.json", metadata)
    print("\n=== Final independent Test metrics ===")
    for key, value in test_report["overall"].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"[OK] Best-validation translation model saved: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and independently test one dialect-to-standard model."
    )
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--source_mlm_model_dir", required=True)
    parser.add_argument("--shared_decoder_model_dir", required=True)
    parser.add_argument("--train_tsv", default="./data/translation/translation_train.tsv")
    parser.add_argument(
        "--validation_tsv", default="./data/translation/translation_validation.tsv"
    )
    parser.add_argument("--test_tsv", default="./data/translation/translation_test.tsv")
    parser.add_argument("--dataset_cache_dir", default="./cache/huggingface_datasets")
    parser.add_argument("--tokenized_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument("--eval_accumulation_steps", type=int, default=8)

    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--tokenize_batch_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation_selection_seed", type=int, default=42)
    parser.add_argument("--validation_generation_samples", type=int, default=50000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
