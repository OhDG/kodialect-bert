import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_NAME = "beomi/kcbert-base"
DEFAULT_DIALECT_TOKENIZER_DIR = "./dialect_bert_tokenizer"
DEFAULT_TRAIN_CORPUS = "../260630_test_1/dialect_train_corpus.txt"
DEFAULT_EVAL_CORPUS = "../260630_test_1/dialect_eval_corpus.txt"
DEFAULT_OUTPUT_DIR = "./kcbert_dialect_tokenizer_mlm"


def resolve_existing_path(primary: str, *fallbacks: str, required: bool = True) -> Optional[Path]:
    for raw_path in (primary, *fallbacks):
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.exists():
            return path

    if required:
        candidates = "\n".join(str(Path(p)) for p in (primary, *fallbacks) if p is not None)
        raise FileNotFoundError(f"Could not find required path. Checked:\n{candidates}")
    return None


def load_dialect_tokenizer(tokenizer_dir: str, max_length: int) -> BertTokenizerFast:
    vocab_path = Path(tokenizer_dir) / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Could not find dialect tokenizer vocab: {vocab_path}\n"
            "Place the tokenizer at ./dialect_bert_tokenizer/vocab.txt or pass --dialect_tokenizer_dir."
        )

    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=False,
        strip_accents=False,
        tokenize_chinese_chars=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    tokenizer.model_max_length = max_length
    return tokenizer


def reinitialize_word_embeddings(model, pad_token_id: Optional[int]) -> None:
    initializer_range = getattr(model.config, "initializer_range", 0.02)
    input_embeddings = model.get_input_embeddings()

    with torch.no_grad():
        input_embeddings.weight.normal_(mean=0.0, std=initializer_range)
        if pad_token_id is not None and 0 <= pad_token_id < input_embeddings.weight.size(0):
            input_embeddings.weight[pad_token_id].zero_()

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
            output_embeddings.weight.normal_(mean=0.0, std=initializer_range)

        if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
            predictions = model.cls.predictions
            if hasattr(predictions, "bias") and predictions.bias is not None:
                predictions.bias.zero_()
            decoder = getattr(predictions, "decoder", None)
            if decoder is not None and getattr(decoder, "bias", None) is not None:
                decoder.bias.zero_()

    model.tie_weights()


def load_mlm_model(args: argparse.Namespace, tokenizer: BertTokenizerFast):
    config = AutoConfig.from_pretrained(args.model_name)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    if args.reinit_word_embeddings:
        print("[INFO] Reinitializing word embeddings for the dialect tokenizer vocabulary.")
        reinitialize_word_embeddings(model, tokenizer.pad_token_id)

    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def require_datasets_package():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets first: pip install datasets") from e
    return load_dataset


def load_and_tokenize_corpus(args: argparse.Namespace, tokenizer: BertTokenizerFast):
    load_dataset = require_datasets_package()

    train_corpus = resolve_existing_path(args.train_corpus, "./dialect_train_corpus.txt")
    eval_corpus = resolve_existing_path(args.eval_corpus, "./dialect_eval_corpus.txt", required=False)

    data_files = {"train": str(train_corpus)}
    if eval_corpus is not None and not args.no_eval:
        data_files["validation"] = str(eval_corpus)

    print(f"[INFO] Train corpus: {train_corpus}")
    if "validation" in data_files:
        print(f"[INFO] Eval corpus:  {eval_corpus}")

    raw_datasets = load_dataset("text", data_files=data_files)

    if args.max_train_samples is not None:
        train_size = min(args.max_train_samples, len(raw_datasets["train"]))
        raw_datasets["train"] = raw_datasets["train"].select(range(train_size))

    if "validation" in raw_datasets and args.max_eval_samples is not None:
        eval_size = min(args.max_eval_samples, len(raw_datasets["validation"]))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(eval_size))

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing MLM corpus",
    )
    return tokenized


def build_training_args(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    import inspect

    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())

    eval_strategy = args.eval_strategy if has_eval else "no"
    save_strategy = args.save_strategy

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
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
        "prediction_loss_only": True,
    }

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = eval_strategy

    if "save_strategy" in supported:
        kwargs["save_strategy"] = save_strategy

    if eval_strategy == "steps":
        kwargs["eval_steps"] = args.eval_steps
    if save_strategy == "steps":
        kwargs["save_steps"] = args.save_steps

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**filtered_kwargs)


def build_trainer(model, training_args, tokenized_datasets, tokenizer, data_collator):
    import inspect

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "data_collator": data_collator,
    }

    if "validation" in tokenized_datasets:
        trainer_kwargs["eval_dataset"] = tokenized_datasets["validation"]

    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


def run_mlm_pretraining(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_dialect_tokenizer(args.dialect_tokenizer_dir, args.max_length)
    model = load_mlm_model(args, tokenizer)
    tokenized_datasets = load_and_tokenize_corpus(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    has_eval = "validation" in tokenized_datasets
    training_args = build_training_args(args, has_eval=has_eval)
    trainer = build_trainer(model, training_args, tokenized_datasets, tokenizer, data_collator)

    print("\n--- Continued MLM pretraining start ---")
    print(f"model_name: {args.model_name}")
    print(f"dialect_tokenizer_dir: {args.dialect_tokenizer_dir}")
    print(f"vocab_size: {len(tokenizer)}")
    print(f"max_length: {args.max_length}")
    print(f"mlm_probability: {args.mlm_probability}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "base_model": args.model_name,
        "dialect_tokenizer_dir": args.dialect_tokenizer_dir,
        "vocab_size": len(tokenizer),
        "train_corpus": args.train_corpus,
        "eval_corpus": None if args.no_eval else args.eval_corpus,
        "max_length": args.max_length,
        "mlm_probability": args.mlm_probability,
        "reinit_word_embeddings": args.reinit_word_embeddings,
    }
    with (output_dir / "mlm_pretraining_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] MLM model saved: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue MLM pretraining of KcBERT with the dialect WordPiece tokenizer."
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dialect_tokenizer_dir", type=str, default=DEFAULT_DIALECT_TOKENIZER_DIR)
    parser.add_argument("--train_corpus", type=str, default=DEFAULT_TRAIN_CORPUS)
    parser.add_argument("--eval_corpus", type=str, default=DEFAULT_EVAL_CORPUS)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--reinit_word_embeddings", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    parser.add_argument("--eval_strategy", type=str, choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

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
    run_mlm_pretraining(parse_args())
