import argparse
import inspect
import os
from pathlib import Path

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from experiment_common import (
    collect_process_gpu_memory_stats,
    TOKENIZER_SPECS,
    build_small_bert_config,
    configure_cuda,
    configure_reproducibility,
    enable_trusted_local_resume,
    initialize_mlm_model_with_shared_backbone,
    load_tokenizer,
    model_state_sha256,
    reset_process_gpu_memory_stats,
    save_json,
    save_tokenizer_compatible,
    shared_backbone_sha256,
)


def load_tokenized_corpus(args: argparse.Namespace, tokenizer):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets before running MLM pretraining.") from exc

    data_files = {
        "train": args.train_corpus,
        "validation": args.validation_corpus,
    }
    Path(args.dataset_cache_dir).mkdir(parents=True, exist_ok=True)
    raw = load_dataset("text", data_files=data_files, cache_dir=args.dataset_cache_dir)
    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_validation_samples is not None:
        raw["validation"] = raw["validation"].select(
            range(min(args.max_validation_samples, len(raw["validation"])))
        )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    return raw.map(
        tokenize,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw["train"].column_names,
        desc=f"Tokenizing MLM corpus ({args.tokenizer_name})",
    )


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
        "fp16": args.fp16,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": args.optim,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
        "prediction_loss_only": True,
    }
    if args.dataloader_num_workers > 0:
        kwargs["dataloader_persistent_workers"] = True
        kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    if "evaluation_strategy" not in supported and "eval_strategy" in supported:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def build_trainer(model, training_args, datasets, tokenizer, data_collator) -> Trainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": data_collator,
    }
    parameters = set(inspect.signature(Trainer.__init__).parameters)
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    return Trainer(**kwargs)


def pretrain(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    enable_trusted_local_resume(args.resume_from_checkpoint, args.output_dir)
    cuda_info = configure_cuda(args.tf32)
    reset_process_gpu_memory_stats()
    configure_reproducibility(args.seed)

    for path in (Path(args.train_corpus), Path(args.validation_corpus)):
        if not path.is_file():
            raise FileNotFoundError(f"Corpus not found: {path}")

    tokenizer, tokenizer_source = load_tokenizer(
        args.tokenizer_name,
        args.dialect_tokenizer_dir,
        args.max_length,
    )
    config = build_small_bert_config(
        tokenizer,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
    )
    model = initialize_mlm_model_with_shared_backbone(config, args.seed)
    full_initialization_hash = model_state_sha256(model)
    backbone_initialization_hash = shared_backbone_sha256(model)
    tokenized = load_tokenized_corpus(args, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8,
    )
    training_args = build_training_arguments(args)
    trainer = build_trainer(model, training_args, tokenized, tokenizer, collator)

    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    print("\n--- Small BERT MLM pretraining from scratch ---")
    print(f"tokenizer: {args.tokenizer_name} ({tokenizer_source})")
    print(f"vocab_size: {len(tokenizer):,}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(
        f"micro/effective train batch: {args.train_batch_size}/{effective_batch}, "
        f"eval batch: {args.eval_batch_size}"
    )
    print(f"shared_backbone_sha256: {backbone_initialization_hash}")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    save_tokenizer_compatible(tokenizer, final_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("validation", eval_metrics)
    process_gpu_memory = collect_process_gpu_memory_stats()

    metadata = {
        "completed": True,
        "training_type": "small_bert_mlm_from_scratch",
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_source": tokenizer_source,
        "vocab_size": len(tokenizer),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "full_initialization_sha256": full_initialization_hash,
        "shared_backbone_sha256": backbone_initialization_hash,
        "effective_train_batch_size": effective_batch,
        "train_metrics": train_result.metrics,
        "validation_metrics": eval_metrics,
        "cuda": cuda_info,
        "process_gpu_memory": process_gpu_memory,
        "arguments": vars(args),
        "config": config.to_dict(),
    }
    save_json(output_dir / "mlm_pretraining_metadata.json", metadata)
    print(f"[OK] MLM model saved: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain one tokenizer-specific Small BERT with MLM.")
    parser.add_argument("--tokenizer_name", choices=sorted(TOKENIZER_SPECS), required=True)
    parser.add_argument("--dialect_tokenizer_dir", default="./dialect_bert_tokenizer")
    parser.add_argument("--train_corpus", default="./data/corpus/dialect_train_corpus.txt")
    parser.add_argument("--validation_corpus", default="./data/corpus/dialect_validation_corpus.txt")
    parser.add_argument("--dataset_cache_dir", default="./cache/huggingface_datasets")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=250)

    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--tokenize_batch_size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
