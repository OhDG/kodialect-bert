import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


REGION_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LABEL2ID = {label: idx for idx, label in enumerate(REGION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

DEFAULT_MODEL_NAME = "beomi/kcbert-base"
DEFAULT_MANIFEST = "corpus_split_manifest.csv"
DEFAULT_STATS_JSON = "corpus_split_stats.json"
DEFAULT_DIALECT_TOKENIZER_DIR = "./dialect_bert_tokenizer"
DEFAULT_CACHE_DIR = "./region_classification_data"
DEFAULT_OUTPUT_DIR = "./kcbert_dialect_region_classifier"


# ============================================================
# step1_prepare_data_final.py와 동일한 텍스트 정제 로직
# ============================================================
def clean_base_text(text: str) -> str:
    cleaned_text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned_text = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned_text).strip()
    return cleaned_text


def clean_extra_text(text: str) -> str:
    cleaned_text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned_text = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def extract_texts_from_json(file_path: Path, source_type: str) -> List[str]:
    texts: List[str] = []

    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if source_type == "base":
        utterances = data.get("utterance", [])
        for u in utterances:
            text = u.get("dialect_form", "")
            if isinstance(text, str) and text.strip():
                cleaned = clean_base_text(text)
                if cleaned:
                    texts.append(cleaned)

    elif source_type == "extra":
        text = data.get("transcription", {}).get("dialect", "")
        if isinstance(text, str) and text.strip():
            cleaned = clean_extra_text(text)
            if cleaned:
                texts.append(cleaned)

    else:
        raise ValueError(f"지원하지 않는 source_type입니다: {source_type}")

    return texts


# ============================================================
# manifest -> region classification TSV 캐시 생성
# ============================================================
def resolve_manifest_data_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [path]

    if not path.is_absolute():
        candidates.append(manifest_path.parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest 파일을 찾을 수 없습니다: {manifest_path}\n"
            "먼저 step1_prepare_data_final.py를 실행해 corpus_split_manifest.csv를 생성하세요."
        )

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cache_meta_path(tsv_path: Path) -> Path:
    return tsv_path.with_suffix(tsv_path.suffix + ".meta.json")


def cache_matches_request(tsv_path: Path, split: str, max_samples: Optional[int]) -> bool:
    meta_path = cache_meta_path(tsv_path)
    if not tsv_path.exists():
        return False
    if not meta_path.exists():
        return max_samples is None

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta.get("split") == split and meta.get("max_samples") == max_samples


def write_labeled_tsv(
    rows: List[Dict[str, str]],
    manifest_path: Path,
    split: str,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter = Counter()
    file_count = 0
    skipped_files = 0
    written = 0

    split_rows = [row for row in rows if row.get("split") == split]
    print(f"\n--- {split} 분류용 TSV 생성 시작 ---")
    print(f"대상 manifest row 수: {len(split_rows):,}")
    print(f"저장 경로: {output_path}")

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "label", "region"])

        for row in tqdm(split_rows, desc=f"{split} TSV 작성"):
            region = row.get("region", "")
            if region not in LABEL2ID:
                skipped_files += 1
                continue

            source_type = row.get("source_type", "")
            json_path = resolve_manifest_data_path(row.get("path", ""), manifest_path)

            try:
                texts = extract_texts_from_json(json_path, source_type)
            except Exception as e:
                skipped_files += 1
                print(f"\n⚠️ JSON 처리 실패: {json_path} - {e}")
                continue

            file_count += 1
            label_id = LABEL2ID[region]

            for text in texts:
                writer.writerow([text, label_id, region])
                label_counts[region] += 1
                written += 1

                if max_samples is not None and written >= max_samples:
                    break

            if max_samples is not None and written >= max_samples:
                break

    meta = {
        "split": split,
        "path": str(output_path),
        "num_examples": written,
        "num_files": file_count,
        "skipped_files": skipped_files,
        "max_samples": max_samples,
        "label_counts": dict(label_counts),
        "label2id": LABEL2ID,
    }

    with cache_meta_path(output_path).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ {split} TSV 생성 완료: {written:,}문장")
    print(f"지역별 문장 수: {dict(label_counts)}")
    if skipped_files:
        print(f"⚠️ 건너뛴 파일 수: {skipped_files:,}")

    return meta


def prepare_labeled_tsv_files(args: argparse.Namespace) -> Tuple[Path, Path]:
    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    train_tsv = cache_dir / "dialect_region_train.tsv"
    eval_tsv = cache_dir / "dialect_region_eval.tsv"

    rows = load_manifest_rows(manifest_path)

    if args.overwrite_cache or not cache_matches_request(train_tsv, "train", args.max_train_samples):
        write_labeled_tsv(
            rows=rows,
            manifest_path=manifest_path,
            split="train",
            output_path=train_tsv,
            max_samples=args.max_train_samples,
        )
    else:
        print(f"✅ 기존 train TSV 사용: {train_tsv}")

    if args.overwrite_cache or not cache_matches_request(eval_tsv, "eval", args.max_eval_samples):
        write_labeled_tsv(
            rows=rows,
            manifest_path=manifest_path,
            split="eval",
            output_path=eval_tsv,
            max_samples=args.max_eval_samples,
        )
    else:
        print(f"✅ 기존 eval TSV 사용: {eval_tsv}")

    return train_tsv, eval_tsv


# ============================================================
# tokenizer / model 로드
# ============================================================
def load_classifier_tokenizer(args: argparse.Namespace):
    if args.tokenizer_mode == "dialect":
        vocab_path = Path(args.dialect_tokenizer_dir) / "vocab.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"사투리 tokenizer vocab.txt를 찾을 수 없습니다: {vocab_path}\n"
                "먼저 step2_train_tokenizer_final.py를 실행해 dialect_bert_tokenizer/vocab.txt를 생성하세요."
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
        tokenizer.model_max_length = args.max_length
        return tokenizer

    if args.tokenizer_mode == "kcbert":
        return AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    raise ValueError(f"지원하지 않는 tokenizer_mode입니다: {args.tokenizer_mode}")


def reinitialize_input_embeddings(model, pad_token_id: Optional[int]) -> None:
    embeddings = model.get_input_embeddings()
    initializer_range = getattr(model.config, "initializer_range", 0.02)

    with torch.no_grad():
        embeddings.weight.normal_(mean=0.0, std=initializer_range)
        if pad_token_id is not None and 0 <= pad_token_id < embeddings.weight.size(0):
            embeddings.weight[pad_token_id].zero_()


def load_model(args: argparse.Namespace, tokenizer):
    id2label = {idx: label for idx, label in ID2LABEL.items()}
    label2id = {label: idx for label, idx in LABEL2ID.items()}

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(REGION_LABELS),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    old_vocab_size = model.get_input_embeddings().num_embeddings
    new_vocab_size = len(tokenizer)

    if old_vocab_size != new_vocab_size:
        print(f"모델 embedding 크기 조정: {old_vocab_size:,} -> {new_vocab_size:,}")
        model.resize_token_embeddings(new_vocab_size)

    if args.reinit_word_embeddings:
        print(
            "사투리 tokenizer vocab ID와 KcBERT 원래 vocab ID가 다르므로 "
            "word embedding을 새로 초기화합니다."
        )
        reinitialize_input_embeddings(model, tokenizer.pad_token_id)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = id2label
    model.config.label2id = label2id
    return model


# ============================================================
# Hugging Face datasets / metrics
# ============================================================
def require_datasets_package():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "이 스크립트는 대용량 TSV 처리를 위해 datasets 패키지가 필요합니다.\n"
            "서버에서 다음 명령으로 설치하세요:\n"
            "pip install datasets\n"
        ) from e

    return load_dataset


def load_and_tokenize_datasets(args: argparse.Namespace, tokenizer, train_tsv: Path, eval_tsv: Path):
    load_dataset = require_datasets_package()

    raw_datasets = load_dataset(
        "csv",
        data_files={"train": str(train_tsv), "validation": str(eval_tsv)},
        delimiter="\t",
    )

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
        desc="토큰화",
    )

    return tokenized_datasets


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> Tuple[Dict[str, float], Dict[str, object]]:
    num_labels = len(REGION_LABELS)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    for true_label, pred_label in zip(labels, preds):
        if 0 <= int(true_label) < num_labels and 0 <= int(pred_label) < num_labels:
            confusion[int(true_label), int(pred_label)] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0

    per_label = {}
    f1_values = []
    precision_values = []
    recall_values = []
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

    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))
    macro_f1 = float(np.mean(f1_values))
    weighted_f1 = float(weighted_f1_sum / total) if total else 0.0

    scalar_metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    report = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": REGION_LABELS,
        "per_label": per_label,
        "confusion_matrix": confusion.tolist(),
    }

    return scalar_metrics, report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics, _ = classification_metrics(np.asarray(labels), np.asarray(preds))
    return metrics


def save_eval_report(trainer: Trainer, eval_dataset, output_dir: Path) -> None:
    prediction_output = trainer.predict(eval_dataset)
    preds = np.argmax(prediction_output.predictions, axis=-1)
    labels = np.asarray(prediction_output.label_ids)

    metrics, report = classification_metrics(labels, preds)
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

    print("\n=== 최종 Eval 결과 ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"상세 리포트 저장: {output_dir / 'eval_classification_report.json'}")


# ============================================================
# Trainer 설정
# ============================================================
def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    import inspect

    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())

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
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "macro_f1",
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


# ============================================================
# Main
# ============================================================
def train_region_classifier(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.reinit_word_embeddings is None:
        args.reinit_word_embeddings = args.tokenizer_mode == "dialect"

    train_tsv, eval_tsv = prepare_labeled_tsv_files(args)

    if args.prepare_only:
        print("\nprepare_only=True 이므로 TSV 생성 후 종료합니다.")
        return

    tokenizer = load_classifier_tokenizer(args)
    model = load_model(args, tokenizer)

    tokenized_datasets = load_and_tokenize_datasets(args, tokenizer, train_tsv, eval_tsv)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- KcBERT 사투리 지역 분류 fine-tuning 시작 ---")
    print(f"model_name: {args.model_name}")
    print(f"tokenizer_mode: {args.tokenizer_mode}")
    print(f"max_length: {args.max_length}")
    print(f"labels: {LABEL2ID}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    save_eval_report(trainer, tokenized_datasets["validation"], output_dir)
    print(f"\n✅ 학습 완료. 최종 모델 저장: {output_dir / 'final_model'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사투리 문장을 5개 지역(강원도/경상도/전라도/제주도/충청도)으로 분류하도록 KcBERT를 fine-tuning합니다."
    )

    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--stats_json", type=str, default=DEFAULT_STATS_JSON)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer_mode", type=str, choices=["dialect", "kcbert"], default="dialect")
    parser.add_argument("--dialect_tokenizer_dir", type=str, default=DEFAULT_DIALECT_TOKENIZER_DIR)
    parser.add_argument("--reinit_word_embeddings", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None, help="빠른 smoke test용. 전체 학습 시 생략")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="빠른 smoke test용. 전체 평가 시 생략")

    return parser.parse_args()


if __name__ == "__main__":
    train_region_classifier(parse_args())
