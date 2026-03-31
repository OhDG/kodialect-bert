import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


def print_label_distribution(df, name):
    label_to_region = {
        0: "강원도",
        1: "경상도",
        2: "전라도",
        3: "제주도",
        4: "충청도"
    }

    print(f"\n--- {name} label 분포 ---")
    counts = df["label"].value_counts().sort_index()
    total = len(df)

    for label in range(5):
        count = int(counts.get(label, 0))
        ratio = (count / total * 100) if total > 0 else 0
        print(f"{label_to_region[label]} ({label}): {count:,}개 ({ratio:.2f}%)")
    print(f"총합: {total:,}개")


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def train_google_bert():
    model_name = "bert-base-multilingual-cased"
    tokenizer_name = "bert-base-multilingual-cased"

    split_dir = "./dialect_bert_full_results_fast/data_splits"
    train_csv_path = os.path.join(split_dir, "train_90_inner.csv")
    val_csv_path = os.path.join(split_dir, "val_from_train.csv")
    holdout_csv_path = os.path.join(split_dir, "holdout_10.csv")

    output_dir = "./google_bert_full_results_fast"
    final_model_dir = "./my_google_bert_final_fast"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작 장치: {device}")

    print("\n--- 기존 분할 파일 로드 중 ---")
    train_df = pd.read_csv(train_csv_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_csv_path, encoding="utf-8-sig")
    holdout_df = pd.read_csv(holdout_csv_path, encoding="utf-8-sig")

    for df in [train_df, val_df, holdout_df]:
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

    print_label_distribution(train_df, "train")
    print_label_distribution(val_df, "validation")
    print_label_distribution(holdout_df, "holdout")

    train_counts = train_df["label"].value_counts().sort_index()
    num_labels = 5
    total_train = len(train_df)

    class_weights = []
    for label in range(num_labels):
        count = int(train_counts.get(label, 0))
        if count == 0:
            raise ValueError(f"label {label}의 학습 데이터가 0개입니다.")
        weight = total_train / (num_labels * count)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"\n클래스 가중치: {class_weights.tolist()}")

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=96
        )

    print("\n--- Google BERT 토크나이징 중 ---")
    train_ds = train_ds.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=["text"]
    )
    val_ds = val_ds.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=["text"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        learning_rate=3e-5,

        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,

        num_train_epochs=2,
        weight_decay=0.01,

        logging_steps=2000,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=8,
        group_by_length=True,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        seed=seed,
        data_seed=seed,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    print("\n--- Google BERT tokenizer 조건으로 전체 학습 시작 ---")
    trainer.train()

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    meta = {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "train_csv": train_csv_path,
        "val_csv": val_csv_path,
        "holdout_csv": holdout_csv_path,
        "max_length": 96,
        "epochs": 2,
        "lr": 3e-5,
        "train_batch_size": 128,
        "eval_batch_size": 256
    }

    with open(os.path.join(final_model_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 최종 모델 저장 완료: {final_model_dir}")
    print(f"✅ 평가용 홀드아웃 파일: {holdout_csv_path}")


if __name__ == "__main__":
    train_google_bert()