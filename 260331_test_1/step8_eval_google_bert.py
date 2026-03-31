import torch
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset


class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model():
    # Google BERT tokenizer로 학습한 최종 모델 경로
    model_path = "./my_google_bert_final_fast"
    holdout_data_path = "./dialect_bert_full_results_fast/data_splits/holdout_10.csv"

    label_names = ["강원", "경상", "전라", "제주", "충청"]
    label_to_region = {
        0: "강원",
        1: "경상",
        2: "전라",
        3: "제주",
        4: "충청"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 장치: {device}")

    # Google tokenizer로 학습된 모델 폴더에서 tokenizer/model 불러오기
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    # 홀드아웃 데이터 로드
    df = pd.read_csv(holdout_data_path, encoding="utf-8-sig")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    test_texts = df["text"].tolist()
    test_labels = df["label"].tolist()

    test_dataset = DialectDataset(test_texts, test_labels, tokenizer, max_length=96)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    predictions = []
    real_labels = []

    print("\n--- Google tokenizer 모델 홀드아웃 평가 중 ---")
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            real_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    real_labels = np.array(real_labels)

    # 전체 성능
    acc = accuracy_score(real_labels, predictions)
    macro_f1 = f1_score(real_labels, predictions, average="macro")
    weighted_f1 = f1_score(real_labels, predictions, average="weighted")

    print("\n" + "=" * 60)
    print(f"✅ Google tokenizer 기준 전체 Accuracy   : {acc * 100:.2f}%")
    print(f"✅ Google tokenizer 기준 전체 Macro F1   : {macro_f1 * 100:.2f}%")
    print(f"✅ Google tokenizer 기준 전체 Weighted F1: {weighted_f1 * 100:.2f}%")
    print("=" * 60)

    # 지역별 accuracy
    print("\n[지역별 Accuracy]")
    for label_id, region_name in label_to_region.items():
        mask = (real_labels == label_id)
        region_total = int(mask.sum())

        if region_total == 0:
            print(f"{region_name}: 데이터 없음")
            continue

        region_correct = int((predictions[mask] == real_labels[mask]).sum())
        region_acc = region_correct / region_total

        print(
            f"{region_name:<4} | "
            f"정확도: {region_acc * 100:6.2f}% | "
            f"정답 수: {region_correct:>6,} | "
            f"전체 수: {region_total:>6,}"
        )

    print("\n[상세 Classification Report]")
    print(
        classification_report(
            real_labels,
            predictions,
            target_names=label_names,
            digits=4
        )
    )


if __name__ == "__main__":
    evaluate_model()