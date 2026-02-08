import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_model():
    model_path = "./my_dialect_bert_final"
    tokenizer_path = "./dialect_bert_tokenizer"
    val_data_path = "train_data_full.csv" # 테스트용으로 동일 파일 사용
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 데이터 준비
    df = pd.read_csv(val_data_path)
    # 실제 프로젝트에선 학습에 안 쓴 별도 test_data를 사용하세요.
    test_texts = df['text'].tolist()
    test_labels = df['label'].tolist()

    test_dataset = DialectDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    predictions = []
    real_labels = []

    print("--- 정확도 측정 중 ---")
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            real_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(real_labels, predictions)
    print(f"\n✅ 최종 정확도(Accuracy): {acc * 100:.2f}%")
    print("\n[상세 리포트]")
    print(classification_report(real_labels, predictions, target_names=["강원", "경상", "전라", "제주", "충청"]))

if __name__ == "__main__":
    evaluate_model()