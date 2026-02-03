import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def train():
    # 1. 환경 설정
    model_name = "bert-base-multilingual-cased" # 베이스는 구글 모델 사용
    tokenizer_path = "./dialect_bert_tokenizer" # 내가 만든 토크나이저 경로
    data_path = "train_data_sampled.csv" # step4에서 만든 파일
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 현재 사용 중인 장치: {device}")

    # 2. 데이터 로드
    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

    # 4. 모델 설정 (라벨 수: 5개)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.to(device)

    # 5. 학습 인자 설정 (빠른 테스트를 위해 epoch=1)
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      # ✅ evaluation_strategy를 eval_strategy로 변경
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # 6. Trainer 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("--- 학습 시작 ---")
    trainer.train()
    
    # 7. 모델 저장
    model.save_pretrained("./my_dialect_bert_model")
    print("✅ 모델 저장 완료: ./my_dialect_bert_model")

if __name__ == "__main__":
    train()