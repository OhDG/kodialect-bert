import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def train_full():
    # 1. 환경 설정
    model_name = "bert-base-multilingual-cased"
    tokenizer_path = "./dialect_bert_tokenizer"
    data_path = "train_data_full.csv" # 생성한 전체 데이터 파일
    output_dir = "./dialect_bert_full_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작 장치: {device}")

    # 2. 대용량 데이터 로드 (스트리밍 방식은 아니지만 메모리 맵핑을 지원함)
    # csv 파일을 datasets 라이브러리로 로드하면 메모리 효율이 매우 좋습니다.
    dataset = load_dataset("csv", data_files=data_path)["train"]
    
    # 3. 데이터셋 분할 (99% 학습 / 1% 검증 - 670만 개라 1%만 해도 충분)
    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]

    # 4. 토크나이저 준비
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # 5. 토크나이징 (num_proc을 늘려 멀티 프로세싱으로 속도 향상)
    print("--- 데이터 토크나이징 중 (멀티코어 사용) ---")
    train_ds = train_ds.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])

    # 6. 모델 설정
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.to(device)

    # 7. 전체 학습용 인자 설정
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     eval_strategy="steps",        # 대용량일 땐 epoch가 너무 기니 steps 단위로 평가
    #     eval_steps=5000,               # 5000 스텝마다 평가
    #     save_strategy="steps",
    #     save_steps=5000,               # 5000 스텝마다 체크포인트 저장 (중요!)
    #     save_total_limit=3,            # 체크포인트는 최근 3개만 유지 (용량 확보)
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=32, # GPU 메모리 봐서 16~64 사이 조절
    #     gradient_accumulation_steps=2,  # 배치 사이즈를 키우는 효과 (메모리 절약)
    #     num_train_epochs=3,             # 보통 3~5 epoch 권장
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=100,
    #     fp16=True,                      # GPU가 지원한다면 Mixed Precision으로 속도 2배 향상
    #     load_best_model_at_end=True,    # 가장 성능 좋은 체크포인트를 최종 모델로 선택
    # )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=10000,              # 670만 개이므로 평가를 너무 자주 하면 느려짐
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=5,
        learning_rate=2e-5,
        
        # --- A6000 맞춤 설정 ---
        per_device_train_batch_size=128, # 48GB VRAM이므로 64~128까지 과감하게 키워보세요.
        gradient_accumulation_steps=1,   # 배치가 충분히 크다면 1로 설정해 속도 극대화
        # -----------------------

        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        fp16=True,                       # A6000은 Tensor Core 성능이 좋아 fp16 필수입니다.
        dataloader_num_workers=8,        # 데이터 로딩 속도를 높이기 위해 CPU 코어 활용
        group_by_length=True,            # 문장 길이가 비슷한 것끼리 묶어 패딩 최소화 (속도 향상)
    )


    # 8. Trainer 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("--- 670만 개 데이터 전체 학습 시작 ---")
    trainer.train()
    
    # 9. 최종 모델 저장
    model.save_pretrained("./my_dialect_bert_final")
    tokenizer.save_pretrained("./my_dialect_bert_final")
    print("✅ 최종 모델 저장 완료: ./my_dialect_bert_final")

if __name__ == "__main__":
    train_full()