# ==============================================================================
# ### 0단계: 환경 설정
# ==============================================================================
import os
import json
import re
import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

print(f"PyTorch CUDA 사용 가능 여부: {torch.cuda.is_available()}")

# ==============================================================================
# ### 1단계: 데이터 로딩 (Fine-tuning용)
# ==============================================================================
print("===== [1단계] Fine-tuning용 사투리 데이터 및 라벨 로딩 시작 =====")

# 지역별 폴더명과 라벨 정의
region_dirs = {
    "강원도": "JSON만_모은폴더_강원도",
    "경상도": "JSON만_모은폴더_경상도",
    "전라도": "JSON만_모은폴더_전라도",
    "제주도": "JSON만_모은폴더_제주도",
    "충청도": "JSON만_모은폴더_충청도"
}
region_label = {"강원도": 0, "경상도": 1, "전라도": 2, "제주도": 3, "충청도": 4}
file_dir = "../../project1_dataset"

all_texts = []
all_labels = []

# 각 지역 디렉토리를 순회하면서 모든 JSON 파일 읽기
for region, subdir in region_dirs.items():
    dir_path = os.path.join(file_dir, subdir)
    if not os.path.exists(dir_path):
        continue
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    utterances = data.get("utterance", [])
                    # 데이터 클리닝 추가
                    for u in utterances:
                        text = u.get("dialect_form", "")
                        if isinstance(text, str) and text.strip():
                            # 정규 표현식 클리닝
                            cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                            cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                            if cleaned_text:
                                all_texts.append(cleaned_text)
                                all_labels.append(region_label[region])
            except Exception as e:
                print(f"⚠️ 파일 오류 발생: {file_path} - {e}")

            # ✅ 첫 번째 파일만 처리하고 나머지는 무시
            break

# 데이터프레임 생성 및 train/test 분리
df_all = pd.DataFrame({"text": all_texts, "label": all_labels})
train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

print(f"데이터 로딩 완료: Train {len(train_dataset):,}개, Test {len(test_dataset):,}개")

# ==============================================================================
# ### 2단계: 커스텀 토크나이저 로딩 및 모델 준비 (핵심 수정 부분)
# ==============================================================================
print("\n===== [2단계] 커스텀 토크나이저 로딩 및 모델 리사이즈 시작 =====")

# ✅ 1. 기본 KoBERT 토크나이저를 먼저 로드합니다.
# tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=True)
print(f"기존 KoBERT 토크나이저 어휘 사전 크기: {len(tokenizer)}")

# ✅ 2. SentencePiece로 만든 .vocab 파일에서 새로운 단어들을 읽어옵니다.
#    (이전에 생성한 'dialect_spm.vocab' 파일이 이 스크립트와 같은 위치에 있어야 합니다.)
new_tokens = []
vocab_file_path = "dialect_spm.vocab" # SentencePiece로 만든 vocab 파일
with open(vocab_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        token = line.strip().split('\t')[0]
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

# ✅ 3. 기존 토크나이저에 새로운 단어들을 추가합니다.
tokenizer.add_tokens(new_tokens)
print(f"새로운 사투리 토큰 {len(new_tokens)}개 추가 완료!")
print(f"확장된 토크나이저 어휘 사전 크기: {len(tokenizer)}")


# ✅ 4. KoBERT 모델을 로드합니다.
# model = AutoModelForSequenceClassification.from_pretrained(
#     "monologg/kobert",
#     num_labels=5
# )
model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=5)


# ✅ 5. 모델의 임베딩 레이어 크기를 확장된 토크나이저 크기에 맞게 조정합니다. (매우 중요!)
model.resize_token_embeddings(len(tokenizer))
print("모델의 Token Embedding 레이어 리사이즈 완료!")


# ==============================================================================
# ### 3단계: 데이터 토큰화
# ==============================================================================
print("\n===== [3단계] 데이터셋 토큰화 시작 =====")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128) # max_length 조정 가능

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ==============================================================================
# ### 4단계: 가중치 적용 Trainer 및 학습 설정
# ==============================================================================
print("\n===== [4단계] Trainer 설정 시작 =====")

# 클래스 가중치 계산을 위한 Trainer 정의
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 가중치 계산
train_labels = np.array(train_dataset['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# 평가 지표 계산 함수
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results_custom_tokenizer",
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    warmup_steps=3000,
    weight_decay=0.01,
    logging_dir="./logs_custom_tokenizer",
    logging_steps=10000,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_total_limit=2,
    save_steps=10000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="tensorboard",
)

# Trainer 정의
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    class_weights=class_weights_tensor
)

# ==============================================================================
# ### 5단계: 모델 학습 및 저장
# ==============================================================================
print("\n===== [5단계] 모델 학습 시작 =====")
trainer.train()

# 가장 성능이 좋았던 모델 저장
model_save_path = "./my_best_dialect_model_custom_tokenizer"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path) # 토크나이저도 함께 저장해야 나중에 불러오기 편합니다.

print(f"최적 모델과 토크나이저를 '{model_save_path}'에 저장했습니다.")

# ==============================================================================
# ### 6단계: 최종 평가 및 예측 테스트
# ==============================================================================
print("\n===== [6단계] 최종 평가 및 테스트 시작 =====")

# 테스트 데이터셋으로 최종 평가
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
accuracy = accuracy_score(true_labels, pred_labels)
print(f"✅ 최종 테스트 정확도: {accuracy * 100:.2f}%")


# 예측 테스트
# ------------------------------------------------------------------------------
# 저장된 모델과 토크나이저를 다시 불러와서 테스트 (실제 서빙 환경과 유사)
print("\n--- 저장된 모델로 예측 테스트 ---")
model_path = "./my_best_dialect_model_custom_tokenizer"
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_model.eval()

sentence = "거시기 저짝 가보랑께"
id2region = {0: "강원도", 1: "경상도", 2: "전라도", 3: "제주도", 4: "충청도"}

inputs = loaded_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()

print(f"🗣 입력 문장: {sentence}")
print(f"📍 예측 지역: {id2region[pred_label]} (확률: {probs[0][pred_label].item():.2%})")