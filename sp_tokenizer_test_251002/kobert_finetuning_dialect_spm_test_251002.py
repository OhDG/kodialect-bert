# ==============================================================================
# ### 0단계: 환경 설정
# ==============================================================================
import os
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,      # ✅ '느린' 토크나이저를 임포트합니다.
    PreTrainedTokenizerFast,  # ✅ '빠른' 토크나이저로 변환하기 위해 임포트합니다.
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
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
# (데이터 로딩 부분은 이전과 동일하며, 생략합니다. 필요시 이전 코드 내용을 여기에 붙여넣으세요.)
print("===== [1단계] Fine-tuning용 사투리 데이터 및 라벨 로딩 시작 =====")

# 지역별 폴더명과 라벨 정의
region_dirs = {
    "강원도": "JSON만_모은폴더_강원도", "경상도": "JSON만_모은폴더_경상도",
    "전라도": "JSON만_모은폴더_전라도", "제주도": "JSON만_모은폴더_제주도",
    "충청도": "JSON만_모은폴더_충청도"
}
region_label = {"강원도": 0, "경상도": 1, "전라도": 2, "제주도": 3, "충청도": 4}
file_dir = "../../project1_dataset"

all_texts, all_labels = [], []

for region, subdir in region_dirs.items():
    dir_path = os.path.join(file_dir, subdir)
    if not os.path.exists(dir_path): continue
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    utterances = data.get("utterance", [])
                    for u in utterances:
                        text = u.get("dialect_form", "")
                        if isinstance(text, str) and text.strip():
                            cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                            cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                            if cleaned_text:
                                all_texts.append(cleaned_text)
                                all_labels.append(region_label[region])
            except Exception as e:
                print(f"⚠️ 파일 오류 발생: {file_path} - {e}")

df_all = pd.DataFrame({"text": all_texts, "label": all_labels})
train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
print(f"데이터 로딩 완료: Train {len(train_dataset):,}개, Test {len(test_dataset):,}개")


# # ==============================================================================
# # ### 2단계: 커스텀 토크나이저 로딩 및 모델 준비 (핵심 수정 부분)
# # ==============================================================================
# print("\n===== [2단계] 커스텀 토크나이저 로딩 및 모델 리사이즈 시작 =====")

# # ✅ 1. SentencePiece로 만든 .model 파일을 직접 로드하여 토크나이저를 생성합니다.
# #    (이전에 생성한 'dialect_spm.model' 파일이 이 스크립트와 같은 위치에 있어야 합니다.)
# tokenizer_file = "dialect_spm.model"
# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file=tokenizer_file,
#     pad_token="<pad>", # PAD 토큰 명시
#     unk_token="<unk>", # UNK 토큰 명시
#     bos_token="<s>",   # BOS 토큰 명시
#     eos_token="</s>",   # EOS 토큰 명시
# )
# print(f"커스텀 토크나이저 로딩 완료. 어휘 사전 크기: {len(tokenizer)}")


# # ✅ 2. KoBERT 모델을 로드합니다.
# model = AutoModelForSequenceClassification.from_pretrained(
#     "monologg/kobert",
#     num_labels=5
# )
# print(f"기존 KoBERT 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")



# # ✅ 3. (매우 중요!) 모델의 임베딩 레이어 크기를 새로운 토크나이저 크기에 맞게 조정합니다.
# #    이 과정은 기존 임베딩 지식을 무효화하고, 새로운 크기에 맞춰 레이어를 사실상 재초기화합니다.
# model.resize_token_embeddings(len(tokenizer))
# print(f"리사이즈된 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")


# ==============================================================================
# ### 2단계: 커스텀 토크나이저 로딩 및 모델 준비 (최종 수정)
# ==============================================================================
print("\n===== [2단계] 커스텀 토크나이저 로딩 및 모델 리사이즈 시작 =====")

# ✅ 1. '느린' 토크나이저로 .model과 .vocab 파일을 로드합니다.
#    XLMRobertaTokenizer는 sentencepiece 모델을 다룰 수 있는 대표적인 '느린' 토크나이저입니다.
# slow_tokenizer = XLMRobertaTokenizer(vocab_file="dialect_spm.vocab", sp_model_file="dialect_spm.model")


slow_tokenizer = XLMRobertaTokenizer(vocab_file="dialect_spm.model")

# ✅ 2. (중요) BERT 계열 모델에서 사용하는 특수 토큰들을 추가해줍니다.
special_tokens_to_add = ['[CLS]', '[SEP]', '[MASK]']
slow_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

# ✅ 3. '느린' 토크나이저를 '빠른' 토크나이저로 변환합니다.
#    이제 이 tokenizer 객체는 다른 빠른 토크나이저처럼 동작합니다.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=slow_tokenizer,
    pad_token="<pad>",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
print(f"커스텀 토크나이저 로딩 및 변환 완료. 어휘 사전 크기: {len(tokenizer)}")

# ✅ 4. KoBERT 모델을 로드합니다.
model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/kobert",
    num_labels=5
)
print(f"기존 KoBERT 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")

# ✅ 5. (매우 중요!) 모델의 임베딩 레이어 크기를 새로운 토크나이저 크기에 맞게 조정합니다.
model.resize_token_embeddings(len(tokenizer))
print(f"리사이즈된 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")


# ==============================================================================
# ### 3단계: 데이터 토큰화
# ==============================================================================
# (이하 코드는 이전과 동일)
print("\n===== [3단계] 데이터셋 토큰화 시작 =====")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ==============================================================================
# ### 4단계: 가중치 적용 Trainer 및 학습 설정
# ==============================================================================
print("\n===== [4단계] Trainer 설정 시작 =====")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels"); outputs = model(**inputs); logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device)); loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

train_labels = np.array(train_dataset['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

def compute_metrics(p):
    pred, labels = p; pred = np.argmax(pred, axis=1); accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results_replace_tokenizer", num_train_epochs=20, per_device_train_batch_size=64,
    per_device_eval_batch_size=128, learning_rate=5e-5, warmup_steps=3000, weight_decay=0.01,
    logging_dir="./logs_replace_tokenizer", logging_steps=10000, evaluation_strategy="steps",
    eval_steps=10000, save_total_limit=2, save_steps=10000, load_best_model_at_end=True,
    metric_for_best_model="accuracy", greater_is_better=True, report_to="tensorboard",
)

trainer = WeightedTrainer(
    model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test,
    compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    class_weights=class_weights_tensor
)

# ==============================================================================
# ### 5단계: 모델 학습 및 저장
# ==============================================================================
print("\n===== [5단계] 모델 학습 시작 =====")
trainer.train()

model_save_path = "./my_best_dialect_model_replace_tokenizer"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"최적 모델과 토크나이저를 '{model_save_path}'에 저장했습니다.")

# ==============================================================================
# ### 6단계: 최종 평가 및 예측 테스트
# ==============================================================================
print("\n===== [6단계] 최종 평가 및 테스트 시작 =====")
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
accuracy = accuracy_score(true_labels, pred_labels)
print(f"✅ 최종 테스트 정확도: {accuracy * 100:.2f}%")

print("\n--- 저장된 모델로 예측 테스트 ---")
model_path = "./my_best_dialect_model_replace_tokenizer"
loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_model.eval()

sentence = "거시기 저짝 가보랑께"; id2region = {0: "강원도", 1: "경상도", 2: "전라도", 3: "제주도", 4: "충청도"}
inputs = loaded_tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = loaded_model(**inputs); logits = outputs.logits; probs = F.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
print(f"🗣 입력 문장: {sentence}"); print(f"📍 예측 지역: {id2region[pred_label]} (확률: {probs[0][pred_label].item():.2%})")