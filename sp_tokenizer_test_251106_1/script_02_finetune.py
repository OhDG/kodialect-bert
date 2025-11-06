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

print(f"PyTorch CUDA 사용 가능 여부: {torch.cuda.is_available()}")

# ==============================================================================
# ### 1단계: 데이터 로딩 (Fine-tuning용)
# ==============================================================================
print("===== [1단계] Fine-tuning용 사투리 데이터 및 라벨 로딩 시작 =====")
region_dirs = {
    "강원도": "JSON만_모은폴더_강원도",
    "경상도": "JSON만_모은폴더_경상도",
    "전라도": "JSON만_모은폴더_전라도",
    "제주도": "JSON만_모은폴더_제주도",
    "충청도": "JSON만_모은폴더_충청도"
}
region_label = {"강원도": 0, "경상도": 1, "전라도": 2, "제주도": 3, "충청도": 4}
file_dir = "../../project1_dataset"
all_texts, all_labels = [], []

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
            
            # ✅✅✅ L35: 빠른 테스트를 위해 1개 파일만 로드
            print(f"    -> {region} 1개 파일 로딩 완료.")
            break

df_all = pd.DataFrame({"text": all_texts, "label": all_labels})
# (데이터가 너무 적으면 stratify가 실패할 수 있으므로, 테스트 데이터가 0개여도 진행)
if len(df_all) > 10:
    train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)
else:
    train_df = df_all
    test_df = df_all.copy() # 그냥 복사
    
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
print(f"데이터 로딩 완료: Train {len(train_dataset):,}개, Test {len(test_dataset):,}개")

# ==============================================================================
# ### 2단계: 커스텀 "확장" 모델 및 토크나이저 로딩
# ==============================================================================
print("\n===== [2단계] Script 1에서 저장한 확장 모델/토크나이저 로딩 =====")
extended_model_path = "./KoBERT-Extended-with-Dialect" 
tokenizer = AutoTokenizer.from_pretrained(extended_model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    extended_model_path,
    num_labels=5,
    ignore_mismatched_sizes=True
)
print(f"확장된 모델 로딩 완료.")
print(f"확장된 토크나이저 어휘 사전 크기: {len(tokenizer)}")
print(f"모델의 분류 Head 크기: {model.num_labels}")

# ==============================================================================
# ### 3단계: 데이터 토큰화 (및 검증)
# ==============================================================================
print("\n===== [3단계] 데이터셋 토큰화 시작 =====")
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

print("\n===== [3.5단계] 토큰 ID 및 임베딩 크기 검증 =====")
emb_size = model.get_input_embeddings().num_embeddings
print(f"모델의 실제 임베딩 크기 (num_embeddings): {emb_size}")
print(f"토크나이저 어휘 사전 크기 (len): {len(tokenizer)}")

if emb_size != len(tokenizer):
    print(f"‼️‼️‼️ 오류: 모델 임베딩 크기({emb_size})와 토크나이저 크기({len(tokenizer)})가 일치하지 않습니다!")
    print("Script 1을 다시 실행하세요.")
    exit()

def max_id(ds):
    try: return max(max(row) for row in ds['input_ids'])
    except ValueError: return 0 

max_train_id = max_id(tokenized_train)
max_test_id = max_id(tokenized_test)
print(f"Max token ID in Train data: {max_train_id}")
print(f"Max token ID in Test data:  {max_test_id}")

if max_train_id >= emb_size or max_test_id >= emb_size:
    print(f"💥💥💥 오류: 데이터의 토큰 ID ({max(max_train_id, max_test_id)})가 모델 임베딩 크기 ({emb_size})보다 큽니다.")
    print("이것이 CUDA assert 오류의 원인입니다. Script 1을 다시 실행하세요.")
    exit()
else:
    print("✅ 검증 통과: 모든 토큰 ID가 모델 임베딩 크기보다 작습니다.")

# ==============================================================================
# ### 4단계: 가중치 적용 Trainer 및 학습 설정
# ==============================================================================
print("\n===== [4단계] Trainer 설정 시작 =====")
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
train_labels = np.array(train_dataset['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    return {"accuracy": accuracy_score(y_true=labels, y_pred=pred)}

# ✅✅✅ L161: 작은 데이터셋에 맞게 파라미터 조정
training_args = TrainingArguments(
    output_dir="./results_custom_tokenizer", 
    num_train_epochs=5, # ✅ Epochs 20 -> 5 (빠른 테스트)
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=128,
    learning_rate=5e-5, 
    warmup_steps=50, # ✅ 300 -> 50
    weight_decay=0.01,
    logging_dir="./logs_custom_tokenizer", 
    logging_steps=25, # ✅ 100 -> 25 (약 1 epoch마다)
    evaluation_strategy="steps", 
    eval_steps=25, # ✅ 100 -> 25 (약 1 epoch마다)
    save_total_limit=2,
    save_steps=25, # ✅ 100 -> 25
    load_best_model_at_end=True,
    metric_for_best_model="accuracy", 
    greater_is_better=True,
    report_to="tensorboard",
)
trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_train, eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    class_weights=class_weights_tensor
)

# ==============================================================================
# ### 5단계: 모델 학습 및 저장
# ==============================================================================
print("\n===== [5단계] 모델 학습 시작 =====")
trainer.train()

# ==============================================================================
# ### 6단계: 최종 평가 및 예측 테스트
# ==============================================================================
print("\n===== [6단계] 최종 평가 및 테스트 시작 =====")
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
accuracy = accuracy_score(predictions.label_ids, pred_labels)
print(f"✅ 최종 테스트 정확도: {accuracy * 100:.2f}%")

print("\n--- 저장된 모델로 예측 테스트 ---")
model_save_path = "./my_best_dialect_model_custom_tokenizer"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
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