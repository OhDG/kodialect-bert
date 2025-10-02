# ==============================================================================
# ### 0단계: 환경 설정
# ==============================================================================
import os
import json
import re
import pandas as pd
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# ### 1단계: 사투리 데이터 로딩 및 클리닝
# ==============================================================================
print("===== [1단계] JSON 파일에서 사투리 데이터 로딩 및 클리닝 시작 =====")

region_dirs = {
    "강원도": "JSON만_모은폴더_강원도", "경상도": "JSON만_모은폴더_경상도",
    "전라도": "JSON만_모은폴더_전라도", "제주도": "JSON만_모은폴더_제주도",
    "충청도": "JSON만_모은폴더_충청도"
}
file_dir = "../../project1_dataset"
all_texts = []

for region, subdir in region_dirs.items():
    dir_path = os.path.join(file_dir, subdir)
    if not os.path.exists(dir_path):
        continue
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    utterances = data.get("utterance", [])
                    all_texts.extend([u.get("dialect_form", "") for u in utterances])
            except Exception as e:
                print(f"File Error: {file_path} - {e}")

df_all = pd.DataFrame({"text": all_texts})
df_all.dropna(subset=['text'], inplace=True)
df_all = df_all[df_all['text'].str.strip() != '']
df_all['text'] = df_all['text'].str.replace(r'\([^)]*\)|\[[^)]*\]', '', regex=True)
df_all['text'] = df_all['text'].str.replace(r'[^가-힣a-zA-Z0-9.,?! ]', '', regex=True)
df_all['text'] = df_all['text'].str.strip()
df_all = df_all[df_all['text'] != '']

print(f"최종 훈련 문장 수: {len(df_all):,}")

# ==============================================================================
# ### 2단계: KoBERT 로딩
# ==============================================================================
print("\n===== [2단계] 기존 KoBERT 모델 로딩 =====")

base_model_name = "skt/kobert-base-v1"

# base_model_name = "monologg/kobert"
# label 정보가 없으면 임시로 num_labels=2 지정
model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2, trust_remote_code=True)
original_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

print(f"KoBERT 로딩 완료. 원본 vocab 크기: {len(original_tokenizer):,}")

# ==============================================================================
# ### 3단계: SentencePiece로 사투리 어휘 학습
# ==============================================================================
print("\n===== [3단계] SentencePiece 학습 시작 =====")

# train_file = "dialect_sentences.txt"
# with open(train_file, "w", encoding="utf-8") as f:
  #  for sentence in df_all['text'].tolist():
   #     f.write(sentence + "\n")
        
train_file = "dialect_sentences.txt"
df_all['text'].to_csv(train_file, index=False, header=False)
     

spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix="dialect_spm",
    vocab_size=16000,          # 적당한 크기 (추가 단어 후보 추출용)
    character_coverage=0.9995,
    model_type="bpe",
    input_sentence_size=500000,
    shuffle_input_sentence=True
)

print("SentencePiece 학습 완료 → dialect_spm.model, dialect_spm.vocab")

# ==============================================================================
# ### 4단계: 새로운 토큰 추출
# ==============================================================================
print("\n===== [4단계] 새로운 사투리 토큰 추출 =====")

# 학습된 SentencePiece vocab 읽기
new_vocab = []
with open("dialect_spm.vocab", "r", encoding="utf-8") as f:
    for line in f:
        piece = line.strip().split("\t")[0]
        # <unk>, <s>, </s> 같은 특수토큰 제외
        if not piece.startswith("▁") and len(piece) > 1:
            new_vocab.append(piece)

# KoBERT 기존 vocab과 비교 → 신규 토큰만 추출
original_vocab = set(original_tokenizer.get_vocab().keys())
candidate_tokens = [tok for tok in new_vocab if tok not in original_vocab]

print(f"추출된 후보 토큰 수: {len(candidate_tokens):,}")

# ==============================================================================
# ### 5단계: KoBERT 토크나이저 확장 + 모델 임베딩 리사이즈
# ==============================================================================
print("\n===== [5단계] 토크나이저 확장 및 모델 리사이즈 =====")

# 토큰 추가
num_added = original_tokenizer.add_tokens(candidate_tokens)
print(f"KoBERT 토크나이저에 {num_added:,}개의 새로운 토큰이 추가됨")

# 모델 임베딩 리사이즈
model.resize_token_embeddings(len(original_tokenizer))
print(f"리사이즈된 임베딩 크기: {len(original_tokenizer):,}")

# ==============================================================================
# ### 6단계: 저장 및 검증
# ==============================================================================
print("\n===== [6단계] 저장 및 검증 =====")

save_path = "./KoBERT-Extended-with-Dialect"
os.makedirs(save_path, exist_ok=True)
original_tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"확장된 모델과 토크나이저가 '{save_path}'에 저장됨.")

# 다양한 지역 사투리 문장 샘플
test_sentences = [
    "점심은 잡솼슈? 어데 가노? 마씸 참말이우꽈?",   # 혼합 예시
    "오늘 날씨 진짜 좋다 아이가?",                   # 경상도
    "그거 뭐라카노? 내가 다 했다 아이가!",          # 경상도
    "니 집에 언제 올껴?",                          # 전라도
    "거 참말로 오지게 좋구마잉!",                  # 전라도
    "어데 감둥?",                                 # 강원도
    "점심은 잡솼슈? 마씸 참말이우꽈?",              # 제주도
    "혼저 옵서예, 반갑수다!",                      # 제주도
    "그려, 오늘 날씨가 참말로 좋당께.",             # 충청도
    "어이구, 그라믄 안 되지유~"                    # 충청도
]

print("\n--- KoBERT 토크나이저 검증 ---")
for sentence in test_sentences:
    tokens_before = AutoTokenizer.from_pretrained(base_model_name).tokenize(sentence)
    tokens_after = original_tokenizer.tokenize(sentence)

    print(f"\n입력 문장: {sentence}")
    print(f"[BEFORE] 원본 KoBERT 토큰화: {tokens_before}")
    print(f"[AFTER ] 확장 KoBERT 토큰화: {tokens_after}")
    print(f"UNK BEFORE: {tokens_before.count('[UNK]')}, UNK AFTER: {tokens_after.count('[UNK]')}")