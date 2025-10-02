# ==============================================================================
# ### 0단계: 환경 설정
# ==============================================================================
import os
import json
import re
import pandas as pd
import sentencepiece as spm

# ==============================================================================
# ### 1단계: 데이터 로딩 및 클리닝
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
# ### 2단계: 텍스트 파일로 저장 (SentencePiece 학습용)
# ==============================================================================
print("\n===== [2단계] 텍스트 파일 생성 =====")

# train_file = "dialect_sentences.txt"
# with open(train_file, "w", encoding="utf-8") as f:
#     for sentence in df_all['text'].tolist():
#         f.write(sentence + "\n")

# print(f"텍스트 파일 '{train_file}' 저장 완료 (총 {len(df_all):,} 문장)")

train_file = "dialect_sentences.txt"
df_all['text'].to_csv(train_file, index=False, header=False)


# ==============================================================================
# ### 3단계: SentencePiece 모델 학습
# ==============================================================================
print("\n===== [3단계] SentencePiece 모델 학습 시작 =====")

model_prefix = "dialect_spm"
vocab_size = 32000
character_coverage = 0.9995   # 한국어 + 일부 특수문자 거의 모두 커버
model_type = "bpe"            # bpe / unigram / char / word 가능

spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    model_type=model_type,
    input_sentence_size=1000000,     # 최대 100만 문장 샘플링
    shuffle_input_sentence=True,     # 샘플링 시 랜덤 셔플
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print(f"모델 학습 완료: {model_prefix}.model, {model_prefix}.vocab 생성됨")

# ==============================================================================
# ### 4단계: 모델 로드 및 검증 (여러 문장)
# ==============================================================================
print("\n===== [4단계] SentencePiece 모델 검증 =====")

sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

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

for sentence in test_sentences:
    tokens = sp.encode(sentence, out_type=str)
    decoded = sp.decode(sp.encode(sentence, out_type=int))
    
    print(f"\n입력 문장: {sentence}")
    print(f"토큰화 결과: {tokens}")
    print(f"디코딩 결과: {decoded}")