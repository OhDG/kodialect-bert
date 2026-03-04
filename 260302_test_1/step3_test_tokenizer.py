import os
import random
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

def get_tokenizer_stats(tokenizer, sentences, name):
    """
    대규모 문장 리스트에 대해 토크나이저 통계를 계산합니다.
    """
    total_words = 0
    total_tokens = 0
    total_subwords = 0 # ##이 붙은 토큰 수
    total_unks = 0     # [UNK] 토큰 수
    seq_lengths = []   # 문장당 토큰 수 리스트

    print(f"\n[{name}] 통계 분석 중...")
    
    for sentence in tqdm(sentences):
        words = sentence.split()
        if not words: continue
        
        tokens = tokenizer.tokenize(sentence)
        
        total_words += len(words)
        total_tokens += len(tokens)
        total_subwords += sum(1 for t in tokens if t.startswith("##"))
        total_unks += sum(1 for t in tokens if t == "[UNK]")
        seq_lengths.append(len(tokens))

    fertility = total_tokens / total_words if total_words > 0 else 0
    subword_rate = (total_subwords / total_tokens) * 100 if total_tokens > 0 else 0
    unk_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0
    avg_seq_len = np.mean(seq_lengths)

    return {
        "Fertility": fertility,
        "Subword Rate (%)": subword_rate,
        "UNK Rate (%)": unk_rate,
        "Avg Seq Len": avg_seq_len
    }

def run_statistical_comparison():
    # 1. 토크나이저 로드
    my_tokenizer_path = "./dialect_bert_tokenizer"
    try:
        my_tokenizer = BertTokenizer.from_pretrained(my_tokenizer_path, do_lower_case=False)
    except:
        print("⚠️ 내 토크나이저 경로를 확인하세요.")
        return

    google_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    klue_tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

    # 2. 대규모 데이터셋 로드 및 샘플링
    corpus_file = "dialect_corpus.txt"
    if not os.path.exists(corpus_file):
        print(f"⚠️ {corpus_file} 파일이 없습니다. 먼저 생성하세요.")
        return

    print("데이터 로딩 중...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        all_sentences = f.readlines()
    
    # 전체를 다 돌리기엔 시간이 걸리므로 10만 문장 샘플링 (논문용으로 충분)
    sample_size = min(100000, len(all_sentences))
    test_sentences = random.sample(all_sentences, sample_size)
    print(f"✅ {sample_size:,}개의 문장을 샘플링하여 분석을 시작합니다.")

    # 3. 통계 계산
    tokenizers = {
        "My Dialect Tokenizer": my_tokenizer,
        "Google Multilingual": google_tokenizer,
        "KLUE BERT (Standard)": klue_tokenizer
    }

    results = {}
    for name, tokenizer in tokenizers.items():
        results[name] = get_tokenizer_stats(tokenizer, test_sentences, name)

    # 4. 결과 출력 (논문용 Table 꼴)
    print("\n" + "="*85)
    print(f"{'Tokenizer':<25} | {'Fertility':<10} | {'Subword %':<10} | {'UNK %':<10} | {'AvgLen':<10}")
    print("-" * 85)
    for name, res in results.items():
        print(f"{name:<25} | {res['Fertility']:<10.4f} | {res['Subword Rate (%)']:<10.2f} | {res['UNK Rate (%)']:<10.4f} | {res['Avg Seq Len']:<10.2f}")
    print("="*85)
    
    print("\n💡 해석 가이드:")
    print("1. Fertility가 낮을수록: 사투리 어휘를 '하나의 의미 단위'로 잘 파악하고 있음.")
    print("2. UNK Rate가 낮을수록: 기존 모델이 모르는 사투리 단어를 내 모델은 알고 있음.")
    print("3. Avg Seq Len이 짧을수록: 학습 및 추론 속도가 더 빠르고 효율적임.")

if __name__ == "__main__":
    run_statistical_comparison()