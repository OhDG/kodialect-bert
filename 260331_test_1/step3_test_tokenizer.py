# import os
# import random
# from transformers import BertTokenizer
# from tqdm import tqdm
# import numpy as np

# def get_tokenizer_stats(tokenizer, sentences, name):
#     """
#     대규모 문장 리스트에 대해 토크나이저 통계를 계산합니다.
#     """
#     total_words = 0
#     total_tokens = 0
#     total_subwords = 0 # ##이 붙은 토큰 수
#     total_unks = 0     # [UNK] 토큰 수
#     seq_lengths = []   # 문장당 토큰 수 리스트

#     print(f"\n[{name}] 통계 분석 중...")
    
#     for sentence in tqdm(sentences):
#         words = sentence.split()
#         if not words: continue
        
#         tokens = tokenizer.tokenize(sentence)
        
#         total_words += len(words)
#         total_tokens += len(tokens)
#         total_subwords += sum(1 for t in tokens if t.startswith("##"))
#         total_unks += sum(1 for t in tokens if t == "[UNK]")
#         seq_lengths.append(len(tokens))

#     fertility = total_tokens / total_words if total_words > 0 else 0
#     subword_rate = (total_subwords / total_tokens) * 100 if total_tokens > 0 else 0
#     unk_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0
#     avg_seq_len = np.mean(seq_lengths)

#     return {
#         "Fertility": fertility,
#         "Subword Rate (%)": subword_rate,
#         "UNK Rate (%)": unk_rate,
#         "Avg Seq Len": avg_seq_len
#     }

# def run_statistical_comparison():
#     # 1. 토크나이저 로드
#     my_tokenizer_path = "./dialect_bert_tokenizer"
#     try:
#         my_tokenizer = BertTokenizer.from_pretrained(my_tokenizer_path, do_lower_case=False)
#     except:
#         print("⚠️ 내 토크나이저 경로를 확인하세요.")
#         return

#     google_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#     klue_tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

#     # 2. 대규모 데이터셋 로드 및 샘플링
#     corpus_file = "dialect_corpus.txt"
#     if not os.path.exists(corpus_file):
#         print(f"⚠️ {corpus_file} 파일이 없습니다. 먼저 생성하세요.")
#         return

#     print("데이터 로딩 중...")
#     with open(corpus_file, "r", encoding="utf-8") as f:
#         all_sentences = f.readlines()
    
#     # 전체를 다 돌리기엔 시간이 걸리므로 10만 문장 샘플링 (논문용으로 충분)
#     sample_size = min(100000, len(all_sentences))
#     test_sentences = random.sample(all_sentences, sample_size)
#     print(f"✅ {sample_size:,}개의 문장을 샘플링하여 분석을 시작합니다.")

#     # 3. 통계 계산
#     tokenizers = {
#         "My Dialect Tokenizer": my_tokenizer,
#         "Google Multilingual": google_tokenizer,
#         "KLUE BERT (Standard)": klue_tokenizer
#     }

#     results = {}
#     for name, tokenizer in tokenizers.items():
#         results[name] = get_tokenizer_stats(tokenizer, test_sentences, name)

#     # 4. 결과 출력 (논문용 Table 꼴)
#     print("\n" + "="*85)
#     print(f"{'Tokenizer':<25} | {'Fertility':<10} | {'Subword %':<10} | {'UNK %':<10} | {'AvgLen':<10}")
#     print("-" * 85)
#     for name, res in results.items():
#         print(f"{name:<25} | {res['Fertility']:<10.4f} | {res['Subword Rate (%)']:<10.2f} | {res['UNK Rate (%)']:<10.4f} | {res['Avg Seq Len']:<10.2f}")
#     print("="*85)
    
#     print("\n💡 해석 가이드:")
#     print("1. Fertility가 낮을수록: 사투리 어휘를 '하나의 의미 단위'로 잘 파악하고 있음.")
#     print("2. UNK Rate가 낮을수록: 기존 모델이 모르는 사투리 단어를 내 모델은 알고 있음.")
#     print("3. Avg Seq Len이 짧을수록: 학습 및 추론 속도가 더 빠르고 효율적임.")

# if __name__ == "__main__":
#     run_statistical_comparison()

import os
import random
import math
from collections import Counter

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer


def compute_zipf_metrics(token_counter):
    """
    Lotz et al. (ACL 2025) 스타일의 간단한 Zipf-inspired intrinsic metrics.
    """
    freqs = np.array(sorted(token_counter.values(), reverse=True), dtype=np.float64)
    if len(freqs) < 10:
        return {
            "Zipf Cardinality": len(freqs),
            "Zipf AUC": 0.0,
            "Zipf Slope": 0.0,
            "PowerLaw RMSE": 0.0,
        }

    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # ACL 2025 논문 설명에 맞춰 log(rank) <= 6 범위 사용
    mask = log_ranks <= 6
    x = log_ranks[mask]
    y = log_freqs[mask]

    if len(x) < 5:
        return {
            "Zipf Cardinality": len(freqs),
            "Zipf AUC": 0.0,
            "Zipf Slope": 0.0,
            "PowerLaw RMSE": 0.0,
        }

    # slope
    slope, intercept = np.polyfit(x, y, 1)

    # power law deviation (RMSE)
    y_pred = intercept + slope * x
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # rank-frequency AUC on log-log
    auc = float(np.trapz(y, x))

    return {
        "Zipf Cardinality": int(len(freqs)),
        "Zipf AUC": auc,
        "Zipf Slope": float(slope),
        "PowerLaw RMSE": rmse,
    }


def word_level_continued_stats(tokenizer, sentence):
    """
    Rust et al. (ACL 2021) 스타일:
    'continued word proportion' = 여러 subtoken으로 쪼개진 단어 비율
    """
    words = sentence.strip().split()
    if not words:
        return 0, 0

    continued_words = 0
    valid_words = 0

    for word in words:
        pieces = tokenizer.tokenize(word)
        if len(pieces) == 0:
            continue
        valid_words += 1
        if len(pieces) >= 2:
            continued_words += 1

    return continued_words, valid_words


def get_tokenizer_stats(tokenizer, sentences, name):
    total_words = 0
    total_tokens = 0
    total_unks = 0
    total_chars = 0
    total_continued_words = 0
    total_valid_words = 0
    seq_lengths = []
    token_counter = Counter()

    print(f"\n[{name}] 통계 분석 중...")

    for sentence in tqdm(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        if not words:
            continue

        tokens = tokenizer.tokenize(sentence)

        total_words += len(words)
        total_tokens += len(tokens)
        total_chars += len(sentence.replace(" ", ""))
        total_unks += sum(1 for t in tokens if t == "[UNK]")
        seq_lengths.append(len(tokens))
        token_counter.update(tokens)

        continued_words, valid_words = word_level_continued_stats(tokenizer, sentence)
        total_continued_words += continued_words
        total_valid_words += valid_words

    fertility = total_tokens / total_words if total_words > 0 else 0.0
    continued_word_rate = (total_continued_words / total_valid_words * 100) if total_valid_words > 0 else 0.0
    unk_rate = (total_unks / total_tokens * 100) if total_tokens > 0 else 0.0
    avg_seq_len = float(np.mean(seq_lengths)) if seq_lengths else 0.0
    chars_per_token = (total_chars / total_tokens) if total_tokens > 0 else 0.0
    tokens_per_char = (total_tokens / total_chars) if total_chars > 0 else 0.0

    zipf_metrics = compute_zipf_metrics(token_counter)

    result = {
        "Fertility": fertility,
        "Continued Word Rate (%)": continued_word_rate,
        "UNK Rate (%)": unk_rate,
        "Avg Seq Len": avg_seq_len,
        "Chars/Token": chars_per_token,
        "Tokens/Char": tokens_per_char,
        "Corpus Token Count": int(total_tokens),
        "Unique Token Count": int(len(token_counter)),
    }
    result.update(zipf_metrics)
    return result


def run_statistical_comparison():
    random.seed(42)

    my_tokenizer_path = "./dialect_bert_tokenizer"
    corpus_file = "dialect_corpus.txt"

    if not os.path.exists(corpus_file):
        print(f"⚠️ {corpus_file} 파일이 없습니다.")
        return

    try:
        my_tokenizer = BertTokenizer.from_pretrained(my_tokenizer_path, do_lower_case=False)
    except Exception as e:
        print(f"⚠️ 내 토크나이저 로드 실패: {e}")
        return

    google_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    klue_tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

    print("데이터 로딩 중...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        all_sentences = [line.strip() for line in f if line.strip()]

    sample_size = min(100000, len(all_sentences))
    test_sentences = random.sample(all_sentences, sample_size)
    print(f"✅ {sample_size:,}개의 문장을 샘플링하여 분석합니다.")

    tokenizers = {
        "My Dialect Tokenizer": my_tokenizer,
        "Google Multilingual": google_tokenizer,
        "KLUE BERT": klue_tokenizer
    }

    results = {}
    for name, tokenizer in tokenizers.items():
        results[name] = get_tokenizer_stats(tokenizer, test_sentences, name)

    print("\n" + "=" * 150)
    header = (
        f"{'Tokenizer':<22} | {'Fertility':>9} | {'ContWord%':>10} | {'UNK%':>8} | "
        f"{'AvgLen':>8} | {'Char/Tok':>8} | {'Tok/Char':>8} | {'UniqueTok':>10} | "
        f"{'ZipfAUC':>10} | {'ZipfSlope':>10} | {'PL-RMSE':>10}"
    )
    print(header)
    print("-" * 150)

    for name, r in results.items():
        print(
            f"{name:<22} | "
            f"{r['Fertility']:>9.4f} | "
            f"{r['Continued Word Rate (%)']:>10.2f} | "
            f"{r['UNK Rate (%)']:>8.4f} | "
            f"{r['Avg Seq Len']:>8.2f} | "
            f"{r['Chars/Token']:>8.4f} | "
            f"{r['Tokens/Char']:>8.4f} | "
            f"{r['Unique Token Count']:>10,d} | "
            f"{r['Zipf AUC']:>10.4f} | "
            f"{r['Zipf Slope']:>10.4f} | "
            f"{r['PowerLaw RMSE']:>10.4f}"
        )

    print("=" * 150)

    print("\n해석 포인트")
    print("1. Fertility가 낮을수록 단어를 덜 잘게 쪼개는 경향이 있습니다.")
    print("2. Continued Word Rate가 낮을수록 단어 단위 보존이 더 잘 됩니다.")
    print("3. UNK Rate가 낮을수록 어휘 커버리지가 좋습니다.")
    print("4. Avg Seq Len, Tokens/Char가 낮을수록 효율성이 좋습니다.")
    print("5. Zipf 지표는 토큰 분포가 자연어 분포와 얼마나 비슷한지 보는 참고 지표입니다.")


if __name__ == "__main__":
    run_statistical_comparison()