from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from collections import Counter
from pathlib import Path
from scipy.integrate import simpson
import numpy as np
import pandas as pd
import argparse
import json


TOKENIZER_SPECS = {
    "Dialect WordPiece": {
        "type": "local_wordpiece",
        "vocab_path": "./dialect_bert_tokenizer/vocab.txt"
    },
    "Google Multilingual": {
        "type": "hf",
        "model_name": "google-bert/bert-base-multilingual-cased",
        "trust_remote_code": False
    },
    "KoBERT": {
        "type": "hf",
        "model_name": "monologg/kobert",
        "trust_remote_code": True
    }
}


def load_tokenizer(spec):
    if spec["type"] == "local_wordpiece":
        vocab_path = Path(spec["vocab_path"])

        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab.txt를 찾을 수 없습니다: {vocab_path}")

        return BertWordPieceTokenizer(
            str(vocab_path),
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
            wordpieces_prefix="##"
        )

    if spec["type"] == "hf":
        return AutoTokenizer.from_pretrained(
            spec["model_name"],
            trust_remote_code=spec.get("trust_remote_code", False)
        )

    raise ValueError(f"지원하지 않는 tokenizer type: {spec['type']}")


def tokenize_batch(tokenizer, tokenizer_type, texts):
    """
    tokenizer 종류가 달라도 tokens list를 동일한 형태로 반환.
    """

    if tokenizer_type == "local_wordpiece":
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        return [enc.tokens for enc in encodings]

    if tokenizer_type == "hf":
        try:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False
            )

            input_ids_batch = encoded["input_ids"]

            # batch가 1개일 때 일부 tokenizer가 list[int]로 줄 수 있어 보정
            if input_ids_batch and isinstance(input_ids_batch[0], int):
                input_ids_batch = [input_ids_batch]

            return [
                tokenizer.convert_ids_to_tokens(input_ids)
                for input_ids in input_ids_batch
            ]

        except Exception:
            # KoBERT 등 custom tokenizer에서 batch call이 불안정할 경우 fallback
            return [tokenizer.tokenize(text) for text in texts]

    raise ValueError(f"지원하지 않는 tokenizer type: {tokenizer_type}")


def iter_batches(corpus_path, batch_size=1024, max_lines=None):
    batch = []
    line_count = 0

    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            batch.append(text)
            line_count += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

            if max_lines is not None and line_count >= max_lines:
                break

    if batch:
        yield batch


def collect_token_stats(tokenizer_name, tokenizer, tokenizer_type, corpus_path, batch_size, max_lines):
    token_counter = Counter()

    total_sentences = 0
    total_tokens = 0
    total_chars = 0
    total_words = 0
    unk_count = 0

    for batch in iter_batches(corpus_path, batch_size=batch_size, max_lines=max_lines):
        tokens_batch = tokenize_batch(tokenizer, tokenizer_type, batch)

        for text, tokens in zip(batch, tokens_batch):
            token_counter.update(tokens)

            total_sentences += 1
            total_tokens += len(tokens)
            total_chars += len(text)
            total_words += len(text.split())
            unk_count += sum(1 for t in tokens if t in ["[UNK]", "<unk>", "<UNK>", "▁<unk>"])

    return {
        "tokenizer": tokenizer_name,
        "token_counter": token_counter,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_words": total_words,
        "unk_count": unk_count
    }


def compute_zipf_metrics(token_counter, rank_log_cutoff=6.0):
    """
    [14] 논문 방식에 맞춘 Zipf 기반 지표 계산.

    Cardinality:
        tokenization 이후 실제 등장한 unique token 수

    Zipf AUC:
        log(rank)-log(frequency) 곡선 아래 면적

    Zipf Slope:
        log(rank)-log(frequency) 선형 근사 기울기

    Power-law Error:
        실제 log-frequency와 선형 근사값 간 평균절대오차

    논문 기준:
        AUC, Slope, Power-law Error는 log(rank) <= 6 구간에서 계산
    """

    if len(token_counter) == 0:
        raise ValueError("token_counter가 비어 있습니다.")

    freqs = np.array(
        sorted(token_counter.values(), reverse=True),
        dtype=np.float64
    )

    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    mask = log_ranks <= rank_log_cutoff

    x = log_ranks[mask]
    y = log_freqs[mask]

    if len(x) < 2:
        raise ValueError("Zipf fitting에 사용할 token 수가 너무 적습니다.")

    cardinality = len(token_counter)

    zipf_auc = float(simpson(y, x=x))

    slope, intercept = np.polyfit(x, y, deg=1)
    slope = float(slope)
    intercept = float(intercept)

    y_hat = intercept + slope * x
    power_law_error = float(np.mean(np.abs(y - y_hat)))

    return {
        "cardinality": cardinality,
        "zipf_auc": zipf_auc,
        "zipf_slope": slope,
        "zipf_intercept": intercept,
        "power_law_error": power_law_error,
        "zipf_fit_token_count": int(len(x))
    }


def compute_basic_metrics(stats):
    total_sentences = stats["total_sentences"]
    total_tokens = stats["total_tokens"]
    total_chars = stats["total_chars"]
    total_words = stats["total_words"]
    unk_count = stats["unk_count"]

    return {
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "avg_sequence_length": total_tokens / total_sentences if total_sentences else 0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "fertility": total_tokens / total_words if total_words else 0,
        "unk_count": unk_count,
        "unk_rate": unk_count / total_tokens if total_tokens else 0
    }


def save_zipf_curve(tokenizer_name, token_counter, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = tokenizer_name.replace(" ", "_").replace("/", "_")

    freqs = np.array(
        sorted(token_counter.values(), reverse=True),
        dtype=np.float64
    )
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    df = pd.DataFrame({
        "rank": ranks.astype(int),
        "frequency": freqs.astype(int),
        "log_rank": np.log(ranks),
        "log_frequency": np.log(freqs)
    })

    df.to_csv(output_dir / f"{safe_name}_zipf_curve.csv", index=False, encoding="utf-8-sig")


def compare_tokenizers(corpus_path, batch_size=1024, max_lines=None, rank_log_cutoff=6.0, output_dir="./tokenizer_eval_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    full_results = {}

    for tokenizer_name, spec in TOKENIZER_SPECS.items():
        print(f"\n=== {tokenizer_name} 평가 시작 ===")

        tokenizer = load_tokenizer(spec)

        stats = collect_token_stats(
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer,
            tokenizer_type=spec["type"],
            corpus_path=corpus_path,
            batch_size=batch_size,
            max_lines=max_lines
        )

        basic_metrics = compute_basic_metrics(stats)
        zipf_metrics = compute_zipf_metrics(
            stats["token_counter"],
            rank_log_cutoff=rank_log_cutoff
        )

        row = {
            **basic_metrics,
            **zipf_metrics
        }

        rows[tokenizer_name] = row

        full_results[tokenizer_name] = {
            "basic_metrics": basic_metrics,
            "zipf_metrics": zipf_metrics,
            "top_50_tokens": stats["token_counter"].most_common(50)
        }

        save_zipf_curve(
            tokenizer_name=tokenizer_name,
            token_counter=stats["token_counter"],
            output_dir=output_dir
        )

        print(f"{tokenizer_name} 완료")
        print(pd.Series(row))

    result_df = pd.DataFrame(rows).T

    # 논문/보고서에 보기 좋게 주요 지표 순서 정리
    columns_order = [
        "total_sentences",
        "total_tokens",
        "avg_sequence_length",
        "tokens_per_char",
        "chars_per_token",
        "fertility",
        "unk_count",
        "unk_rate",
        "cardinality",
        "zipf_auc",
        "zipf_slope",
        "power_law_error",
        "zipf_fit_token_count"
    ]

    result_df = result_df[columns_order]

    result_csv_path = output_dir / "tokenizer_comparison_metrics.csv"
    result_json_path = output_dir / "tokenizer_comparison_full_results.json"

    result_df.to_csv(result_csv_path, encoding="utf-8-sig")

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print("\n\n=== 최종 비교 결과 ===")
    print(result_df)

    print(f"\nCSV 저장 완료: {result_csv_path}")
    print(f"JSON 저장 완료: {result_json_path}")

    return result_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        default="dialect_eval_corpus.txt",
        help="평가용 사투리 corpus txt 파일"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="빠른 테스트용. 전체 평가 시 생략"
    )

    parser.add_argument(
        "--rank_log_cutoff",
        type=float,
        default=6.0,
        help="[14] 논문 기준 log(rank) cutoff"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer_eval_results"
    )

    args = parser.parse_args()

    compare_tokenizers(
        corpus_path=args.corpus,
        batch_size=args.batch_size,
        max_lines=args.max_lines,
        rank_log_cutoff=args.rank_log_cutoff,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()