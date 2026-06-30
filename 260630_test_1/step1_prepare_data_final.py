import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


# ============================================================
# 기본 설정: 기존 코드의 데이터 디렉토리 구조를 그대로 유지
# ============================================================
REGION_DIRS = {
    "강원도": "JSON만_모은폴더_강원도",
    "경상도": "JSON만_모은폴더_경상도",
    "전라도": "JSON만_모은폴더_전라도",
    "제주도": "JSON만_모은폴더_제주도",
    "충청도": "JSON만_모은폴더_충청도",
}

EXTRA_REGION_DIRS = {
    "강원도": [
        "강원도_01_1인발화_따라말하기",
        "강원도_02_1인발화_질문에답하기",
        "강원도_03_2인발화",
    ],
    "경상도": [
        "경상도_01_1인발화_따라말하기",
        "경상도_02_1인발화_질문에답하기",
        "경상도_03_2인발화",
    ],
}

DEFAULT_DATA_DIR = "../../project1_dataset"
DEFAULT_ALL_CORPUS = "dialect_corpus.txt"
DEFAULT_TRAIN_CORPUS = "dialect_train_corpus.txt"
DEFAULT_EVAL_CORPUS = "dialect_eval_corpus.txt"
DEFAULT_MANIFEST = "corpus_split_manifest.csv"
DEFAULT_STATS_JSON = "corpus_split_stats.json"


# ============================================================
# 텍스트 정제 로직: 기존 코드의 정제 방식을 유지
# ============================================================
def clean_base_text(text: str) -> str:
    """기존 지역별 JSON만_모은폴더_* 데이터의 dialect_form 정제."""
    cleaned_text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned_text = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned_text).strip()
    return cleaned_text


def clean_extra_text(text: str) -> str:
    """강원도/경상도 추가 코퍼스의 transcription.dialect 정제."""
    cleaned_text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned_text = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


# ============================================================
# JSON 파일에서 문장 추출
# ============================================================
def extract_texts_from_json(file_path: Path, source_type: str) -> List[str]:
    """
    source_type:
        - base: data["utterance"][i]["dialect_form"]
        - extra: data["transcription"]["dialect"]
    """
    texts: List[str] = []

    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if source_type == "base":
        utterances = data.get("utterance", [])
        for u in utterances:
            text = u.get("dialect_form", "")
            if isinstance(text, str) and text.strip():
                cleaned = clean_base_text(text)
                if cleaned:
                    texts.append(cleaned)

    elif source_type == "extra":
        text = data.get("transcription", {}).get("dialect", "")
        if isinstance(text, str) and text.strip():
            cleaned = clean_extra_text(text)
            if cleaned:
                texts.append(cleaned)

    else:
        raise ValueError(f"지원하지 않는 source_type입니다: {source_type}")

    return texts


# ============================================================
# 파일 목록 수집
# ============================================================
def collect_json_file_records(data_dir: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    # 1. 기존 코퍼스
    for region, subdir in REGION_DIRS.items():
        dir_path = data_dir / subdir
        if not dir_path.exists():
            print(f"⚠️ 경로 없음: {dir_path}")
            continue

        json_files = sorted(p for p in dir_path.iterdir() if p.suffix == ".json")
        for path in json_files:
            records.append(
                {
                    "region": region,
                    "source_group": subdir,
                    "source_type": "base",
                    "path": str(path),
                }
            )

    # 2. 추가 코퍼스
    for region, subdirs in EXTRA_REGION_DIRS.items():
        for subdir in subdirs:
            dir_path = data_dir / subdir
            if not dir_path.exists():
                print(f"⚠️ 경로 없음: {dir_path}")
                continue

            json_files = sorted(p for p in dir_path.iterdir() if p.suffix == ".json")
            for path in json_files:
                records.append(
                    {
                        "region": region,
                        "source_group": subdir,
                        "source_type": "extra",
                        "path": str(path),
                    }
                )

    return records


# ============================================================
# 1차 스캔: 파일별 문장 수/용량 계산
# ============================================================
def scan_file_records(records: List[Dict[str, str]]) -> List[Dict[str, object]]:
    scanned: List[Dict[str, object]] = []

    print("--- 1차 스캔: JSON 파일별 문장 수 계산 시작 ---")
    for rec in tqdm(records, desc="파일 스캔 진행도"):
        file_path = Path(rec["path"])
        try:
            texts = extract_texts_from_json(file_path, rec["source_type"])
            num_sentences = len(texts)
            size_bytes = sum(len((t + "\n").encode("utf-8")) for t in texts)

            if num_sentences > 0:
                scanned.append(
                    {
                        **rec,
                        "num_sentences": num_sentences,
                        "size_bytes": size_bytes,
                    }
                )

        except Exception as e:
            print(f"\n⚠️ 파일 오류 발생: {file_path} - {e}")

    return scanned


# ============================================================
# 파일 단위 train/eval split
# ============================================================
def split_records_by_file(
    records: List[Dict[str, object]],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    같은 JSON 파일 안의 utterance가 train/eval에 동시에 들어가지 않도록 파일 단위로 split.
    region + source_group별로 나누어 각 그룹에서 eval_ratio에 가깝게 분리한다.
    """
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio는 0과 1 사이여야 합니다.")

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["region"]), str(rec["source_group"]))].append(rec)

    split_records: List[Dict[str, object]] = []
    rng = random.Random(seed)

    print("--- 파일 단위 train/eval split 시작 ---")
    for (region, source_group), group_records in sorted(grouped.items()):
        group_records = list(group_records)
        rng.shuffle(group_records)

        total_sentences = sum(int(r["num_sentences"]) for r in group_records)
        target_eval = max(1, int(round(total_sentences * eval_ratio))) if total_sentences > 0 else 0

        eval_sentence_count = 0
        eval_file_count = 0

        for rec in group_records:
            if eval_sentence_count < target_eval:
                split = "eval"
                eval_sentence_count += int(rec["num_sentences"])
                eval_file_count += 1
            else:
                split = "train"

            split_records.append({**rec, "split": split})

        print(
            f"[{region} | {source_group}] "
            f"files={len(group_records):,}, sentences={total_sentences:,}, "
            f"eval_files={eval_file_count:,}, eval_sentences≈{eval_sentence_count:,}"
        )

    return split_records


# ============================================================
# 실제 corpus 파일 작성
# ============================================================
def update_stats(stats: Dict[str, Dict[str, int]], region: str, split: str, output_line: str) -> None:
    line_bytes = len(output_line.encode("utf-8"))

    stats[region][f"{split}_count"] += 1
    stats[region][f"{split}_size"] += line_bytes
    stats[region]["total_count"] += 1
    stats[region]["total_size"] += line_bytes


def write_corpora(
    records: List[Dict[str, object]],
    all_corpus_path: Path,
    train_corpus_path: Path,
    eval_corpus_path: Path,
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {
        region: {
            "train_count": 0,
            "train_size": 0,
            "eval_count": 0,
            "eval_size": 0,
            "total_count": 0,
            "total_size": 0,
        }
        for region in REGION_DIRS
    }

    print("--- 2차 처리: corpus 파일 작성 시작 ---")
    with all_corpus_path.open("w", encoding="utf-8") as f_all, \
        train_corpus_path.open("w", encoding="utf-8") as f_train, \
        eval_corpus_path.open("w", encoding="utf-8") as f_eval:

        for rec in tqdm(records, desc="corpus 작성 진행도"):
            file_path = Path(str(rec["path"]))
            region = str(rec["region"])
            split = str(rec["split"])

            try:
                texts = extract_texts_from_json(file_path, str(rec["source_type"]))
            except Exception as e:
                print(f"\n⚠️ 파일 오류 발생: {file_path} - {e}")
                continue

            target_file = f_train if split == "train" else f_eval

            for text in texts:
                output_line = text + "\n"
                f_all.write(output_line)
                target_file.write(output_line)
                update_stats(stats, region, split, output_line)

    return stats


# ============================================================
# manifest / stats 저장
# ============================================================
def save_manifest(records: List[Dict[str, object]], manifest_path: Path) -> None:
    fieldnames = [
        "region",
        "source_group",
        "source_type",
        "split",
        "num_sentences",
        "size_bytes",
        "path",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def save_stats_json(stats: Dict[str, Dict[str, int]], stats_path: Path, args: argparse.Namespace) -> None:
    total = {
        "train_count": sum(v["train_count"] for v in stats.values()),
        "train_size": sum(v["train_size"] for v in stats.values()),
        "eval_count": sum(v["eval_count"] for v in stats.values()),
        "eval_size": sum(v["eval_size"] for v in stats.values()),
        "total_count": sum(v["total_count"] for v in stats.values()),
        "total_size": sum(v["total_size"] for v in stats.values()),
    }

    payload = {
        "split_config": {
            "data_dir": args.data_dir,
            "eval_ratio": args.eval_ratio,
            "seed": args.seed,
            "all_corpus": args.all_corpus,
            "train_corpus": args.train_corpus,
            "eval_corpus": args.eval_corpus,
            "split_unit": "json_file",
            "note": "동일 JSON 파일의 문장이 train/eval에 동시에 들어가지 않도록 파일 단위로 분리함.",
        },
        "by_region": stats,
        "total": total,
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_stats_table(stats: Dict[str, Dict[str, int]]) -> None:
    total_train = sum(v["train_count"] for v in stats.values())
    total_eval = sum(v["eval_count"] for v in stats.values())
    total_count = sum(v["total_count"] for v in stats.values())
    total_size = sum(v["total_size"] for v in stats.values())

    print("\n" + "=" * 86)
    print(f"{'지역':<10} | {'Train 문장 수':>14} | {'Eval 문장 수':>13} | {'전체 문장 수':>13} | {'Eval 비율':>9} | {'용량'}")
    print("-" * 86)

    for region, data in stats.items():
        train_count = data["train_count"]
        eval_count = data["eval_count"]
        count = data["total_count"]
        ratio = eval_count / count if count else 0
        size_mb = data["total_size"] / (1024 * 1024)
        print(
            f"{region:<10} | {train_count:>14,} | {eval_count:>13,} | "
            f"{count:>13,} | {ratio:>8.2%} | {size_mb:>9.2f} MB"
        )

    print("-" * 86)
    total_ratio = total_eval / total_count if total_count else 0
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(
        f"{'합계':<10} | {total_train:>14,} | {total_eval:>13,} | "
        f"{total_count:>13,} | {total_ratio:>8.2%} | {total_size_gb:>9.2f} GB"
    )
    print("=" * 86)


# ============================================================
# Main
# ============================================================
def prepare_corpus(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)

    all_corpus_path = Path(args.all_corpus)
    train_corpus_path = Path(args.train_corpus)
    eval_corpus_path = Path(args.eval_corpus)
    manifest_path = Path(args.manifest)
    stats_path = Path(args.stats_json)

    print("--- 대용량 데이터 추출 및 train/eval corpus 생성 시작 ---")
    print(f"데이터 디렉토리: {data_dir}")
    print(f"Train corpus: {train_corpus_path}")
    print(f"Eval corpus:  {eval_corpus_path}")
    print(f"All corpus:   {all_corpus_path}")
    print(f"Eval ratio:   {args.eval_ratio:.2%}")
    print(f"Random seed:  {args.seed}")

    file_records = collect_json_file_records(data_dir)
    if not file_records:
        raise RuntimeError(f"처리할 JSON 파일을 찾지 못했습니다. data_dir를 확인하세요: {data_dir}")

    scanned_records = scan_file_records(file_records)
    if not scanned_records:
        raise RuntimeError("정제 후 사용 가능한 문장이 없습니다.")

    split_records = split_records_by_file(scanned_records, eval_ratio=args.eval_ratio, seed=args.seed)
    stats = write_corpora(split_records, all_corpus_path, train_corpus_path, eval_corpus_path)

    save_manifest(split_records, manifest_path)
    save_stats_json(stats, stats_path, args)
    print_stats_table(stats)

    print(f"\n✅ 전체 코퍼스 저장: {all_corpus_path}")
    print(f"✅ 학습 코퍼스 저장: {train_corpus_path}")
    print(f"✅ 평가 코퍼스 저장: {eval_corpus_path}")
    print(f"✅ 파일 단위 split manifest 저장: {manifest_path}")
    print(f"✅ split 통계 저장: {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사투리 JSON 데이터에서 train/eval corpus를 생성합니다.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="평가 corpus 비율. 기본값 0.1 = 10%")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_corpus", type=str, default=DEFAULT_ALL_CORPUS)
    parser.add_argument("--train_corpus", type=str, default=DEFAULT_TRAIN_CORPUS)
    parser.add_argument("--eval_corpus", type=str, default=DEFAULT_EVAL_CORPUS)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--stats_json", type=str, default=DEFAULT_STATS_JSON)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_corpus(parse_args())
