from pathlib import Path
import json


CORPUS_FILES = {
    "전체 코퍼스": "dialect_corpus.txt",
    "학습 코퍼스": "dialect_train_corpus.txt",
    "평가 코퍼스": "dialect_eval_corpus.txt",
}

STATS_JSON = "corpus_split_stats.json"


def format_size(size_bytes: int) -> str:
    """bytes를 보기 좋게 변환"""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def count_lines(file_path: Path) -> int:
    """코퍼스 파일의 문장 수 계산: 한 줄 = 한 문장"""
    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def inspect_corpus_file(name: str, path: str) -> dict:
    file_path = Path(path)

    if not file_path.exists():
        return {
            "name": name,
            "path": str(file_path),
            "exists": False,
        }

    size_bytes = file_path.stat().st_size
    sentence_count = count_lines(file_path)

    return {
        "name": name,
        "path": str(file_path),
        "exists": True,
        "sentence_count": sentence_count,
        "size_bytes": size_bytes,
        "size_readable": format_size(size_bytes),
        "avg_bytes_per_sentence": size_bytes / sentence_count if sentence_count else 0,
    }


def print_corpus_summary():
    print("\n" + "=" * 80)
    print("코퍼스 파일 정보")
    print("=" * 80)

    total_sentences = 0
    total_size = 0

    for name, path in CORPUS_FILES.items():
        info = inspect_corpus_file(name, path)

        if not info["exists"]:
            print(f"[{name}] 파일 없음: {info['path']}")
            continue

        total_sentences += info["sentence_count"]
        total_size += info["size_bytes"]

        print(f"\n[{info['name']}]")
        print(f"- 파일 경로: {info['path']}")
        print(f"- 문장 수: {info['sentence_count']:,}문장")
        print(f"- 파일 크기: {info['size_readable']} ({info['size_bytes']:,} bytes)")
        print(f"- 문장당 평균 용량: {info['avg_bytes_per_sentence']:.2f} bytes")

    print("\n" + "-" * 80)
    print(f"전체 합산 문장 수: {total_sentences:,}문장")
    print(f"전체 합산 파일 크기: {format_size(total_size)} ({total_size:,} bytes)")
    print("=" * 80)


def print_stats_json_summary():
    stats_path = Path(STATS_JSON)

    if not stats_path.exists():
        print(f"\nstats json 파일 없음: {STATS_JSON}")
        return

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    print("\n" + "=" * 80)
    print("corpus_split_stats.json 기준 지역별 통계")
    print("=" * 80)

    by_region = stats.get("by_region", {})
    total = stats.get("total", {})

    for region, data in by_region.items():
        train_count = data.get("train_count", 0)
        eval_count = data.get("eval_count", 0)
        total_count = data.get("total_count", 0)
        total_size = data.get("total_size", 0)

        eval_ratio = eval_count / total_count if total_count else 0

        print(f"\n[{region}]")
        print(f"- Train 문장 수: {train_count:,}")
        print(f"- Eval 문장 수: {eval_count:,}")
        print(f"- 전체 문장 수: {total_count:,}")
        print(f"- Eval 비율: {eval_ratio:.2%}")
        print(f"- 용량: {format_size(total_size)}")

    print("\n" + "-" * 80)
    print("[합계]")
    print(f"- Train 문장 수: {total.get('train_count', 0):,}")
    print(f"- Eval 문장 수: {total.get('eval_count', 0):,}")
    print(f"- 전체 문장 수: {total.get('total_count', 0):,}")
    print(f"- 전체 용량: {format_size(total.get('total_size', 0))}")
    print("=" * 80)


if __name__ == "__main__":
    print_corpus_summary()
    print_stats_json_summary()