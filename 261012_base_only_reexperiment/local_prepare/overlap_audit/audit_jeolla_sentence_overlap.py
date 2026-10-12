"""Read-only exact sentence-set comparison of old Jeolla and new Validation.

Only stores candidate strings from the smaller new corpus in memory. The old
corpus is streamed in bounded batches. No hash approximation or data filtering.
"""
import ast
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r"D:\documents\project_1\git\kodialect-bert")
OLD = Path(r"D:\documents\project_1\git\project1_dataset\JSON만_모은폴더_전라도")
NEW = Path(r"D:\방언 데이터 총정리\015.한국어 방언 발화 데이터(전라도)\01.데이터\2.Validation")
OUT = Path(__file__).resolve().parent
REGION_CODE = ROOT / "260819_4tokenizer_5seed_region_classification/step1_prepare_data_80_10_10.py"
PARALLEL_CODE = ROOT / "260811_dialect_to_standard_4tokenizer_seq2seq/experiment_common.py"
BASES = ("raw_dialect", "normalized_dialect", "cleaned_region_dialect", "normalized_pair")


def load_function(path, name):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    ns = {"re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name], ast.get_source_segment(source, node)


clean_base_text, CLEAN_SOURCE = load_function(REGION_CODE, "clean_base_text")
normalize_text, NORMALIZE_SOURCE = load_function(PARALLEL_CODE, "normalize_text")


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        obj = json.load(stream)
    return path, [(row.get("dialect_form"), row.get("standard_form")) for row in obj.get("utterance", [])]


def iter_files(paths, workers=8, batch_size=128):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(paths), batch_size):
            yield from pool.map(read_rows, paths[offset:offset + batch_size])


def keys(dialect, standard):
    if not isinstance(dialect, str) or not dialect.strip():
        return (None, None, None, None)
    normalized = normalize_text(dialect)
    cleaned = clean_base_text(dialect)
    norm_standard = normalize_text(standard) if isinstance(standard, str) else ""
    return (dialect, normalized or None, cleaned or None,
            (normalized, norm_standard) if normalized and norm_standard else None)


def length_band(key):
    value = key[0] if isinstance(key, tuple) else key
    length = len(value)
    return "chars_0_5" if length <= 5 else "chars_6_19" if length < 20 else "chars_20_plus"


def pct(num, denom):
    return round(num / denom * 100, 6) if denom else None


def main():
    start = time.monotonic()
    old_paths = sorted(OLD.rglob("*.json"))
    new_paths = sorted(NEW.rglob("*.json"))
    assert len(old_paths) == 7412, len(old_paths)
    assert len(new_paths) == 994, len(new_paths)
    old_ids = {p.stem for p in old_paths}
    new_ids = {p.stem for p in new_paths}
    assert len(old_ids) == len(old_paths)
    assert len(new_ids) == len(new_paths)
    candidates = {basis: set() for basis in BASES}
    matches = {basis: set() for basis in BASES}
    new_valid = Counter()
    old_valid = Counter()
    old_matching = Counter()
    raw_new_rows = 0
    raw_old_rows = 0
    first_long_overlap = None
    print("Building exact new-Validation string sets...", flush=True)
    for _, rows in iter_files(new_paths):
        raw_new_rows += len(rows)
        for dialect, standard in rows:
            for basis, key in zip(BASES, keys(dialect, standard)):
                if key is not None:
                    candidates[basis].add(key)
                    new_valid[basis] += 1
    assert raw_new_rows == 275137, raw_new_rows
    print(f"New Validation: {len(new_paths)} files / {raw_new_rows} rows. Scanning old corpus...", flush=True)
    for file_count, (old_path, rows) in enumerate(iter_files(old_paths), start=1):
        raw_old_rows += len(rows)
        for dialect, standard in rows:
            for basis, key in zip(BASES, keys(dialect, standard)):
                if key is not None:
                    old_valid[basis] += 1
                    if key in candidates[basis]:
                        matches[basis].add(key)
                        old_matching[basis] += 1
                        if basis == "normalized_dialect" and len(key) >= 20 and first_long_overlap is None:
                            first_long_overlap = {"old_file_id": old_path.stem, "characters": len(key),
                                                  "normalized_text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()}
                            print("FIRST_LONG_OVERLAP " + json.dumps(first_long_overlap), flush=True)
        if file_count % 512 == 0 or file_count == len(old_paths):
            long_count = sum(len(key) >= 20 for key in matches["normalized_dialect"])
            print(f"Old files {file_count}/{len(old_paths)}; rows {raw_old_rows}; unique long normalized overlaps {long_count}; elapsed {time.monotonic()-start:.1f}s", flush=True)
    assert raw_old_rows == 1992101, raw_old_rows
    new_matching = Counter()
    valid_by_band = {basis: Counter() for basis in BASES}
    matched_by_band = {basis: Counter() for basis in BASES}
    per_file = []
    print("Counting matching new rows and per-file long-text overlap...", flush=True)
    for path, rows in iter_files(new_paths):
        totals = Counter()
        overlap = Counter()
        long_totals = Counter()
        long_overlap = Counter()
        for dialect, standard in rows:
            for basis, key in zip(BASES, keys(dialect, standard)):
                if key is None:
                    continue
                band = length_band(key)
                totals[basis] += 1
                valid_by_band[basis][band] += 1
                if band == "chars_20_plus":
                    long_totals[basis] += 1
                if key in matches[basis]:
                    overlap[basis] += 1
                    new_matching[basis] += 1
                    matched_by_band[basis][band] += 1
                    if band == "chars_20_plus":
                        long_overlap[basis] += 1
        per_file.append({
            "path": str(path), "file_id": path.stem, "raw_rows": len(rows),
            "by_basis": {basis: {
                "valid_rows": totals[basis], "matching_rows": overlap[basis],
                "matching_pct": pct(overlap[basis], totals[basis]),
                "long_valid_rows": long_totals[basis],
                "long_matching_rows": long_overlap[basis],
                "long_matching_pct": pct(long_overlap[basis], long_totals[basis]),
            } for basis in BASES},
        })
    summary = {
        "scope": "Exact cross-file sentence-set overlap, old Jeolla JSON vs new Jeolla Validation JSON; no source mutations.",
        "old_root": str(OLD), "new_root": str(NEW),
        "old_files": len(old_paths), "new_files": len(new_paths),
        "common_file_ids": sorted(old_ids & new_ids),
        "old_raw_utterance_rows": raw_old_rows, "new_raw_utterance_rows": raw_new_rows,
        "first_long_overlap": first_long_overlap,
        "methods": {
            "comparison": "Exact Python string/tuple equality; sets deduplicate comparison keys but row-match counts retain every row occurrence.",
            "raw_dialect": "Unmodified nonblank dialect_form; whitespace and punctuation significant.",
            "normalized_dialect": "Existing translation normalize_text: NFKC and whitespace collapse.",
            "cleaned_region_dialect": "Existing region clean_base_text; discard blank results, no additional whitespace normalization.",
            "normalized_pair": "Both normalized dialect_form and standard_form equal as an ordered tuple.",
            "length_bands": "Python len of the corresponding dialect key, including whitespace/punctuation; thresholds <=5, 6-19, >=20 are descriptive only.",
            "long_file_diagnostic": ">90% of >=20-character new rows match any old row; require >=20 such new rows to avoid tiny denominators. This does not itself prove a copied recording.",
            "clean_function_source": str(REGION_CODE), "clean_function": CLEAN_SOURCE,
            "normalize_function_source": str(PARALLEL_CODE), "normalize_function": NORMALIZE_SOURCE,
            "notes": "Same text can be a generic conversational expression. No fuzzy matching, speaker matching, or automated filtering performed.",
        },
        "by_basis": {},
        "elapsed_seconds": round(time.monotonic()-start, 2),
    }
    for basis in BASES:
        matched_unique_by_band = Counter(length_band(key) for key in matches[basis])
        new_unique_by_band = Counter(length_band(key) for key in candidates[basis])
        high = [f for f in per_file if f["by_basis"][basis]["long_valid_rows"] >= 20
                and f["by_basis"][basis]["long_matching_pct"] > 90]
        top = sorted((f for f in per_file if f["by_basis"][basis]["long_valid_rows"] >= 20),
                     key=lambda f: f["by_basis"][basis]["long_matching_pct"], reverse=True)[:10]
        summary["by_basis"][basis] = {
            "old_valid_rows": old_valid[basis], "new_valid_rows": new_valid[basis],
            "new_unique_keys": len(candidates[basis]), "overlapping_unique_keys": len(matches[basis]),
            "new_matching_rows": new_matching[basis], "new_matching_pct": pct(new_matching[basis], new_valid[basis]),
            "old_matching_rows": old_matching[basis],
            "new_nonmatching_rows": new_valid[basis]-new_matching[basis],
            "by_length": {band: {
                "new_rows": valid_by_band[basis][band], "new_matching_rows": matched_by_band[basis][band],
                "new_matching_pct": pct(matched_by_band[basis][band], valid_by_band[basis][band]),
                "new_unique_keys": new_unique_by_band[band], "overlapping_unique_keys": matched_unique_by_band[band],
            } for band in ("chars_0_5", "chars_6_19", "chars_20_plus")},
            "files_over_90pct_long_rows": [{"file_id": f["file_id"], **f["by_basis"][basis]} for f in high],
            "top_10_files_by_long_matching_pct": [{"file_id": f["file_id"], **f["by_basis"][basis]} for f in top],
        }
    summary["safe_common_phrase_examples"] = [phrase for phrase in ("네", "예", "응", "어", "그렇지", "맞아", "아니", "그래", "네.", "예.") if phrase in matches["normalized_dialect"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jeolla_sentence_overlap.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "jeolla_sentence_overlap_per_file.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_file:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    labels = {"raw_dialect":"방언 원문", "normalized_dialect":"번역 입력 정규화", "cleaned_region_dialect":"지역 분류 전처리", "normalized_pair":"정규화한 방언-표준어 쌍"}
    lines = ["전라도 기존 JSON과 새 Validation 전체 발화 교집합 확인", "",
             f"기존: {OLD}", f"새 자료: {NEW}",
             f"기존 {len(old_paths):,}파일 / {raw_old_rows:,}발화; 새 자료 {len(new_paths):,}파일 / {raw_new_rows:,}발화",
             f"동일 파일 ID: {len(old_ids & new_ids):,}개", "",
             "기준\t새 자료 유효 발화\t기존에 같은 내용이 있는 새 발화\t비율\t교집합 고유 문자열(또는 쌍)"]
    for basis in BASES:
        s = summary["by_basis"][basis]
        lines.append(f"{labels[basis]}\t{s['new_valid_rows']:,}\t{s['new_matching_rows']:,}\t{s['new_matching_pct']:.4f}%\t{s['overlapping_unique_keys']:,}")
        for band, b in s["by_length"].items():
            rate = f"{b['new_matching_pct']:.4f}%" if b['new_matching_pct'] is not None else "N/A"
            lines.append(f"  {band}: 새 발화 {b['new_rows']:,}, 중복 {b['new_matching_rows']:,} ({rate}), 고유 교집합 {b['overlapping_unique_keys']:,}")
        lines.append(f"  20자 이상 발화가 20개 이상이고 그중 90% 초과 중복인 파일: {len(s['files_over_90pct_long_rows']):,}")
    lines.extend(["", "유의 사항:",
                  "- 파일명이 다른 자료 전체를 가로질러 정확히 같은 문자열/문장 쌍이 있는지 비교하였다.",
                  "- 발화 수는 중복 출현을 유지한다. 고유 교집합은 같은 문자열/쌍을 하나로 센다.",
                  "- 네, 예 등 일반적인 대화 표현도 정확 일치에 포함되므로 텍스트 일치 자체가 같은 녹음 중복을 뜻하지는 않는다.",
                  "- 5자와 20자 기준은 길이별 현상을 설명하기 위한 분류일 뿐, 필터링 규칙으로 적용하지 않았다.",
                  "- 기존 실험의 전처리 함수를 AST로 직접 불러와 사용했다. 원천 자료 변경, 중복 제거, 학습 데이터 포함 결정은 하지 않았다.",
                  "- 무오류로 모든 파일을 읽었으며, 예상 원본 발화 수와 일치함을 확인했다.",
                  f"소요 시간: {summary['elapsed_seconds']:.2f}초"])
    (OUT / "jeolla_sentence_overlap.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "jeolla_sentence_overlap.json"),
                      "raw_rows": raw_new_rows,
                      "result": {basis: {key: summary['by_basis'][basis][key] for key in ('new_matching_rows','new_matching_pct','overlapping_unique_keys')} for basis in BASES},
                      "elapsed_seconds": summary['elapsed_seconds']}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
