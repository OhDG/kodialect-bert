"""Read-only census of parenthesis notation in the selected source JSON files.

Counts are utterance rows, not the number of parenthesis occurrences. The
surface pattern (x)/(y) is a candidate paired transcription, not a linguistic
judgment. No source JSON, prepared training data, or model code is modified.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import csv
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ['강원도', '경상도', '전라도', '제주도', '충청도']
PAIR = re.compile(r'(?<!\()\(([^()]*)\)\s*/\s*\(([^()]*)\)(?!\))')
DELETE = re.compile(r'\([^)]*\)|\[[^\]]*\]')
WHITELIST = re.compile(r'[^가-힣a-zA-Z0-9.,?! ]')
SPACE = re.compile(r'\s+')
DOUBLE = re.compile(r'\(\([^()]*\)\)')


def normalize(text):
    return SPACE.sub(' ', unicodedata.normalize('NFKC', text)).strip()


def clean(text):
    return WHITELIST.sub('', DELETE.sub('', text)).strip()


def excerpt(value):
    return value[:400] if isinstance(value, str) else ''


def unbalanced(text):
    level = 0
    for char in text:
        if char == '(':
            level += 1
        elif char == ')':
            level -= 1
            if level < 0:
                return True
    return level != 0


def count_utterance(u):
    counts = Counter(utterance_rows=1)
    fields = {}
    pairs = {}
    for name in ('form', 'dialect_form', 'standard_form'):
        value = u.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        fields[name] = value
        counts[name + '_nonempty'] += 1
        has_round = '(' in value or ')' in value
        counts[name + '_with_open_round'] += '(' in value
        counts[name + '_with_round'] += has_round
        counts[name + '_with_square'] += '[' in value or ']' in value
        counts[name + '_with_fullwidth_round'] += '（' in value or '）' in value
        pairs[name] = [(normalize(m.group(1)), normalize(m.group(2))) for m in PAIR.finditer(value)]
        counts[name + '_with_pair_pattern'] += bool(pairs[name])
        counts[name + '_with_different_pair_sides'] += any(a != b for a, b in pairs[name])
        counts[name + '_with_double_round'] += bool(DOUBLE.search(value))
        if has_round:
            counts[name + '_unbalanced_round'] += unbalanced(value)
        if name == 'dialect_form':
            cleaned = clean(value)
            counts['classification_valid'] += bool(cleaned)
            counts['classification_empty'] += not cleaned
            counts['classification_empty_with_round'] += not cleaned and has_round
            if has_round:
                removed = DELETE.sub('', value)
                counts['dialect_form_parenthesis_rule_changed'] += removed != value
    d, s = fields.get('dialect_form'), fields.get('standard_form')
    kinds = []
    if d and s:
        nd, ns = normalize(d), normalize(s)
        if nd and ns:
            counts['translation_valid_pairs'] += 1
            dr, sr = '(' in d or ')' in d, '(' in s or ')' in s
            dp, sp = pairs.get('dialect_form', []), pairs.get('standard_form', [])
            counts['translation_either_with_round'] += dr or sr
            counts['translation_both_with_round'] += dr and sr
            counts['translation_either_with_pair_pattern'] += bool(dp or sp)
            counts['translation_both_with_pair_pattern'] += bool(dp and sp)
            shared = set(dp).intersection(sp)
            counts['translation_shared_pair_pattern'] += bool(shared)
            shared_different = any(a != b for a, b in shared)
            counts['translation_shared_different_pair_pattern'] += shared_different
            if shared_different:
                kinds.append('shared_different_pair_pattern')
            if dr and not dp:
                kinds.append('source_round_without_pair_pattern')
            if DOUBLE.search(d):
                kinds.append('source_double_round')
            cd, cs = clean(d), clean(s)
            counts['translation_empty_if_mlm_rule'] += not cd or not cs
            counts['translation_changed_now'] += nd != ns
            if cd and cs:
                counts['translation_changed_if_mlm_rule'] += cd != cs
                collapsed = nd != ns and cd == cs
                counts['translation_changed_to_identical_if_mlm_rule'] += collapsed
                if collapsed and (dr or sr):
                    kinds.append('changed_to_identical_with_round')
            if not cd or not cs:
                kinds.append('empty_if_mlm_rule')
    return counts, kinds


def process_file(row):
    path = Path(row['path'])
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != row['sha256']:
        raise ValueError(f'Source changed since preparation: {path}')
    data = json.loads(raw.decode('utf-8-sig'))
    counts = Counter(files=1)
    examples = {}
    for index, u in enumerate(data.get('utterance', [])):
        current, kinds = count_utterance(u)
        counts.update(current)
        for kind in kinds:
            if kind not in examples:
                examples[kind] = {
                    'path': str(path), 'utterance_index': index,
                    'utterance_id': u.get('id'),
                    'form': excerpt(u.get('form')),
                    'dialect_form': excerpt(u.get('dialect_form')),
                    'standard_form': excerpt(u.get('standard_form')),
                }
    if counts['classification_valid'] != int(row['num_sentences']):
        raise ValueError(f'Classification count differs from built manifest: {path}')
    return row['region'], row['split'], row['source_id'], counts, examples


def percent(n, denominator):
    return round(100 * n / denominator, 4) if denominator else 0.0


def run(args):
    manifest = ROOT / 'server/region_classification/data/corpus_split_manifest_80_10_10.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = json.loads((ROOT / 'local_prepare/build_report.json').read_text(encoding='utf-8'))
    assert len(rows) == expected['total_selected_files']
    report_path = Path(__file__).with_name('parenthesis_report.json')
    per_file_path = Path(__file__).with_name('per_file_counts.jsonl')
    if report_path.exists() or per_file_path.exists():
        raise FileExistsError('Census output exists; do not overwrite previous audit.')
    regions = {r: Counter() for r in REGIONS}
    splits = defaultdict(Counter)
    totals = Counter()
    examples = {r: defaultdict(list) for r in REGIONS}
    started = datetime.now(timezone.utc).isoformat()
    # Bounded in-flight work; preserve deterministic manifest order on Windows.
    with ProcessPoolExecutor(max_workers=args.workers) as pool, per_file_path.open('w', encoding='utf-8') as detail:
        iterator = iter(rows)
        pending = []
        for _ in range(args.workers * 3):
            row = next(iterator, None)
            if row is not None:
                pending.append(pool.submit(process_file, row))
        finished = 0
        while pending:
            result = pending.pop(0).result()
            region, split, source_id, counts, found = result
            regions[region].update(counts)
            splits[split].update(counts)
            totals.update(counts)
            for kind, sample in found.items():
                if len(examples[region][kind]) < 3:
                    examples[region][kind].append(sample)
            detail.write(json.dumps({'source_id': source_id, 'region': region, 'split': split, **counts}, ensure_ascii=False) + '\n')
            finished += 1
            if finished % 1000 == 0 or finished == len(rows):
                print(f"[AUDIT] {finished:,}/{len(rows):,} files; {totals['utterance_rows']:,} utterances", flush=True)
            row = next(iterator, None)
            if row is not None:
                pending.append(pool.submit(process_file, row))
    for region in REGIONS:
        assert regions[region]['utterance_rows'] == expected['raw_utterance_counts'][region], region
        assert regions[region]['files'] == expected['selected_file_counts'][region], region
    for split in ('train', 'validation', 'test'):
        assert splits[split]['classification_valid'] == expected['classification']['split_sentence_counts'][split]
        assert splits[split]['translation_valid_pairs'] == expected['translation']['pair_counts'][split]
    percentages = {}
    for region, counts in [*regions.items(), ('전체', totals)]:
        percentages[region] = {
            'dialect_with_round_pct': percent(counts['dialect_form_with_round'], counts['dialect_form_nonempty']),
            'dialect_with_pair_pattern_pct': percent(counts['dialect_form_with_pair_pattern'], counts['dialect_form_nonempty']),
            'standard_with_round_pct': percent(counts['standard_form_with_round'], counts['standard_form_nonempty']),
            'shared_different_pair_pattern_pct': percent(counts['translation_shared_different_pair_pattern'], counts['translation_valid_pairs']),
        }
    report = {
        'completed': True, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All selected base JSON files: new four regions and old Jeolla only.',
        'unit': 'Utterance rows, NOT counts of parenthesis occurrences.',
        'source_hashes_verified': True, 'built_data_counts_reconciled': True,
        'round_definition': 'Literal ASCII ( or ) in the raw field before normalization.',
        'pair_pattern': PAIR.pattern,
        'interpretation': 'A surface-pattern census; not all parentheses are erroneous. Each side of a simple pair excludes round parentheses, but the pair may occur inside a larger parenthetical span. Pair sides are compared after NFKC and whitespace normalization.',
        'totals': dict(totals), 'by_region': regions, 'by_split': splits,
        'percentages': percentages, 'examples': examples,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['원본 JSON 괄호 표기 전수 점검', '',
             '현재 재실험 대상 34,213개 JSON 전체. 방언·표준어의 원문 기준이며, 괄호 개수가 아니라 포함 발화 수.',
             '괄호 포함 자체는 오류 판정이 아니다. form은 방언·표준어를 함께 기록하는 원래 전사 필드이다.',
             '단순 (내용)/(내용) 패턴은 형태만 검출하며, 모든 유형의 의미나 정답을 자동 판정하지 않는다.', '',
             '지역 | 원본 방언 발화 | 괄호 포함 | 괄호 비율 | 단순 쌍 포함 | 양쪽 필드에 같은 서로 다른 표기 쌍 잔존']
    for region, c in [*regions.items(), ('전체', totals)]:
        lines.append(f"{region} | {c['dialect_form_nonempty']:,} | {c['dialect_form_with_round']:,} | {percentages[region]['dialect_with_round_pct']:.2f}% | {c['dialect_form_with_pair_pattern']:,} | {c['translation_shared_different_pair_pattern']:,}")
    lines += ['', '모든 원본 SHA256 및 이미 구축한 분류/번역의 분할별 입력 수와 일치 확인.',
              '상세 집계와 예시는 parenthesis_report.json, 파일별 집계는 per_file_counts.jsonl 참조.',
              '원본 JSON, 전처리 데이터, 학습 코드는 변경하지 않았음.']
    report_path.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('[OK] Full census complete: ' + str(report_path), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    run(args)
