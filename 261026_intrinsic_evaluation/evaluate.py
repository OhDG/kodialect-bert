#!/usr/bin/env python3
"""Evaluate four exact saved tokenizers on the common-preprocessed Test corpus."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from metrics import (collect_token_stats, compute_basic_metrics, compute_zipf_metrics,
                     continued_word_stats, file_sha256, scan_corpus)
from tools.pack_tokenizers import TOKENIZERS, verify_directory


ROOT = Path(__file__).resolve().parent
LABELS = {"dialect": "KRDiaBERT 토크나이저", "klue": "KLUE-BERT 토크나이저",
          "kobert": "KoBERT 토크나이저", "mbert": "mBERT 토크나이저"}
DEFAULT_CORPUS = ROOT.parent / "261012_common_preprocessing_reexperiment/server/region_classification/data/corpus/dialect_test_corpus.txt"
EXPECTED_SENTENCES = 1_111_735
EXPECTED_VOCAB_SIZES = {"dialect": 32000, "klue": 32000, "kobert": 8002, "mbert": 119547}


def validate_tokenizer_vocab(tokenizer, name, scope):
    size = len(tokenizer)
    if scope == "full" and size != EXPECTED_VOCAB_SIZES[name]:
        raise ValueError(f"{name}: FULL evaluation requires vocabulary size "
                         f"{EXPECTED_VOCAB_SIZES[name]:,}, found {size:,}; "
                         "tiny fixtures or another tokenizer cannot be published as production results")
    return size


def load_tokenizer(directory):
    # Set before importing Transformers; all caches remain inside this project.
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory), use_fast=True,
                                         trust_remote_code=True, local_files_only=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def save_zipf_curve(path, counter, tokenizer):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "token_id", "token", "frequency", "log_rank", "log_frequency"])
        for rank, (token_id, frequency) in enumerate(counter.most_common(), 1):
            writer.writerow([rank, token_id, tokenizer.convert_ids_to_tokens(token_id), frequency,
                             math.log(rank), math.log(frequency)])


def markdown_report(results, scope, corpus_stats):
    lines = ["# Tokenizer intrinsic evaluation", "",
             f"Scope: **{scope.upper()}**. Sentences: {corpus_stats['total_sentences']:,}.", "",
             "Percentages are marked explicitly; displayed decimals are rounded to four places.", "",
             "| Tokenizer | Fertility | CWR (%) | UNK count | UNK (%) | Chars/token | Cardinality | Power-law MAE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in TOKENIZERS:
        row = results[name]["metrics"]
        lines.append(f"| {LABELS[name]} | {row['fertility']:.4f} | {row['continued_word_rate'] * 100:.4f} | "
                     f"{row['unk_count']:,} | {row['unk_rate'] * 100:.4f} | {row['chars_per_token']:.4f} | "
                     f"{row['cardinality']:,} | {row['power_law_error']:.4f} |")
    lines.extend(["", "Character ratios exclude ASCII spaces. The including-space ratios are retained in JSON/CSV.",
                  "CWR counts whitespace words with at least two tokens, weighted by corpus occurrences.",
                  "An unknown word represented by one UNK is not a continued word; interpret CWR together with UNK.",
                  "Cardinality and Zipf metrics describe distributions and do not establish downstream superiority.",
                  "A displayed 0.0000% can reflect rounding; inspect UNK count and full-precision JSON.", "",
                  "| Tokenizer | Total tokens | Mean sequence length | Tokens/char | Zipf AUC | Zipf slope | Fit tokens |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name in TOKENIZERS:
        row = results[name]["metrics"]
        lines.append(f"| {LABELS[name]} | {row['total_tokens']:,} | {row['avg_sequence_length']:.4f} | "
                     f"{row['tokens_per_char']:.4f} | {row['zipf_auc']:.4f} | {row['zipf_slope']:.4f} | "
                     f"{row['zipf_fit_token_count']:,} |")
    if scope == "smoke":
        lines.extend(["", "**SMOKE ONLY: this is the first N nonempty rows, not the independent full Test result.**"])
    return "\n".join(lines) + "\n"


def latex_table(results, scope):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{토크나이저 내재적 평가 결과" + (" (스모크 테스트; 논문 결과로 사용하지 않음)" if scope == "smoke" else "") + "}",
             r"\label{tab:intrinsic}", r"\scriptsize",
             r"\begin{tabularx}{\textwidth}{Y *{7}{>{\centering\arraybackslash}X}}", r"\toprule",
             r"토크나이저 & Fertility $\downarrow$ & CWR (\%) $\downarrow$ & UNK (개) & UNK (\%) $\downarrow$ & Chars/Token & Cardinality & Power-law MAE \\",
             r"\midrule"]
    for name in TOKENIZERS:
        row = results[name]["metrics"]
        values = [LABELS[name], f"{row['fertility']:.4f}", f"{row['continued_word_rate'] * 100:.4f}",
                  f"{row['unk_count']:,}", f"{row['unk_rate'] * 100:.4f}", f"{row['chars_per_token']:.4f}",
                  f"{row['cardinality']:,}", f"{row['power_law_error']:.4f}"]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def run(args):
    if args.batch_size <= 0 or args.expected_sentences <= 0 or args.progress_every < 0:
        raise ValueError("Batch size/expected count must be positive; progress interval cannot be negative")
    if args.max_lines is not None and args.max_lines <= 0:
        raise ValueError("--max-lines must be positive")
    if not math.isfinite(args.rank_log_cutoff) or args.rank_log_cutoff <= 0:
        raise ValueError("--rank-log-cutoff must be finite and positive")
    scope = "smoke" if args.max_lines is not None else "full"
    output = args.output_dir or ROOT / "results" / (f"smoke_{args.max_lines}" if scope == "smoke" else "full")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new --output-dir: {output}")
    print(f"[INPUT] Verifying exact saved tokenizer files: {args.tokenizers_root}", flush=True)
    manifest = verify_directory(args.tokenizers_root)
    print(f"[INPUT] Scanning {scope.upper()} corpus with UTF-8 strict decoding: {args.corpus}", flush=True)
    corpus_hash = file_sha256(args.corpus)
    corpus_stats, word_counts = scan_corpus(args.corpus, args.max_lines)
    if not corpus_stats["total_sentences"]:
        raise ValueError("Corpus contains no nonempty sentences")
    if scope == "full" and corpus_stats["total_sentences"] != args.expected_sentences:
        raise ValueError(f"Expected {args.expected_sentences:,} Test sentences, found "
                         f"{corpus_stats['total_sentences']:,}; do not publish results from a different corpus")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"generated_at": datetime.now(timezone.utc).isoformat(), "scope": scope,
                  "python": {"executable": sys.executable, "version": sys.version},
                  "evaluation_code_sha256": {str(path.relative_to(ROOT)): file_sha256(path)
                                              for path in (ROOT / "evaluate.py", ROOT / "metrics.py",
                                                           ROOT / "tools/pack_tokenizers.py")},
                  "max_lines": args.max_lines, "expected_full_test_sentences": args.expected_sentences,
                  "corpus": {"path": str(args.corpus.resolve()), "bytes": args.corpus.stat().st_size,
                             "sha256": corpus_hash, **corpus_stats},
                  "tokenizer_files": manifest["files"], "tokenizer_manifest": str((args.tokenizers_root / "tokenizers_manifest.json").resolve()),
                  "definitions": {"normalization": "prepared text unchanged except line.strip(); empty lines skipped",
                                  "special_tokens": False, "truncation": False, "padding": False,
                                  "char_ratios": "ASCII spaces excluded; including-space ratios retained",
                                  "fertility": "whole-sentence token count / whitespace-word count",
                                  "cwr": "word-occurrence-weighted proportion with >=2 tokens; zero-piece words excluded",
                                  "unk": "input token IDs equal to tokenizer.unk_token_id",
                                  "zipf_logarithm": "natural", "rank_log_cutoff": args.rank_log_cutoff,
                                  "zipf_auc": "scipy.integrate.simpson", "power_law_error": "log-frequency residual MAE"},
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("transformers", "tokenizers", "sentencepiece", "numpy", "scipy", "pandas")}}
    write_json(output / "provenance.json", provenance)
    results = {}
    for name in TOKENIZERS:
        print(f"\n[START] {name} ({LABELS[name]}); scope={scope}", flush=True)
        started = time.monotonic()
        tokenizer = load_tokenizer(args.tokenizers_root / name)
        vocabulary_size = validate_tokenizer_vocab(tokenizer, name, scope)
        stats = collect_token_stats(tokenizer, args.corpus, corpus_stats, args.batch_size,
                                    args.max_lines, args.progress_every, name)
        cwr = continued_word_stats(tokenizer, word_counts, args.batch_size, args.progress_every, name)
        values = {**compute_basic_metrics(stats), **cwr,
                  **compute_zipf_metrics(stats["token_counter"], args.rank_log_cutoff)}
        results[name] = {"label": LABELS[name], "tokenizer_class": type(tokenizer).__name__,
                         "vocab_size": vocabulary_size, "unk_token_id": tokenizer.unk_token_id,
                         "metrics": values, "elapsed_seconds": time.monotonic() - started,
                         "top_50_tokens": [{"id": token_id, "token": tokenizer.convert_ids_to_tokens(token_id),
                                            "count": frequency}
                                           for token_id, frequency in stats["token_counter"].most_common(50)]}
        save_zipf_curve(output / f"{name}_zipf_curve.csv", stats["token_counter"], tokenizer)
        write_json(output / f"{name}_metrics.json", results[name])
        print(f"[DONE] {name}: Fertility={values['fertility']:.4f}, "
              f"CWR={values['continued_word_rate'] * 100:.4f}%, "
              f"UNK={values['unk_count']:,} ({values['unk_rate'] * 100:.4f}%), "
              f"elapsed={results[name]['elapsed_seconds'] / 60:.1f} min", flush=True)
        del tokenizer, stats
    # Avoid quietly evaluating a corpus or tokenizer tree changed mid-run.
    if file_sha256(args.corpus) != corpus_hash:
        raise ValueError("Corpus changed during evaluation")
    verify_directory(args.tokenizers_root)
    payload = {"completed": True, "scope": scope, "provenance": provenance, "results": results}
    write_json(output / "tokenizer_comparison_full_results.json", payload)
    columns = list(results[TOKENIZERS[0]]["metrics"])
    with (output / "tokenizer_comparison_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tokenizer", "scope", *columns])
        writer.writeheader()
        for name in TOKENIZERS:
            writer.writerow({"tokenizer": name, "scope": scope, **results[name]["metrics"]})
    (output / "intrinsic_results.md").write_text(markdown_report(results, scope, corpus_stats), encoding="utf-8")
    (output / "table_03_intrinsic.tex").write_text(latex_table(results, scope), encoding="utf-8")
    print(f"\n[OK] {scope.upper()} results saved: {output.resolve()}", flush=True)
    return payload


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizers-root", type=Path, default=ROOT / "inputs/tokenizers")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-lines", type=int, help="SMOKE ONLY: first N nonempty rows")
    parser.add_argument("--expected-sentences", type=int, default=EXPECTED_SENTENCES)
    parser.add_argument("--rank-log-cutoff", type=float, default=6.0)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f"EVALUATION ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

