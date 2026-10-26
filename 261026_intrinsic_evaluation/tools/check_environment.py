#!/usr/bin/env python3
"""Fast read-only CPU environment, corpus path and exact-tokenizer checks."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import EXPECTED_SENTENCES, load_tokenizer, validate_tokenizer_vocab
from tools.pack_tokenizers import TOKENIZERS, verify_directory


def check(corpus, tokenizers_root):
    if sys.version_info[:2] != (3, 10):
        raise ValueError(f"Python 3.10 is required, got {sys.version}")
    versions = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Package version mismatch: {name} expected {expected}, got {actual}")
        versions[name] = actual
    corpus = Path(corpus)
    if not corpus.is_file():
        raise ValueError(f"Prepared Test corpus is missing: {corpus}")
    manifest = verify_directory(tokenizers_root)
    tokenizer_reports = {}
    for name in TOKENIZERS:
        tokenizer = load_tokenizer(Path(tokenizers_root) / name)
        size = validate_tokenizer_vocab(tokenizer, name, "full")
        if tokenizer.unk_token_id is None:
            raise ValueError(f"{name}: missing unk_token_id")
        sample = tokenizer("방언 문장을 확인합니다.", add_special_tokens=False,
                           padding=False, truncation=False)["input_ids"]
        if not sample:
            raise ValueError(f"{name}: tokenizer returned no sample tokens")
        tokenizer_reports[name] = {"class": type(tokenizer).__name__, "vocab_size": size,
                                    "unk_token_id": tokenizer.unk_token_id,
                                    "sample_tokens": len(sample)}
    return {"status": "ready", "device": "CPU tokenizer evaluation; no model weights loaded",
            "python": sys.executable, "versions": versions,
            "pytorch_installed": importlib.util.find_spec("torch") is not None,
            "corpus": str(corpus.resolve()), "corpus_bytes": corpus.stat().st_size,
            "expected_test_sentences": EXPECTED_SENTENCES,
            "test_sentence_count_verified": False,
            "corpus_check": "existence only; full UTF-8/count validation occurs during --full evaluation",
            "tokenizer_files_verified": len(manifest["files"]), "tokenizers": tokenizer_reports}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizers-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check(args.corpus, args.tokenizers_root)
    except (OSError, ValueError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f"CHECK ERROR: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

