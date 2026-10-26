#!/usr/bin/env python3
"""Tiny offline CPU fixture, not the production dialect tokenizer evaluation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pack_tokenizers import TOKENIZERS, export_archive, import_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "smoke/cpu_fixture")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Fixture output already exists; choose another --output-dir: {output}")
    output.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import BertTokenizerFast

    source = output / "source"
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "하나", "둘", "나", "##는", "."]
    for name in TOKENIZERS:
        directory = source / name / "final_model"
        directory.mkdir(parents=True)
        vocab = directory / "vocab.txt"
        vocab.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False, strip_accents=False)
        tokenizer.save_pretrained(str(directory))
        # Deliberately add a fake weight file: it must never be exported.
        (directory / "model.safetensors").write_bytes(b"not-a-real-model")
    archive = output / "fixture_tokenizers.tar.gz"
    export_archive(source, archive)
    imported = output / "imported"
    import_archive(archive, imported)
    corpus = output / "fixture_corpus.txt"
    corpus.write_text("하나 둘\n나는 하나.\n낯선 하나\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "evaluate.py"), "--corpus", str(corpus),
               "--tokenizers-root", str(imported), "--max-lines", "3", "--batch-size", "2",
               "--output-dir", str(output / "results"), "--progress-every", "1"]
    subprocess.run(command, check=True)
    report = json.loads((output / "results/tokenizer_comparison_full_results.json").read_text(encoding="utf-8"))
    assert report["completed"] and report["scope"] == "smoke"
    assert set(report["results"]) == set(TOKENIZERS)
    assert report["provenance"]["corpus"]["total_sentences"] == 3
    reference = report["results"]["dialect"]["metrics"]
    assert reference["total_tokens"] == 8
    assert reference["unk_count"] == 1
    assert reference["total_words"] == 6
    assert abs(reference["fertility"] - 8 / 6) < 1e-12
    assert abs(reference["continued_word_rate"] - 2 / 6) < 1e-12
    for name, values in report["results"].items():
        assert values["metrics"] == reference, name
        assert not (imported / name / "model.safetensors").exists()
    print("[PASS] Four tiny local tokenizer fixtures: export/import, CPU metrics, JSON/CSV/Markdown/TeX output.")
    print("[NOTE] Fixtures share a tiny WordPiece vocabulary and do not certify real KoBERT or production results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

