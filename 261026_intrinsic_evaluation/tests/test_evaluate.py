import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import run, validate_tokenizer_vocab
from tools.pack_tokenizers import TOKENIZERS, export_archive, import_archive


class EvaluationGuardsTests(unittest.TestCase):
    def test_tiny_vocabulary_only_allowed_for_smoke(self):
        class TinyTokenizer:
            def __len__(self):
                return 10
        self.assertEqual(validate_tokenizer_vocab(TinyTokenizer(), "dialect", "smoke"), 10)
        with self.assertRaisesRegex(ValueError, "requires vocabulary size 32,000"):
            validate_tokenizer_vocab(TinyTokenizer(), "dialect", "full")

    def test_full_test_count_mismatch_stops_before_loading_tokenizers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            for name in TOKENIZERS:
                target = source / name / "final_model"
                target.mkdir(parents=True)
                (target / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "BertTokenizerFast"}), encoding="utf-8")
                (target / "vocab.txt").write_text("[UNK]\nword\n", encoding="utf-8")
            export_archive(source, directory / "tokens.tar.gz")
            import_archive(directory / "tokens.tar.gz", directory / "imported")
            corpus = directory / "corpus.txt"
            corpus.write_text("word\n", encoding="utf-8")
            args = argparse.Namespace(batch_size=2, expected_sentences=1_111_735, progress_every=0,
                                      max_lines=None, rank_log_cutoff=6.0, output_dir=directory / "results",
                                      tokenizers_root=directory / "imported", corpus=corpus)
            with self.assertRaisesRegex(ValueError, "Expected 1,111,735 Test sentences, found 1"):
                run(args)
            self.assertFalse((directory / "results").exists())


if __name__ == "__main__":
    unittest.main()

