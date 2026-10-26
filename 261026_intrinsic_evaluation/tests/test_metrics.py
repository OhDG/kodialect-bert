from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import (collect_token_stats, compute_basic_metrics, compute_zipf_metrics,
                     continued_word_stats, encode_batch, scan_corpus)


class FakeTokenizer:
    unk_token_id = 99

    def __init__(self):
        self.calls = []

    def __call__(self, texts, add_special_tokens, padding, truncation):
        if add_special_tokens or padding or truncation:
            raise AssertionError("Intrinsic evaluation must not add/pad/truncate tokens")
        self.calls.extend(texts)
        mapping = {"하나": [1], "둘": [2, 3], "빈": []}
        return {"input_ids": [[token for word in text.split() for token in mapping.get(word, [99])]
                              for text in texts]}


class MetricsTests(unittest.TestCase):
    def test_occurrence_weighted_metrics_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.txt"
            corpus.write_text("하나 둘\n하나 낯\n", encoding="utf-8")
            corpus_stats, words = scan_corpus(corpus)
            tokenizer = FakeTokenizer()
            stats = collect_token_stats(tokenizer, corpus, corpus_stats, batch_size=1, progress_every=0)
            basic = compute_basic_metrics(stats)
            self.assertEqual(basic["total_tokens"], 5)
            self.assertEqual(basic["fertility"], 1.25)
            self.assertEqual(basic["avg_sequence_length"], 2.5)
            self.assertEqual(basic["unk_count"], 1)
            self.assertEqual(basic["unk_rate"], 0.2)
            self.assertEqual(basic["total_chars_including_spaces"], 8)
            self.assertEqual(basic["total_chars_excluding_spaces"], 6)
            self.assertEqual(basic["chars_per_token"], 1.2)
            self.assertEqual(basic["chars_per_token_including_spaces"], 1.6)
            self.assertEqual(len(stats["token_counter"]), 4)
            tokenizer.calls.clear()
            cwr = continued_word_stats(tokenizer, words, batch_size=2, progress_every=0)
            self.assertEqual(cwr["continued_word_rate"], 0.25)
            self.assertEqual(cwr["continued_words"], 1)
            self.assertEqual(cwr["cwr_valid_words"], 4)
            self.assertEqual(len(tokenizer.calls), 3)  # Unique words, not 4 occurrences.

    def test_cwr_zero_piece_words_excluded_and_frequencies_weighted(self):
        cwr = continued_word_stats(FakeTokenizer(), Counter({"하나": 100, "둘": 2, "빈": 3, "낯": 5}),
                                   progress_every=0)
        self.assertEqual(cwr["cwr_valid_words"], 107)
        self.assertEqual(cwr["continued_words"], 2)
        self.assertEqual(cwr["zero_piece_words"], 3)
        self.assertAlmostEqual(cwr["continued_word_rate"], 2 / 107)

    def test_strict_utf8_and_smoke_prefix_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.txt"
            corpus.write_text("하나\n\n둘\n낯\n", encoding="utf-8")
            stats, _ = scan_corpus(corpus, max_lines=2)
            self.assertEqual(stats["total_sentences"], 2)
            corpus.write_bytes(b"a\n\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                scan_corpus(corpus)

    def test_batch_size_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            encode_batch(lambda *args, **kwargs: {"input_ids": []}, ["하나"])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is needed for numeric Zipf tests")
    def test_exact_inverse_rank_law(self):
        counter = Counter({i: frequency for i, frequency in enumerate([60, 30, 20, 15, 12, 10])})
        values = compute_zipf_metrics(counter)
        self.assertEqual(values["cardinality"], 6)
        self.assertEqual(values["zipf_fit_token_count"], 6)
        self.assertAlmostEqual(values["zipf_slope"], -1.0, places=12)
        self.assertLess(values["power_law_error"], 1e-12)
        expected_auc = math.log(60) * math.log(6) - math.log(6) ** 2 / 2
        self.assertAlmostEqual(values["zipf_auc"], expected_auc, places=12)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is needed for numeric Zipf tests")
    def test_natural_log_cutoff_contains_403_ranks(self):
        values = compute_zipf_metrics(Counter({i: 1000 - i for i in range(500)}))
        self.assertEqual(values["zipf_fit_token_count"], 403)


if __name__ == "__main__":
    unittest.main()

