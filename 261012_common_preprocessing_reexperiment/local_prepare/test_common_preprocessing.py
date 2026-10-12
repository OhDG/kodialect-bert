"""Checks for semantic preservation and exclusion boundaries of the policy."""
import copy
import unittest

from common_preprocessing import normalize_text, process_utterance


def process(source, target=None):
    return process_utterance({"dialect_form": source, "standard_form": target})


class CommonPreprocessingTests(unittest.TestCase):
    def test_inline_dialect_pair_is_resolved_without_dropping_content(self):
        result = process("신기한거(닮아)/(같아) 이", "신기한거(닮아)/(같아) #이")
        self.assertEqual(result["source_text"], "신기한거닮아 이")
        self.assertEqual(result["target_text"], "신기한거같아 #이")
        self.assertTrue(result["has_standard"])
        self.assertIsNone(result["drop_reason"])

    def test_each_field_is_resolved_independently(self):
        result = process("(집드레)/(집으로) 가라.", "집으로 (가라.)/(가더라.)")
        self.assertEqual(result["source_text"], "집드레 가라.")
        self.assertEqual(result["target_text"], "집으로 가더라.")

    def test_multiple_pairs_and_whitespace(self):
        result = process("  (집드레)\n / \t(집으로)   (가라)/(가더라)  ",
                         " (집드레)/(집으로)\n(가라)/(가더라) ")
        self.assertEqual(result["source_text"], "집드레 가라")
        self.assertEqual(result["target_text"], "집으로 가더라")
        self.assertEqual(result["stats"]["source_pairs_resolved"], 2)

    def test_identical_raw_fields_can_become_a_changed_canonical_pair(self):
        result = process("(집드레)/(집으로)", "(집드레)/(집으로)")
        self.assertFalse(result["changed_raw"])
        self.assertNotEqual(result["source_text"], result["target_text"])
        self.assertEqual(result["stats"]["canonical_pair_different"], 1)

    def test_raw_comparison_uses_only_nfkc_and_spaces(self):
        result = process("Ａ\t말", "A 말")
        self.assertFalse(result["changed_raw"])
        self.assertEqual(result["source_text"], "A 말")

    def test_uncertainty_in_either_field_excludes_the_shared_row(self):
        for source, target in (("그 ((말)) 좋아", "그 말 좋아"),
                               ("그 말 좋아", "그 (()) 좋아"),
                               ("( ( 말 ) )", "말")):
            with self.subTest(source=source, target=target):
                result = process(source, target)
                self.assertEqual(result["drop_reason"], "uncertain_transcription")
                self.assertEqual(result["source_text"], "")
                self.assertIsNone(result["target_text"])
                self.assertFalse(result["has_standard"])

    def test_missing_target_keeps_classification_and_mlm_source(self):
        for target in (None, "", " \n\t", 123):
            with self.subTest(target=target):
                result = process("(집드레)/(집으로) 가라", target)
                self.assertEqual(result["source_text"], "집드레 가라")
                self.assertFalse(result["has_standard"])
                self.assertIsNone(result["target_text"])
                self.assertIsNone(result["drop_reason"])

    def test_missing_source_drops_entire_row(self):
        for source in (None, "", "  ", 123):
            result = process(source, "표준어")
            self.assertEqual(result["drop_reason"], "missing_source")
            self.assertFalse(result["has_standard"])

    def test_ordinary_parentheses_are_preserved(self):
        for text in ("언어모델(BERT)", "계산 (x + 2) = 3", "예시 (말(소리))", "온도는 (영하 2도)"):
            with self.subTest(text=text):
                result = process(text, text)
                self.assertEqual(result["source_text"], text)
                self.assertEqual(result["target_text"], text)
                self.assertIsNone(result["drop_reason"])

    def test_non_korean_parenthesized_ratio_is_not_a_dialect_annotation(self):
        for text in ("계산 (x)/(y) = 2", "계산 x/(y) = 2", "계산 (x)/y = 2", "비율 (x)/(y)/(z)"):
            with self.subTest(text=text):
                result = process(text, text)
                self.assertEqual(result["source_text"], text)
                self.assertIsNone(result["drop_reason"])
                self.assertIn("source:non_korean_parenthesized_slash_preserved", result["flags"])

    def test_unbalanced_standalone_parenthesis_is_preserved_and_flagged(self):
        result = process("설명(미완성", "설명(미완성")
        self.assertEqual(result["source_text"], "설명(미완성")
        self.assertIsNone(result["drop_reason"])
        self.assertIn("source:unbalanced_standalone_parentheses_preserved", result["flags"])

    def test_truncated_chained_and_empty_pairs_are_not_guessed(self):
        for text in ("(방언)/(표준", "방언)/(표준어)", "(가)/(나)/(다)",
                     "(가(나))/(다)", "(방언)/()", "()/표준어",
                     "(방언)/표준어", "방언/(표준어)", "(가)/(나)/(x)"):
            with self.subTest(text=text):
                result = process(text, text)
                self.assertEqual(result["drop_reason"], "ambiguous_transcription_pair")
                self.assertEqual(result["source_text"], "")

    def test_pair_inside_outer_parentheses_is_excluded_without_guessing(self):
        result = process("(설명 (방언)/(표준어) 내용)", "(설명 (방언)/(표준어) 내용)")
        self.assertEqual(result["drop_reason"], "ambiguous_transcription_pair")
        self.assertIn("source:nested_transcription_pair", result["flags"])

    def test_fullwidth_annotation_normalizes_before_pair_selection(self):
        result = process("（집드레）／（집으로）", "（집드레）／（집으로）")
        self.assertEqual(result["source_text"], "집드레")
        self.assertEqual(result["target_text"], "집으로")

    def test_dialect_jamo_archaic_hangul_and_non_whitelist_symbols_survive(self):
        text = "ᄒᆞᆫ ㆍ ㅸ ㆆ ᄢ ᅙ ᆞ #이 @웃음 * ~ [기호] {기호} 한자漢字"
        result = process(text, text)
        self.assertEqual(result["source_text"], normalize_text(text))
        self.assertEqual(result["target_text"], normalize_text(text))
        self.assertIn("漢字", result["source_text"])
        self.assertIn("#이", result["source_text"])
        self.assertIsNone(result["drop_reason"])

    def test_archaic_jamo_in_pairs_are_recognized(self):
        result = process("(ᄒᆞᆫ)/(한)", "(ᄒᆞᆫ)/(한)")
        self.assertEqual(result["source_text"], normalize_text("ᄒᆞᆫ"))
        self.assertEqual(result["target_text"], "한")

    def test_determinism_idempotence_and_no_input_mutation(self):
        original = {"dialect_form": "  말(방언) (집드레)/(집으로) #이  ",
                    "standard_form": "말(방언) (집드레)/(집으로) #이"}
        before = copy.deepcopy(original)
        first = process_utterance(original)
        self.assertEqual(first, process_utterance(original))
        self.assertEqual(original, before)
        second = process(first["source_text"], first["target_text"])
        self.assertEqual(first["source_text"], second["source_text"])
        self.assertEqual(first["target_text"], second["target_text"])
        self.assertEqual(first["has_standard"], second["has_standard"])
        self.assertIsNone(second["drop_reason"])

    def test_return_contract_and_invalid_record(self):
        for utterance in ({"dialect_form": "안녕", "standard_form": "안녕"}, None):
            result = process_utterance(utterance)
            self.assertIsInstance(result["source_text"], str)
            self.assertTrue(result["target_text"] is None or isinstance(result["target_text"], str))
            self.assertIsInstance(result["has_standard"], bool)
            self.assertIsInstance(result["changed_raw"], bool)
            self.assertTrue(all(isinstance(k, str) and isinstance(v, int) for k, v in result["stats"].items()))
            self.assertTrue(all(isinstance(flag, str) for flag in result["flags"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
