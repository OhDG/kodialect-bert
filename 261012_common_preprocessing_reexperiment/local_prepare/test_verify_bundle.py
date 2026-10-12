import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / 'server/verify_bundle.py'
S = importlib.util.spec_from_file_location('bundle_validator', P)
V = importlib.util.module_from_spec(S)
S.loader.exec_module(V)


class AlignmentTests(unittest.TestCase):
    def make_data(self, root, source='가자', identifier='file:2'):
        p = root / 'region_classification/data/region_classification/dialect_region_test.tsv'
        q = root / 'translation/data/translation/translation_test.tsv'
        p.parent.mkdir(parents=True)
        q.parent.mkdir(parents=True)
        for path, header, rows in [
            (p, V.CLASS_HEADER, [['안녕', '0', '강원도', 'file', 'base', 'file:0'], ['가자', '0', '강원도', 'file', 'base', 'file:2']]),
            (q, V.PARALLEL_HEADER, [[source, '가요', '강원도', 'file', 'base', '1', identifier]]),
        ]:
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(header)
                writer.writerows(rows)
        return {'has_standard': [False, True], 'changed_clean': [False, True]}

    def test_common_subset_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = self.make_data(root)
            self.assertEqual(V.validate_shared_sources(root, 'test', flags), 1)

    def test_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = self.make_data(root, source='다른 문장')
            with self.assertRaisesRegex(ValueError, 'Common source differs'):
                V.validate_shared_sources(root, 'test', flags)

    def test_unknown_utterance_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = self.make_data(root, identifier='file:9')
            with self.assertRaises(ValueError):
                V.validate_shared_sources(root, 'test', flags)

    def test_duplicate_utterance_fails(self):
        seen = {}
        V.check_utterance_id('file:2', 'file', seen, 'fixture')
        with self.assertRaises(ValueError):
            V.check_utterance_id('file:2', 'file', seen, 'fixture')


if __name__ == '__main__':
    unittest.main()
