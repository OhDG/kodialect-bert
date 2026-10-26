import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pack_tokenizers import TOKENIZERS, verify_directory
from tools.prepare_server_inputs import find_code, prepare_inputs


class ServerInputTests(unittest.TestCase):
    def make_sources(self, directory):
        for name in TOKENIZERS:
            output = directory / name / 'final_model'
            output.mkdir(parents=True)
            (output / 'tokenizer_config.json').write_text(
                json.dumps({'tokenizer_class': 'BertTokenizer'}), encoding='utf-8')
            (output / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n하나\n', encoding='utf-8')
            (output / 'model.safetensors').write_bytes(b'model-weights-must-not-be-read-or-copied')

    def test_primary_sources_are_preserved_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source', Path(temporary) / 'output'
            self.make_sources(source)
            report = prepare_inputs(source, output)
            self.assertEqual(report['status'], 'prepared')
            manifest = verify_directory(output)
            self.assertEqual(len(manifest['files']), 8)
            for name in TOKENIZERS:
                self.assertEqual((output / name / 'vocab.txt').read_bytes(),
                                 (source / name / 'final_model/vocab.txt').read_bytes())
                self.assertFalse((output / name / 'model.safetensors').exists())
            self.assertEqual(prepare_inputs(source, output)['status'], 'already_verified')
            (source / 'dialect/final_model/vocab.txt').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                prepare_inputs(source, output)

    def test_remote_routing_localized_using_existing_code_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output, cache = [Path(temporary) / name for name in ['source', 'output', 'cache']]
            self.make_sources(source)
            cache.mkdir()
            (cache / 'tokenization_fixture.py').write_text(
                'from .helper import helper\nclass FixtureTokenizer: pass\n', encoding='utf-8')
            (cache / 'helper.py').write_text('helper = True\n', encoding='utf-8')
            config = {'tokenizer_class': 'FixtureTokenizer', 'do_lower_case': False,
                      'auto_map': {'AutoTokenizer': ['example/repo--tokenization_fixture.FixtureTokenizer', None]}}
            original = source / 'kobert/final_model/tokenizer_config.json'
            original.write_text(json.dumps(config), encoding='utf-8')
            prepare_inputs(source, output, [cache])
            self.assertEqual(json.loads(original.read_text(encoding='utf-8')), config)
            copied = json.loads((output / 'kobert/tokenizer_config.json').read_text(encoding='utf-8'))
            self.assertEqual(copied['auto_map']['AutoTokenizer'], ['tokenization_fixture.FixtureTokenizer', None])
            self.assertIs(copied['do_lower_case'], False)
            self.assertTrue((output / 'kobert/helper.py').is_file())
            self.assertTrue((output / 'kobert/tokenization_fixture.py').is_file())

    def test_ambiguous_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, text in [('first', 'v=1'), ('second', 'v=2')]:
                (root / name).mkdir()
                (root / name / 'tokenization_fixture.py').write_text(text, encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Multiple different'):
                find_code('tokenization_fixture.py', root / 'absent', [root])

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Tokenizer runtime is required')
    def test_localized_custom_code_loads_and_preserves_token_ids(self):
        from evaluate import load_tokenizer
        with tempfile.TemporaryDirectory() as temporary:
            source, output, cache = [Path(temporary) / name for name in ['source', 'output', 'cache']]
            self.make_sources(source)
            cache.mkdir()
            (cache / 'tokenization_fixture.py').write_text(
                'from transformers import BertTokenizer\nclass FixtureTokenizer(BertTokenizer):\n    pass\n',
                encoding='utf-8')
            config = {'tokenizer_class': 'FixtureTokenizer', 'do_lower_case': False, 'strip_accents': False,
                      'auto_map': {'AutoTokenizer': ['example/repo--tokenization_fixture.FixtureTokenizer', None]}}
            (source / 'kobert/final_model/tokenizer_config.json').write_text(json.dumps(config), encoding='utf-8')
            prepare_inputs(source, output, [cache])
            tokenizer = load_tokenizer(output / 'kobert')
            from transformers import BertTokenizer
            reference = BertTokenizer(vocab_file=str(source / 'kobert/final_model/vocab.txt'),
                                      do_lower_case=False, strip_accents=False)
            probes = ['하나 하나', 'unknown', '하나.']
            self.assertEqual(tokenizer(probes, add_special_tokens=False)['input_ids'],
                             reference(probes, add_special_tokens=False)['input_ids'])
            self.assertEqual(tokenizer.get_vocab(), reference.get_vocab())


if __name__ == '__main__':
    unittest.main()
