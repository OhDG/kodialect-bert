"""Make saved custom tokenizers portable using the old server's existing cache.

This module loads tokenizer code already used by the experiment. It does not
download code, load BERT weights, train a tokenizer, or edit source directories.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

from verify_inputs import InputError, TOKENIZERS, require

PROBES = [
    "오늘 날씨가 참 좋네.", "밥 묵었나?", "혼저 옵서예.", "이거 뭐여?",
    "Hello BERT 123!", "(방언)/(표준어) 공백  두 개\t그리고\n줄바꿈", "제주어 ㆍ ᄒᆞ다",
]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def tokenizer_identity(tokenizer):
    return {
        "vocabulary_sha256": digest_json(tokenizer.get_vocab()),
        "special_tokens_sha256": digest_json(tokenizer.special_tokens_map),
        "special_ids": tokenizer.all_special_ids,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "probe_ids": tokenizer(PROBES, add_special_tokens=True)["input_ids"],
    }


def portable_tokenizers(files, staging):
    """Return an overlay of archive-relative paths; preserve every original file."""
    result = dict(files)
    report = {}
    for name in TOKENIZERS:
        prefix = f"models/mlm/{name}/final_model/"
        config_path = files[prefix + "tokenizer_config.json"]
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not config.get("auto_map"):
            continue
        try:
            from transformers import AutoTokenizer
            source = config_path.parent
            print(f"[TOKENIZER] Resolving {name} from its saved config and local cache only", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True,
                trust_remote_code=True, local_files_only=True)
            before = tokenizer_identity(tokenizer)
            code_path = Path(inspect.getfile(type(tokenizer)))
            # Legacy KoBERT save_vocabulary omits the modern optional argument.
            if "filename_prefix" not in inspect.signature(tokenizer.save_vocabulary).parameters:
                original_save = tokenizer.save_vocabulary
                def compatible_save(_self, save_directory, filename_prefix=None):
                    if filename_prefix is not None:
                        raise ValueError("Filename prefixes are not supported for this export")
                    return original_save(save_directory)
                tokenizer.save_vocabulary = MethodType(compatible_save, tokenizer)
            tokenizer.register_for_auto_class("AutoTokenizer")
            destination = Path(staging) / name
            destination.mkdir(parents=True)
            tokenizer.save_pretrained(destination)
            saved = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
            refs = saved.get("auto_map", {}).get("AutoTokenizer", [])
            require(refs and all(ref is None or "--" not in ref for ref in refs),
                    f"{name}: export still references remote tokenizer code")
            restored = AutoTokenizer.from_pretrained(destination, use_fast=True,
                trust_remote_code=True, local_files_only=True)
            require(tokenizer_identity(restored) == before,
                    f"{name}: tokenizer vocabulary, special tokens or token IDs changed during export")
            exported = []
            for path in sorted(destination.iterdir()):
                require(path.is_file() and not path.is_symlink(), f"Unexpected tokenizer export path: {path}")
                result[prefix + path.name] = path
                exported.append(path.name)
            report[name] = {"source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "resolved_class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
                "resolved_code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
                "identity_verified": True, "identity": before, "exported_files": exported,
                "network_downloads": False, "source_files_modified": False}
            print(f"[TOKENIZER] {name}: bundled local code; vocabulary and token IDs verified", flush=True)
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"Cannot export {name} tokenizer from the local cache: {exc}. "
                             "Run --resolve-tokenizer-code in the old server's dialect_translate environment; "
                             "its original Hugging Face code cache must be available.") from exc
    return result, report
