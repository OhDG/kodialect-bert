import hashlib
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, set_seed


REGION_LABELS = [
    "\uac15\uc6d0\ub3c4",
    "\uacbd\uc0c1\ub3c4",
    "\uc804\ub77c\ub3c4",
    "\uc81c\uc8fc\ub3c4",
    "\ucda9\uccad\ub3c4",
]
TOKENIZER_NAMES = ["dialect", "klue", "kobert", "mbert"]
SEEDS = [13, 21, 42, 87, 100]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def configure_reproducibility(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def configure_cuda(tf32: bool = True) -> Dict[str, object]:
    info: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if not torch.cuda.is_available():
        return info
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    props = torch.cuda.get_device_properties(0)
    info.update(
        {
            "device": torch.cuda.get_device_name(0),
            "total_memory_gb": props.total_memory / (1024**3),
            "tf32": tf32,
        }
    )
    print(
        f"[INFO] CUDA: {info['device']} ({info['total_memory_gb']:.1f} GB), "
        f"TF32={tf32}, cuDNN benchmark=True"
    )
    return info


def start_process_measurement() -> Dict[str, object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return {"pid": os.getpid(), "started_perf_counter": time.perf_counter()}


def finish_process_measurement(started: Dict[str, object]) -> Dict[str, object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - float(started["started_perf_counter"])
    result: Dict[str, object] = {
        "pid": int(started["pid"]),
        "wall_seconds": wall_seconds,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "torch_peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )
    return result


def patch_legacy_tokenizer_save_vocabulary(tokenizer) -> None:
    if getattr(tokenizer, "_filename_prefix_compat", False):
        return
    import inspect

    parameters = inspect.signature(tokenizer.save_vocabulary).parameters
    if "filename_prefix" in parameters:
        tokenizer._filename_prefix_compat = True
        return
    original = tokenizer.save_vocabulary

    def compatible(_self, save_directory, filename_prefix=None):
        del filename_prefix
        return original(save_directory)

    tokenizer.save_vocabulary = MethodType(compatible, tokenizer)
    tokenizer._filename_prefix_compat = True


def load_local_tokenizer(model_dir: Path, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, trust_remote_code=True
    )
    patch_legacy_tokenizer_save_vocabulary(tokenizer)
    tokenizer.model_max_length = max_length
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer has no pad token: {model_dir}")
    return tokenizer


def save_tokenizer_compatible(tokenizer, output_dir: Path) -> None:
    patch_legacy_tokenizer_save_vocabulary(tokenizer)
    tokenizer.save_pretrained(str(output_dir))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def tokenizer_fingerprint(tokenizer) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def generation_metrics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {
            "count": 0,
            "chrf_plus_plus": 0.0,
            "sacrebleu": 0.0,
            "cer": 0.0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
        }
    try:
        import sacrebleu
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise ImportError("Install sacrebleu and rapidfuzz for translation metrics.") from exc

    preds = [str(text).strip() for text in predictions]
    refs = [str(text).strip() for text in references]
    normalized_preds = [normalize_text(text) for text in preds]
    normalized_refs = [normalize_text(text) for text in refs]
    char_errors = sum(
        Levenshtein.distance(prediction, reference)
        for prediction, reference in zip(normalized_preds, normalized_refs)
    )
    reference_chars = sum(len(reference) for reference in normalized_refs)
    return {
        "count": len(preds),
        "chrf_plus_plus": float(
            sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
        ),
        "sacrebleu": float(sacrebleu.corpus_bleu(preds, [refs], tokenize="none").score),
        "cer": float(char_errors / reference_chars) if reference_chars else 0.0,
        "exact_match": float(np.mean([p == r for p, r in zip(preds, refs)])),
        "normalized_exact_match": float(
            np.mean([p == r for p, r in zip(normalized_preds, normalized_refs)])
        ),
    }


def subset_generation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    changed: Sequence[bool],
    regions: Sequence[str],
) -> Dict[str, object]:
    overall = generation_metrics(predictions, references)
    changed_indices = [idx for idx, value in enumerate(changed) if bool(value)]
    identity_indices = [idx for idx, value in enumerate(changed) if not bool(value)]

    def select(values: Sequence[str], indices: Iterable[int]) -> List[str]:
        return [values[idx] for idx in indices]

    report: Dict[str, object] = {
        "overall": overall,
        "changed_only": generation_metrics(
            select(predictions, changed_indices), select(references, changed_indices)
        ),
        "identity_only": generation_metrics(
            select(predictions, identity_indices), select(references, identity_indices)
        ),
        "changed_fraction": len(changed_indices) / len(changed) if changed else 0.0,
        "per_region": {},
    }
    for region in REGION_LABELS:
        indices = [idx for idx, value in enumerate(regions) if value == region]
        report["per_region"][region] = generation_metrics(
            select(predictions, indices), select(references, indices)
        )
    return report


def latest_complete_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints: List[Tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / "trainer_state.json").is_file():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def enable_trusted_local_resume(checkpoint: Optional[str], output_dir: str) -> None:
    if checkpoint is None:
        return
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output_dir).resolve()
    if not checkpoint_path.is_dir() or not checkpoint_path.is_relative_to(output_path):
        raise ValueError(f"Resume checkpoint must be inside output_dir: {checkpoint_path}")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
