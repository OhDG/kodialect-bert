import hashlib
import json
import os
import random
from types import MethodType
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, BertTokenizerFast, set_seed


REGION_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LABEL2ID = {label: idx for idx, label in enumerate(REGION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

TOKENIZER_SPECS = {
    "dialect": {"source": "./dialect_bert_tokenizer", "description": "Dialect WordPiece"},
    "klue": {"source": "klue/bert-base", "description": "KLUE-BERT tokenizer"},
    "kobert": {"source": "monologg/kobert", "description": "KoBERT tokenizer"},
    "mbert": {
        "source": "google-bert/bert-base-multilingual-cased",
        "description": "Google multilingual BERT tokenizer",
    },
}

TOKEN_DEPENDENT_STATE_KEYS = {
    "bert.embeddings.word_embeddings.weight",
    "cls.predictions.bias",
    "cls.predictions.decoder.weight",
    "cls.predictions.decoder.bias",
}


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


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

    properties = torch.cuda.get_device_properties(0)
    info.update(
        {
            "device": torch.cuda.get_device_name(0),
            "total_memory_gb": properties.total_memory / (1024**3),
            "tf32": tf32,
        }
    )
    print(
        f"[INFO] CUDA: {info['device']} ({info['total_memory_gb']:.1f} GB), "
        f"TF32={tf32}, cuDNN benchmark=True"
    )
    return info


def resolve_existing_path(primary: str, *fallbacks: str) -> Path:
    checked = []
    for raw_path in (primary, *fallbacks):
        path = Path(raw_path)
        checked.append(path)
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in checked)
    raise FileNotFoundError(f"Could not find required path. Checked:\n{candidates}")


def load_tokenizer(tokenizer_name: str, dialect_tokenizer_dir: str, max_length: int):
    if tokenizer_name not in TOKENIZER_SPECS:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    if tokenizer_name == "dialect":
        tokenizer_dir = resolve_existing_path(dialect_tokenizer_dir, "./dialect_bert_tokenizer")
        vocab_path = tokenizer_dir / "vocab.txt"
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Dialect tokenizer vocab not found: {vocab_path}")
        tokenizer = BertTokenizerFast(
            vocab_file=str(vocab_path),
            do_lower_case=False,
            strip_accents=False,
            tokenize_chinese_chars=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        source = str(tokenizer_dir)
    else:
        source = str(TOKENIZER_SPECS[tokenizer_name]["source"])
        tokenizer_kwargs = {"trust_remote_code": tokenizer_name == "kobert"}
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **tokenizer_kwargs)
        except (ImportError, ValueError, TypeError) as exc:
            print(f"[WARN] Fast tokenizer unavailable for {tokenizer_name}; using slow tokenizer: {exc}")
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **tokenizer_kwargs)

    tokenizer.model_max_length = max_length
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{tokenizer_name} tokenizer has no pad token.")
    if tokenizer.mask_token_id is None:
        raise ValueError(f"{tokenizer_name} tokenizer has no mask token.")
    patch_legacy_tokenizer_save_vocabulary(tokenizer)
    return tokenizer, source


def patch_legacy_tokenizer_save_vocabulary(tokenizer) -> None:
    """Patch legacy KoBERT before Trainer attempts checkpoint saves."""
    if getattr(tokenizer, "_filename_prefix_compat", False):
        return
    import inspect

    parameters = inspect.signature(tokenizer.save_vocabulary).parameters
    if "filename_prefix" in parameters:
        tokenizer._filename_prefix_compat = True
        return
    original_save_vocabulary = tokenizer.save_vocabulary

    def save_vocabulary_compat(_self, save_directory, filename_prefix=None):
        del filename_prefix
        return original_save_vocabulary(save_directory)

    tokenizer.save_vocabulary = MethodType(save_vocabulary_compat, tokenizer)
    tokenizer._filename_prefix_compat = True


def save_tokenizer_compatible(tokenizer, output_dir: Path) -> None:
    patch_legacy_tokenizer_save_vocabulary(tokenizer)
    tokenizer.save_pretrained(str(output_dir))


def build_small_bert_config(
    tokenizer,
    hidden_size: int = 384,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 6,
    intermediate_size: int = 1536,
    dropout: float = 0.1,
    max_position_embeddings: int = 512,
) -> BertConfig:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    return BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.pad_token_id,
    )


def _state_sha256(items: Iterable[Tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    return _state_sha256(model.state_dict().items())


def shared_backbone_sha256(model: torch.nn.Module) -> str:
    items = (
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if name not in TOKEN_DEPENDENT_STATE_KEYS
    )
    return _state_sha256(items)


def initialize_mlm_model_with_shared_backbone(config: BertConfig, seed: int) -> BertForMaskedLM:
    """Give every tokenizer the same non-vocabulary BERT initialization."""
    reference_config = BertConfig.from_dict(config.to_dict())
    reference_config.vocab_size = 32000
    reference_config.pad_token_id = 0

    configure_reproducibility(seed)
    reference_model = BertForMaskedLM(reference_config)
    reference_state = reference_model.state_dict()

    configure_reproducibility(seed)
    model = BertForMaskedLM(config)
    target_state = model.state_dict()
    shared_state = {
        name: tensor
        for name, tensor in reference_state.items()
        if name not in TOKEN_DEPENDENT_STATE_KEYS
        and name in target_state
        and target_state[name].shape == tensor.shape
    }
    missing, unexpected = model.load_state_dict(shared_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected shared initialization keys: {unexpected}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003)
    word_embeddings = torch.empty(config.vocab_size, config.hidden_size, dtype=torch.float32)
    word_embeddings.normal_(mean=0.0, std=config.initializer_range, generator=generator)
    if config.pad_token_id is not None:
        word_embeddings[config.pad_token_id].zero_()

    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight.copy_(word_embeddings)
        model.cls.predictions.bias.zero_()
    model.tie_weights()

    allowed_missing = TOKEN_DEPENDENT_STATE_KEYS
    unknown_missing = set(missing) - allowed_missing
    if unknown_missing:
        raise RuntimeError(f"Shared initialization missed non-vocabulary keys: {sorted(unknown_missing)}")
    del reference_model
    return model


def reset_classification_modules(model: torch.nn.Module, seed: int) -> None:
    """Initialize the newly added pooler and classifier identically for paired seeds."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2_000_003)
    initializer_range = float(model.config.initializer_range)

    with torch.no_grad():
        if getattr(model.bert, "pooler", None) is not None:
            pooler_weight = torch.empty_like(model.bert.pooler.dense.weight, device="cpu")
            pooler_weight.normal_(mean=0.0, std=initializer_range, generator=generator)
            model.bert.pooler.dense.weight.copy_(pooler_weight)
            model.bert.pooler.dense.bias.zero_()

        classifier_weight = torch.empty_like(model.classifier.weight, device="cpu")
        classifier_weight.normal_(mean=0.0, std=initializer_range, generator=generator)
        model.classifier.weight.copy_(classifier_weight)
        if model.classifier.bias is not None:
            model.classifier.bias.zero_()


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Tuple[Dict[str, float], Dict[str, object]]:
    num_labels = len(REGION_LABELS)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        confusion[int(true_label), int(predicted_label)] += 1

    total = int(confusion.sum())
    per_label: Dict[str, Dict[str, float]] = {}
    precision_values = []
    recall_values = []
    f1_values = []
    weighted_f1_sum = 0.0

    for idx, region in ID2LABEL.items():
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support
        per_label[region] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    metrics = {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(weighted_f1_sum / total) if total else 0.0,
    }
    report: Dict[str, object] = {
        **metrics,
        "labels": REGION_LABELS,
        "per_label": per_label,
        "confusion_matrix": confusion.tolist(),
    }
    return metrics, report


def latest_complete_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
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
    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "").lower() in {"1", "true", "yes", "y"}:
        raise RuntimeError("Unset TORCH_FORCE_WEIGHTS_ONLY_LOAD before resuming a trusted local checkpoint.")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
