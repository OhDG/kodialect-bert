"""CPU intrinsic metrics; Zipf definitions retain the 260630 implementation."""

from collections import Counter
import hashlib
from pathlib import Path
import time


def iter_batches(corpus_path, batch_size=1024, max_lines=None):
    batch = []
    count = 0
    # Prepared corpus already has common normalization. No additional text
    # normalization or annotation/character removal is performed here.
    with Path(corpus_path).open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            batch.append(text)
            count += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if max_lines is not None and count >= max_lines:
                break
    if batch:
        yield batch


def scan_corpus(corpus_path, max_lines=None):
    stats = {"total_sentences": 0, "total_words": 0,
             "total_chars_including_spaces": 0, "total_chars_excluding_spaces": 0}
    word_counts = Counter()
    for batch in iter_batches(corpus_path, max_lines=max_lines):
        for text in batch:
            words = text.split()
            stats["total_sentences"] += 1
            stats["total_words"] += len(words)
            stats["total_chars_including_spaces"] += len(text)
            # The common preprocessor normalizes whitespace to ASCII spaces.
            # Preserve the paper's former non-space character convention.
            stats["total_chars_excluding_spaces"] += len(text.replace(" ", ""))
            word_counts.update(words)
    stats["unique_words"] = len(word_counts)
    return stats, word_counts


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_batch(tokenizer, texts):
    encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    ids = encoded["input_ids"]
    if len(ids) != len(texts) or (ids and isinstance(ids[0], int)):
        raise ValueError("Tokenizer did not preserve the input batch size")
    return ids


def collect_token_stats(tokenizer, corpus_path, corpus_stats, batch_size=1024,
                        max_lines=None, progress_every=100000, label="tokenizer"):
    counter = Counter()
    count = total_tokens = unk_count = zero_token_sentences = 0
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        raise ValueError(f"{label}: tokenizer has no unk_token_id")
    started = time.monotonic()
    next_progress = progress_every
    for texts in iter_batches(corpus_path, batch_size, max_lines):
        ids_batch = encode_batch(tokenizer, texts)
        for ids in ids_batch:
            counter.update(ids)
            count += 1
            total_tokens += len(ids)
            unk_count += sum(token_id == unk_id for token_id in ids)
            zero_token_sentences += not ids
        if progress_every and count >= next_progress:
            seconds = max(time.monotonic() - started, 0.001)
            print(f"[{label}] sentences {count:,}/{corpus_stats['total_sentences']:,}; "
                  f"{count / seconds:,.0f} sentences/s", flush=True)
            next_progress = (count // progress_every + 1) * progress_every
    if count != corpus_stats["total_sentences"]:
        raise ValueError(f"Corpus row count changed during {label} evaluation: {count}")
    if not total_tokens:
        raise ValueError(f"{label}: no tokens were produced")
    return {**corpus_stats, "total_tokens": total_tokens, "unk_count": unk_count,
            "zero_token_sentences": zero_token_sentences, "token_counter": counter}


def continued_word_stats(tokenizer, word_counts, batch_size=1024,
                         progress_every=100000, label="tokenizer"):
    """Tokenize each unique whitespace word once and weight its length by count.

    This is the 260331 per-word CWR definition, including its exclusion of words
    for which the tokenizer yields no pieces. UNK alone is one piece.
    """
    items = list(word_counts.items())
    continued = valid = zero_piece = 0
    next_progress = progress_every
    for offset in range(0, len(items), batch_size):
        chunk = items[offset:offset + batch_size]
        encoded = encode_batch(tokenizer, [word for word, _ in chunk])
        for (_, frequency), ids in zip(chunk, encoded):
            if not ids:
                zero_piece += frequency
                continue
            valid += frequency
            if len(ids) >= 2:
                continued += frequency
        finished = offset + len(chunk)
        if progress_every and finished >= next_progress:
            print(f"[{label}] CWR unique words {finished:,}/{len(items):,}", flush=True)
            next_progress = (finished // progress_every + 1) * progress_every
    return {"continued_words": continued, "cwr_valid_words": valid,
            "zero_piece_words": zero_piece,
            "continued_word_rate": continued / valid if valid else 0.0}


def compute_basic_metrics(stats):
    """260630 ratios, extended with an explicit non-space character convention."""
    total_sentences = stats["total_sentences"]
    total_tokens = stats["total_tokens"]
    total_chars = stats["total_chars_excluding_spaces"]
    total_words = stats["total_words"]
    unk_count = stats["unk_count"]
    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_chars_including_spaces": stats["total_chars_including_spaces"],
        "total_chars_excluding_spaces": total_chars,
        "avg_sequence_length": total_tokens / total_sentences if total_sentences else 0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "tokens_per_char_including_spaces": total_tokens / stats["total_chars_including_spaces"]
        if stats["total_chars_including_spaces"] else 0,
        "chars_per_token_including_spaces": stats["total_chars_including_spaces"] / total_tokens
        if total_tokens else 0,
        "fertility": total_tokens / total_words if total_words else 0,
        "unk_count": unk_count,
        "unk_rate": unk_count / total_tokens if total_tokens else 0,
        "zero_token_sentences": stats["zero_token_sentences"],
    }


def compute_zipf_metrics(token_counter, rank_log_cutoff=6.0):
    # Copied with minimal changes from 260630_test_1/step3_test_tokenizer_final.py.
    import numpy as np
    from scipy.integrate import simpson

    if len(token_counter) == 0:
        raise ValueError("token_counter is empty")
    freqs = np.array(sorted(token_counter.values(), reverse=True), dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    mask = log_ranks <= rank_log_cutoff
    x = log_ranks[mask]
    y = log_freqs[mask]
    if len(x) < 2:
        raise ValueError("At least two tokens are required for the Zipf fit")
    cardinality = len(token_counter)
    zipf_auc = float(simpson(y, x=x))
    slope, intercept = np.polyfit(x, y, deg=1)
    slope = float(slope)
    intercept = float(intercept)
    y_hat = intercept + slope * x
    power_law_error = float(np.mean(np.abs(y - y_hat)))
    return {"cardinality": cardinality, "zipf_auc": zipf_auc,
            "zipf_slope": slope, "zipf_intercept": intercept,
            "power_law_error": power_law_error, "zipf_fit_token_count": int(len(x))}

