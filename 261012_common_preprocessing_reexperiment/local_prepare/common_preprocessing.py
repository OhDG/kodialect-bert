"""Conservative shared text preparation for the five-region dialect corpus.

This module never reads or modifies source files. See PREPROCESSING_POLICY.txt
for source references, interpretation limits, and deliberate exclusion rules.
"""
from collections.abc import Mapping
import re
import unicodedata

PREPROCESSING_VERSION = "common-dialect-text-v1-2026-10-12"

_SPACE = re.compile(r"\s+")
_UNCERTAIN = re.compile(r"\(\s*\(")
_FLAT_PAIR = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")
_PAIR_HINT = re.compile(r"\)\s*/|/\s*\(")
_PAIR_BOUNDARY = re.compile(r"\)\s*/\s*\(")
_EMPTY_PAIR_SIDE = re.compile(r"\(\s*\)\s*/|/\s*\(\s*\)")
_RESOLVED_CHAIN = re.compile("\x00ANNOTATION\x00" + r"\s*/\s*\(")
_KOREAN = re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\uffa0-\uffdc]")
_MARKERS = {
    "hash_marker_preserved": "#",
    "asterisk_marker_preserved": "*",
    "at_marker_preserved": "@",
    "tilde_marker_preserved": "~",
    "square_bracket_preserved": "[]",
    "brace_preserved": "{}",
    "backslash_preserved": "\\",
}


def normalize_text(text: str) -> str:
    """Apply NFKC and collapse whitespace, without a character whitelist."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _ambiguous_pair_syntax(probe: str) -> bool:
    """Reject residual annotation syntax without treating x/(y) as annotation."""
    if _PAIR_BOUNDARY.search(probe) or _EMPTY_PAIR_SIDE.search(probe) or _RESOLVED_CHAIN.search(probe):
        return True
    for match in _PAIR_HINT.finditer(probe):
        token = match.group(0)
        if token.startswith(")"):
            start = probe.rfind("(", 0, match.start())
            if start >= 0 and _KOREAN.search(probe[start + 1:match.start()]):
                return True
        else:
            start = match.end()
            end = probe.find(")", start)
            if _KOREAN.search(probe[start:end if end >= 0 else len(probe)]):
                return True
    return False


def _prepare_field(raw: str, side: str):
    """Return (text, error, stats, flags); error text is a stable reason code."""
    stats = {}
    flags = []
    text = normalize_text(raw)
    if unicodedata.normalize("NFKC", raw) != raw:
        stats[f"{side}_nfkc_changed"] = 1
    if _SPACE.sub(" ", unicodedata.normalize("NFKC", raw)).strip() != unicodedata.normalize("NFKC", raw):
        stats[f"{side}_whitespace_changed"] = 1

    uncertain_count = len(_UNCERTAIN.findall(text))
    if uncertain_count:
        stats[f"{side}_uncertain_markers"] = uncertain_count
        flags.append(f"{side}:uncertain_transcription")
        return "", "uncertain_transcription", stats, flags

    # Mask complete, flat spans only while checking for remaining pair syntax.
    # This also preserves (x)/(y), whose interpretation is not a dialect pair.
    pieces = []
    syntax_probe = []
    cursor = 0
    surrounding_depth = 0
    for match in _FLAT_PAIR.finditer(text):
        prefix = text[cursor:match.start()]
        pieces.append(prefix)
        syntax_probe.append(prefix)
        surrounding_depth += prefix.count("(") - prefix.count(")")
        left, right = match.group(1), match.group(2)
        if not left.strip() or not right.strip():
            stats[f"{side}_empty_pair_branches"] = stats.get(f"{side}_empty_pair_branches", 0) + 1
            flags.append(f"{side}:empty_pair_branch")
            return "", "ambiguous_transcription_pair", stats, flags
        if _KOREAN.search(left) or _KOREAN.search(right):
            if surrounding_depth > 0:
                stats[f"{side}_nested_pair_syntax"] = 1
                flags.append(f"{side}:nested_transcription_pair")
                return "", "ambiguous_transcription_pair", stats, flags
            pieces.append(left if side == "source" else right)
            syntax_probe.append("\x00ANNOTATION\x00")
            stats[f"{side}_pairs_resolved"] = stats.get(f"{side}_pairs_resolved", 0) + 1
        else:
            pieces.append(match.group(0))
            syntax_probe.append("\x00LITERALPAIR\x00")
            stats[f"{side}_non_korean_pairs_preserved"] = stats.get(f"{side}_non_korean_pairs_preserved", 0) + 1
            flags.append(f"{side}:non_korean_parenthesized_slash_preserved")
        cursor = match.end()
    pieces.append(text[cursor:])
    syntax_probe.append(text[cursor:])

    # A leftover pair boundary signals a truncated, nested or chained annotation.
    # Do not guess which of several alternatives belongs to the source/target.
    probe = "".join(syntax_probe)
    if _ambiguous_pair_syntax(probe):
        stats[f"{side}_unresolved_pair_syntax"] = 1
        flags.append(f"{side}:unresolved_transcription_pair")
        return "", "ambiguous_transcription_pair", stats, flags
    if _PAIR_HINT.search(probe):
        flags.append(f"{side}:non_korean_parenthesized_slash_preserved")
        stats[f"{side}_non_korean_slash_boundary_rows"] = 1

    result = normalize_text("".join(pieces))
    if "(" in result or ")" in result:
        flags.append(f"{side}:standalone_parentheses_preserved")
        stats[f"{side}_standalone_parentheses_rows"] = 1
        if not _balanced_parentheses(result):
            flags.append(f"{side}:unbalanced_standalone_parentheses_preserved")
            stats[f"{side}_unbalanced_standalone_parentheses_rows"] = 1
    for name, chars in _MARKERS.items():
        if any(char in result for char in chars):
            flags.append(f"{side}:{name}")
            stats[f"{side}_{name}"] = 1
    if any(unicodedata.category(char) == "Cf" for char in result):
        flags.append(f"{side}:format_character_preserved")
        stats[f"{side}_format_character_preserved"] = 1
    return result, None, stats, flags


def process_utterance(utterance) -> dict:
    """Prepare one source/target pair using one shared source representation.

    Return keys:
      source_text: canonical source, or empty when the entire row is excluded.
      target_text: canonical target, or None when unavailable/excluded.
      has_standard: whether the returned canonical pair has a valid target.
      changed_raw: comparison of the original NFKC/space-normalized fields;
                   False if either original field is absent/blank/non-string.
      stats: integer diagnostic counters (occurrences or row counts as named).
      flags: sorted unique diagnostic labels; flags alone do not exclude a row.
      drop_reason: None for an accepted source, otherwise a stable reason code.

    The caller computes changed_clean from source_text != target_text when
    has_standard is true. Do not reuse changed_raw as the canonical pair label.
    """
    stats = {"utterances_seen": 1}
    flags = []
    result = {
        "source_text": "", "target_text": None, "has_standard": False,
        "changed_raw": False, "stats": stats, "flags": flags,
        "drop_reason": None,
    }
    if not isinstance(utterance, Mapping):
        result["drop_reason"] = "invalid_utterance_type"
        flags.append("invalid_utterance_type")
        stats["utterances_dropped"] = 1
        return result

    raw_source = utterance.get("dialect_form")
    raw_target = utterance.get("standard_form")
    valid_source = isinstance(raw_source, str) and bool(normalize_text(raw_source))
    valid_target = isinstance(raw_target, str) and bool(normalize_text(raw_target))
    if valid_source and valid_target:
        result["changed_raw"] = normalize_text(raw_source) != normalize_text(raw_target)
        stats["original_pair_available"] = 1
        stats["original_pair_different"] = int(result["changed_raw"])
    if not valid_source:
        result["drop_reason"] = "missing_source"
        flags.append("source:missing_or_blank")
        stats["utterances_dropped"] = 1
        return result

    source, source_error, source_stats, source_flags = _prepare_field(raw_source, "source")
    stats.update(source_stats)
    flags.extend(source_flags)
    target = None
    target_error = None
    if valid_target:
        target, target_error, target_stats, target_flags = _prepare_field(raw_target, "target")
        stats.update(target_stats)
        flags.extend(target_flags)
    else:
        flags.append("target:missing_or_blank")
        stats["target_missing_rows"] = 1

    error = source_error or target_error
    if error is not None:
        result["drop_reason"] = error
        stats["utterances_dropped"] = 1
    elif not source:
        result["drop_reason"] = "empty_source_after_preprocessing"
        stats["utterances_dropped"] = 1
    else:
        result["source_text"] = source
        result["target_text"] = target or None
        result["has_standard"] = bool(target)
        stats["utterances_accepted"] = 1
        stats["canonical_pair_available"] = int(bool(target))
        stats["canonical_pair_different"] = int(bool(target) and source != target)
    result["flags"] = sorted(set(flags))
    return result
