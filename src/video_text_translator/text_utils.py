"""Pure text helpers used by Detector, Tracker and Translator.

All functions here are stateless and deterministic.
"""

from __future__ import annotations

# Unicode ranges for CJK Unified Ideographs and CJK Ext A.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
)


def has_cjk(text: str) -> bool:
    """Return True if ``text`` contains at least one CJK code point.

    Used by the Detector to filter out non-Chinese OCR output.
    """
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def normalize_text(text: str) -> str:
    """Strip leading and trailing whitespace; preserves case."""
    return text.strip()


def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Implementation uses the iterative two-row algorithm; O(len(a)*len(b))
    time and O(min(len(a), len(b))) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Make `a` the shorter string to minimize allocated row size.
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    curr = [0] * (len(a) + 1)
    for i, cb in enumerate(b, 1):
        curr[0] = i
        for j, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[len(a)]


def content_similarity(a: str, b: str) -> float:
    """Normalized similarity in [0.0, 1.0] based on Levenshtein distance.

    1.0 means identical (after stripping), 0.0 means completely different.
    Two empty strings are considered identical.
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    distance = levenshtein(a_norm, b_norm)
    longest = max(len(a_norm), len(b_norm))
    return 1.0 - distance / longest
