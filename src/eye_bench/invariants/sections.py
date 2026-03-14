from collections import Counter

from ._common import CorpusLike, _as_int_array, _normalize_messages
from .isomorphs import isomorph_windows


def ngrams(
    seq: list[int] | tuple[int, ...],
    n: int,
) -> list[tuple[int, ...]]:
    """
    Return all contiguous n-grams from a single sequence.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    arr = _as_int_array(seq)
    length = arr.size
    if length < n:
        return []

    return [tuple(arr[i : i + n].tolist()) for i in range(length - n + 1)]


def ngram_count_map(
    seq: list[int] | tuple[int, ...],
    n: int,
) -> dict[tuple[int, ...], int]:
    """
    Count n-grams in a single sequence.
    """
    return dict(Counter(ngrams(seq, n)))


def pooled_ngram_counts(
    data: CorpusLike,
    n: int,
) -> dict[tuple[int, ...], int]:
    """
    Count n-grams pooled across one message or many messages, without crossing
    message boundaries.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    messages = _normalize_messages(data)
    counts: Counter[tuple[int, ...]] = Counter()

    for msg in messages:
        counts.update(ngrams(msg.tolist(), n))

    return dict(counts)


def shared_ngram_count(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
    n: int,
) -> int:
    """
    Number of distinct shared exact n-grams between two sequences.
    """
    a_set = set(ngrams(a, n))
    b_set = set(ngrams(b, n))
    return len(a_set & b_set)


def shared_ngram_jaccard(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
    n: int,
) -> float:
    """
    Jaccard similarity over distinct exact n-grams.
    """
    a_set = set(ngrams(a, n))
    b_set = set(ngrams(b, n))

    union = a_set | b_set
    if not union:
        return 0.0
    return float(len(a_set & b_set) / len(union))


def shared_isomorph_window_count(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
    window: int,
) -> int:
    """
    Number of distinct shared isomorph-window signatures between two sequences.
    """
    a_set = set(isomorph_windows(a, window))
    b_set = set(isomorph_windows(b, window))
    return len(a_set & b_set)


def shared_isomorph_window_jaccard(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
    window: int,
) -> float:
    """
    Jaccard similarity over distinct isomorph-window signatures.
    """
    a_set = set(isomorph_windows(a, window))
    b_set = set(isomorph_windows(b, window))

    union = a_set | b_set
    if not union:
        return 0.0
    return float(len(a_set & b_set) / len(union))
