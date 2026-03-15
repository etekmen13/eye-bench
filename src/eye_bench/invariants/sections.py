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


def longest_common_substring_length(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
) -> int:
    """
    Length of the longest contiguous shared substring between two sequences.
    """
    a_arr = _as_int_array(a)
    b_arr = _as_int_array(b)

    if a_arr.size == 0 or b_arr.size == 0:
        return 0

    prev = [0] * (b_arr.size + 1)
    best = 0

    for i in range(1, a_arr.size + 1):
        curr = [0] * (b_arr.size + 1)
        a_token = int(a_arr[i - 1])
        for j in range(1, b_arr.size + 1):
            if a_token == int(b_arr[j - 1]):
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr

    return best


def normalized_longest_common_substring(
    a: list[int] | tuple[int, ...],
    b: list[int] | tuple[int, ...],
) -> float:
    """
    Longest-common-substring length normalized by the shorter sequence.
    """
    a_arr = _as_int_array(a)
    b_arr = _as_int_array(b)

    denom = min(a_arr.size, b_arr.size)
    if denom == 0:
        return 0.0
    return float(longest_common_substring_length(a, b) / denom)
