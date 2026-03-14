from collections import Counter

from ._common import CorpusLike, _as_int_array, _normalize_messages


def pattern_signature(tokens: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """
    Map a token window to its first-occurrence pattern.

    Examples:
        [5, 9, 2, 5] -> (0, 1, 2, 0)
        [7, 8, 8, 7] -> (0, 1, 1, 0)
    """
    remap: dict[int, int] = {}
    nxt = 0
    out: list[int] = []

    for t in tokens:
        if t not in remap:
            remap[t] = nxt
            nxt += 1
        out.append(remap[t])

    return tuple(out)


def isomorph_windows(
    seq: list[int] | tuple[int, ...],
    window: int,
) -> list[tuple[int, ...]]:
    """
    Return all isomorph signatures for sliding windows of length `window`
    in a single sequence.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")

    arr = _as_int_array(seq)
    n = arr.size
    if n < window:
        return []

    out: list[tuple[int, ...]] = []
    for i in range(n - window + 1):
        out.append(pattern_signature(arr[i : i + window].tolist()))
    return out


def isomorph_window_counts(
    data: CorpusLike,
    window: int,
) -> dict[tuple[int, ...], int]:
    """
    Count isomorph window signatures pooled across one message or many messages,
    without crossing message boundaries.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")

    messages = _normalize_messages(data)
    counts: Counter[tuple[int, ...]] = Counter()

    for msg in messages:
        n = msg.size
        if n < window:
            continue
        for i in range(n - window + 1):
            sig = pattern_signature(msg[i : i + window].tolist())
            counts[sig] += 1

    return dict(counts)


def top_isomorph_patterns(
    data: CorpusLike,
    window: int,
    top_k: int = 20,
) -> dict[tuple[int, ...], int]:
    """
    Return the most common isomorph signatures for a given window length.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    counts = isomorph_window_counts(data, window)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return dict(items)
