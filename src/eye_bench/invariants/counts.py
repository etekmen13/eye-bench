import numpy as np

from ._common import CorpusLike, IntArray, _normalize_messages, _validate_messages


def unigram_counts(data: CorpusLike, alphabet_size: int) -> IntArray:
    """
    Aggregate unigram counts across one message or many messages.
    """
    messages = _normalize_messages(data)
    _validate_messages(messages, alphabet_size)

    counts = np.zeros(alphabet_size, dtype=np.int64)
    for msg in messages:
        if msg.size == 0:
            continue
        counts += np.bincount(msg, minlength=alphabet_size).astype(np.int64)
    return counts


def transition_matrix(
    data: CorpusLike,
    alphabet_size: int,
    lag: int = 1,
) -> IntArray:
    """
    Aggregate lag-k transition counts across messages:

        mat[a, b] = count of positions where C_t = a and C_{t+lag} = b

    Boundaries are respected automatically.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1.")

    messages = _normalize_messages(data)
    _validate_messages(messages, alphabet_size)

    mat = np.zeros((alphabet_size, alphabet_size), dtype=np.int64)
    for msg in messages:
        if msg.size <= lag:
            continue
        src = msg[:-lag]
        dst = msg[lag:]
        np.add.at(mat, (src, dst), 1)

    return mat


def context_count_dict(
    data: CorpusLike,
    alphabet_size: int,
    k: int,
) -> tuple[dict[tuple[int, ...], IntArray], int]:
    """
    Build counts for a kth-order Markov context model.

    Returns:
        counts_by_context:
            maps context tuple (length k) -> counts over next symbols
        total_transitions:
            number of observed next-symbol events across all messages
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    messages = _normalize_messages(data)
    _validate_messages(messages, alphabet_size)

    counts_by_context: dict[tuple[int, ...], IntArray] = {}
    total_transitions = 0

    for msg in messages:
        n = msg.size
        if n <= k:
            continue

        for t in range(k, n):
            context = tuple(int(x) for x in msg[t - k : t])
            nxt = int(msg[t])

            row = counts_by_context.get(context)
            if row is None:
                row = np.zeros(alphabet_size, dtype=np.int64)
                counts_by_context[context] = row

            row[nxt] += 1
            total_transitions += 1

    return counts_by_context, total_transitions
