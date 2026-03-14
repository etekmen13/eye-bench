import numpy as np

from ._common import CorpusLike, _validate_probs
from .counts import context_count_dict, unigram_counts


def entropy_from_counts(counts: np.ndarray) -> float:
    """
    Shannon entropy H(X) in bits from a histogram of counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-np.dot(p, np.log2(p)))


def entropy_from_probs(probs: np.ndarray) -> float:
    """
    Shannon entropy H(X) in bits from a probability vector.
    """
    probs = _validate_probs(probs)
    p = probs[probs > 0]
    return float(-np.dot(p, np.log2(p)))


def unigram_entropy(data: CorpusLike, alphabet_size: int) -> float:
    """
    Empirical unigram entropy H(C_t).
    """
    return entropy_from_counts(unigram_counts(data, alphabet_size))


def conditional_entropy_k(
    data: CorpusLike,
    alphabet_size: int,
    k: int,
) -> float:
    """
    Empirical conditional entropy

        H(C_t | C_{t-k}, ..., C_{t-1})

    in bits, aggregated across messages without crossing boundaries.
    """
    counts_by_context, total_transitions = context_count_dict(
        data=data,
        alphabet_size=alphabet_size,
        k=k,
    )

    if total_transitions == 0:
        return 0.0

    total = 0.0
    for counts in counts_by_context.values():
        row_total = int(counts.sum())
        if row_total == 0:
            continue
        weight = row_total / total_transitions
        total += weight * entropy_from_counts(counts)

    return float(total)


def conditional_entropy_1(data: CorpusLike, alphabet_size: int) -> float:
    """
    Convenience wrapper for H(C_t | C_{t-1}).
    """
    return conditional_entropy_k(data, alphabet_size, k=1)
