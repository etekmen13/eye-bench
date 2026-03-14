import math

import numpy as np

from ._common import (
    CorpusLike,
    _normalize_messages,
    _validate_messages,
    _validate_probs,
)
from .counts import transition_matrix, unigram_counts


def fit_unigram_probs(
    data: CorpusLike,
    alphabet_size: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Smoothed unigram model q(x) using add-alpha smoothing.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    counts = unigram_counts(data, alphabet_size).astype(np.float64)
    probs = counts + alpha
    total = probs.sum()
    if total <= 0:
        raise ValueError("Cannot fit unigram model on empty data with alpha=0.")
    return probs / total


def fit_markov1_probs(
    data: CorpusLike,
    alphabet_size: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Smoothed first-order Markov transition model q(y | x).
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    counts = transition_matrix(data, alphabet_size=alphabet_size, lag=1).astype(
        np.float64
    )
    probs = counts + alpha
    row_sums = probs.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError(
            "Cannot fit first-order Markov model with zero-sum rows; "
            "use alpha > 0 or provide more data."
        )

    return probs / row_sums


def cross_entropy_from_counts(
    counts: np.ndarray,
    model_probs: np.ndarray,
) -> float:
    """
    Cross-entropy H(p, q) = -sum_x p(x) log2 q(x),
    where p is the empirical distribution from counts and q is a model.

    Returns +inf if q assigns zero probability to an event seen in counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    model_probs = _validate_probs(np.asarray(model_probs, dtype=np.float64))

    if counts.shape != model_probs.shape:
        raise ValueError("counts and model_probs must have the same shape.")

    total = counts.sum()
    if total <= 0:
        return 0.0

    mask = counts > 0
    if np.any(model_probs[mask] <= 0):
        return math.inf

    p = counts[mask] / total
    q = model_probs[mask]
    return float(-np.dot(p, np.log2(q)))


def cross_entropy_unigram(
    data: CorpusLike,
    model_probs: np.ndarray,
    alphabet_size: int,
) -> float:
    """
    Cross-entropy of the empirical unigram distribution under a model q(x).
    """
    counts = unigram_counts(data, alphabet_size)
    return cross_entropy_from_counts(counts, model_probs)


def cross_entropy_markov1(
    data: CorpusLike,
    trans_probs: np.ndarray,
) -> float:
    """
    Average negative log-likelihood per transition under a first-order Markov model:

        -(1 / N) * sum log2 q(C_t | C_{t-1})

    aggregated across messages without crossing boundaries.
    """
    messages = _normalize_messages(data)

    trans_probs = np.asarray(trans_probs, dtype=np.float64)
    if trans_probs.ndim != 2 or trans_probs.shape[0] != trans_probs.shape[1]:
        raise ValueError("trans_probs must be a square matrix.")

    alphabet_size = trans_probs.shape[0]
    _validate_messages(messages, alphabet_size)

    row_sums = trans_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each row of trans_probs must sum to a positive value.")

    q = trans_probs / row_sums

    total_nll = 0.0
    total_transitions = 0

    for msg in messages:
        if msg.size < 2:
            continue

        prev = msg[:-1]
        curr = msg[1:]
        probs = q[prev, curr]

        if np.any(probs <= 0):
            return math.inf

        total_nll += float(-np.sum(np.log2(probs)))
        total_transitions += msg.size - 1

    if total_transitions == 0:
        return 0.0

    return float(total_nll / total_transitions)


def perplexity(bits_per_symbol: float) -> float:
    """
    Convert entropy / cross-entropy in bits to perplexity.
    """
    return float(2.0**bits_per_symbol)
