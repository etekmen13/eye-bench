from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

CorpusLike = np.ndarray | Sequence[int] | Sequence[np.ndarray | Sequence[int]]


def _as_int_array(seq: np.ndarray | Sequence[int]) -> IntArray:
    arr = np.asarray(seq, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D integer sequence.")
    return arr


def _is_scalar_like(x: object) -> bool:
    return np.isscalar(x) or isinstance(x, (int, np.integer))


def _normalize_messages(data: CorpusLike) -> list[IntArray]:
    """
    Normalize input into a list of 1D integer arrays.

    Accepted forms:
        - np.array([0, 1, 2])
        - [0, 1, 2]
        - [np.array([0, 1]), np.array([2, 3])]
        - [[0, 1], [2, 3]]
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [_as_int_array(data)]
        if data.ndim == 2:
            return [_as_int_array(row) for row in data]
        raise ValueError("NumPy input must be 1D or 2D.")

    items = list(data)
    if not items:
        return []

    if _is_scalar_like(items[0]):
        return [_as_int_array(items)]

    return [_as_int_array(msg) for msg in items]


def _validate_messages(messages: Sequence[IntArray], alphabet_size: int) -> None:
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")

    for msg in messages:
        if msg.size == 0:
            continue
        if msg.min() < 0:
            raise ValueError("Sequence contains negative symbols.")
        if msg.max() >= alphabet_size:
            raise ValueError("Sequence contains symbols outside the alphabet.")


def _validate_probs(probs: np.ndarray, axis: int | None = None) -> FloatArray:
    probs = np.asarray(probs, dtype=np.float64)

    if np.any(probs < 0):
        raise ValueError("Probabilities must be nonnegative.")

    if axis is None:
        total = probs.sum()
        if total <= 0:
            raise ValueError("Probability vector sums to zero.")
        return probs / total

    totals = probs.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Probability array has a zero-sum slice.")
    return probs / totals
