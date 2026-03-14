import numpy as np


def self_follow_rate(seq: np.ndarray) -> float:
    if len(seq) < 2:
        return 0.0
    return float(np.mean(seq[:-1] == seq[1:]))


def transition_matrix(seq: np.ndarray, alphabet_size: int) -> np.ndarray:
    mat = np.zeros((alphabet_size, alphabet_size), dtype=np.int32)
    for a, b in zip(seq[:-1], seq[1:]):
        mat[a, b] += 1
    return mat
