import numpy as np


def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def unigram_entropy(seq: np.ndarray())
