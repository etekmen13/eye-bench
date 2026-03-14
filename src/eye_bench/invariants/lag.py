import numpy as np


def repeat_rate_by_lag(seq: np.ndarray, max_lag: int = 20) -> dict[int, float]:
    """
    Determine the average repeat rate over a sliding window (lag).
    """
    out = {}
    n = len(seq)
    for k in range(1, min(max_lag, n - 1) + 1):
        out[k] = float(np.mean(seq[:-k] == seq[k:]))

    return out
