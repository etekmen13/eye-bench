import numpy as np

from ._common import CorpusLike, _normalize_messages
from .counts import transition_matrix


def self_follow_rate(data: CorpusLike) -> float:
    """
    Fraction of adjacent positions where C_t == C_{t+1}, aggregated across
    messages without crossing message boundaries.
    """
    messages = _normalize_messages(data)

    hits = 0
    total = 0
    for msg in messages:
        if msg.size < 2:
            continue
        hits += int(np.sum(msg[:-1] == msg[1:]))
        total += msg.size - 1

    if total == 0:
        return 0.0
    return float(hits / total)


def repeat_rate_by_lag(data: CorpusLike, max_lag: int = 20) -> dict[int, float]:
    """
    Repeat rate at lag k:

        P(C_t == C_{t+k})

    aggregated across messages without crossing boundaries.
    """
    if max_lag < 1:
        return {}

    messages = _normalize_messages(data)
    out: dict[int, float] = {}

    for lag in range(1, max_lag + 1):
        hits = 0
        total = 0
        for msg in messages:
            if msg.size <= lag:
                continue
            hits += int(np.sum(msg[:-lag] == msg[lag:]))
            total += msg.size - lag

        out[lag] = 0.0 if total == 0 else float(hits / total)

    return out


def mutual_information(
    data: CorpusLike,
    alphabet_size: int,
    lag: int = 1,
) -> float:
    """
    Empirical lag-k mutual information I(C_t ; C_{t-k}) in bits.
    """
    counts = transition_matrix(data, alphabet_size=alphabet_size, lag=lag)
    total = counts.sum()
    if total == 0:
        return 0.0

    pxy = counts.astype(np.float64) / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py

    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))
