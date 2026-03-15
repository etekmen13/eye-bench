from collections.abc import Mapping

from ..invariants.benchmark import CorpusStats


def _abs_error(a: float | int, b: float | int) -> float:
    return abs(float(a) - float(b))


def _mean_abs_error_dict(
    a: Mapping[int, float],
    b: Mapping[int, float],
) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    return float(
        sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys) / len(keys)
    )


def _normalize_count_map(counts: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(max(int(v), 0) for v in counts.values()))
    if total <= 0.0:
        return {}
    return {k: float(v) / total for k, v in counts.items() if v > 0}


def _tv_distance_count_maps(
    a: Mapping[str, int],
    b: Mapping[str, int],
) -> float:
    """
    Total variation distance between two normalized count maps.

    Returns a value in [0, 1].
    """
    pa = _normalize_count_map(a)
    pb = _normalize_count_map(b)

    keys = set(pa) | set(pb)
    if not keys:
        return 0.0

    l1 = sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)
    return 0.5 * float(l1)


def _mean_tv_distance_nested_maps(
    a: Mapping[int, Mapping[str, int]],
    b: Mapping[int, Mapping[str, int]],
) -> float:
    """
    Compare nested count maps like:
        {window: {"0,1,2,0": 14, ...}, ...}
    or:
        {n: {"east1__west2": 3, ...}, ...}
    """
    outer_keys = sorted(set(a) | set(b))
    if not outer_keys:
        return 0.0

    total = 0.0
    for k in outer_keys:
        total += _tv_distance_count_maps(a.get(k, {}), b.get(k, {}))
    return float(total / len(outer_keys))


def _count_histogram(counts: Mapping[str, int]) -> dict[str, int]:
    """
    Collapse a pair-id keyed map into a histogram over observed overlap counts.

    This makes pairwise-overlap features invariant to message-id renaming.
    """
    histogram: dict[str, int] = {}
    for value in counts.values():
        key = str(max(int(value), 0))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _mean_tv_distance_pairwise_overlap_maps(
    a: Mapping[int, Mapping[str, int]],
    b: Mapping[int, Mapping[str, int]],
) -> float:
    """
    Compare pairwise-overlap maps after discarding message-pair identifiers.

    The inner keys are corpus-specific pair ids like "east1__west2", so comparing
    them directly produces a maximal distance when two otherwise identical corpora
    use different message ids.
    """
    outer_keys = sorted(set(a) | set(b))
    if not outer_keys:
        return 0.0

    total = 0.0
    for k in outer_keys:
        total += _tv_distance_count_maps(
            _count_histogram(a.get(k, {})),
            _count_histogram(b.get(k, {})),
        )
    return float(total / len(outer_keys))


def _sorted_numeric_values(values: Mapping[str, float | int]) -> list[float]:
    return sorted(float(value) for value in values.values())


def _mean_abs_error_vectors(
    a: list[float],
    b: list[float],
) -> float:
    length = max(len(a), len(b))
    if length == 0:
        return 0.0

    a_padded = a + [0.0] * (length - len(a))
    b_padded = b + [0.0] * (length - len(b))
    return float(
        sum(abs(a_value - b_value) for a_value, b_value in zip(a_padded, b_padded))
        / length
    )


def _pairwise_value_distribution_distance(
    a: Mapping[str, float | int],
    b: Mapping[str, float | int],
) -> float:
    """
    Compare pairwise feature distributions while discarding corpus-specific pair ids.
    """
    return _mean_abs_error_vectors(
        _sorted_numeric_values(a),
        _sorted_numeric_values(b),
    )


def _mean_pairwise_value_distribution_distance_nested_maps(
    a: Mapping[int, Mapping[str, float | int]],
    b: Mapping[int, Mapping[str, float | int]],
) -> float:
    outer_keys = sorted(set(a) | set(b))
    if not outer_keys:
        return 0.0

    total = 0.0
    for k in outer_keys:
        total += _pairwise_value_distribution_distance(a.get(k, {}), b.get(k, {}))
    return float(total / len(outer_keys))


def default_feature_weights() -> dict[str, float]:
    """
    Default weights for the benchmark distance components.

    These are intentionally simple and easy to tune later.
    """
    return {
        "alphabet_size": 5.0,
        "num_messages": 5.0,
        "total_length": 5.0,
        "unigram_entropy": 1.0,
        "conditional_entropy_by_k": 1.5,
        "self_follow_rate": 2.0,
        "repeat_rate_by_lag": 2.0,
        "mutual_info_by_lag": 2.0,
        "isomorph_counts_by_window": 2.0,
        "exact_ngram_counts_by_n": 1.5,
        "shared_exact_ngram_counts_by_n": 0.25,
        "shared_isomorph_counts_by_window": 1.25,
        "shared_exact_ngram_jaccard_by_n": 0.25,
        "shared_isomorph_jaccard_by_window": 0.75,
        "local_alignment_ratio_by_pair": 0.25,
    }


def corpus_stats_distance(
    real: CorpusStats,
    synthetic: CorpusStats,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """
    Compare two CorpusStats objects and return interpretable component distances
    plus a weighted total.

    Smaller is better. Zero means exact match under this feature set.
    """
    w = dict(default_feature_weights())
    if weights is not None:
        w.update(weights)

    components: dict[str, float] = {
        "alphabet_size": _abs_error(real.alphabet_size, synthetic.alphabet_size),
        "num_messages": _abs_error(real.num_messages, synthetic.num_messages),
        "total_length": _abs_error(real.total_length, synthetic.total_length),
        "unigram_entropy": _abs_error(real.unigram_entropy, synthetic.unigram_entropy),
        "conditional_entropy_by_k": _mean_abs_error_dict(
            real.conditional_entropy_by_k,
            synthetic.conditional_entropy_by_k,
        ),
        "self_follow_rate": _abs_error(
            real.self_follow_rate, synthetic.self_follow_rate
        ),
        "repeat_rate_by_lag": _mean_abs_error_dict(
            real.repeat_rate_by_lag,
            synthetic.repeat_rate_by_lag,
        ),
        "mutual_info_by_lag": _mean_abs_error_dict(
            real.mutual_info_by_lag,
            synthetic.mutual_info_by_lag,
        ),
        "isomorph_counts_by_window": _mean_tv_distance_nested_maps(
            real.isomorph_counts_by_window,
            synthetic.isomorph_counts_by_window,
        ),
        "exact_ngram_counts_by_n": _mean_tv_distance_nested_maps(
            real.exact_ngram_counts_by_n,
            synthetic.exact_ngram_counts_by_n,
        ),
        "shared_exact_ngram_counts_by_n": _mean_tv_distance_pairwise_overlap_maps(
            real.shared_exact_ngram_counts_by_n,
            synthetic.shared_exact_ngram_counts_by_n,
        ),
        "shared_isomorph_counts_by_window": _mean_tv_distance_pairwise_overlap_maps(
            real.shared_isomorph_counts_by_window,
            synthetic.shared_isomorph_counts_by_window,
        ),
        "shared_exact_ngram_jaccard_by_n": (
            _mean_pairwise_value_distribution_distance_nested_maps(
                real.shared_exact_ngram_jaccard_by_n,
                synthetic.shared_exact_ngram_jaccard_by_n,
            )
        ),
        "shared_isomorph_jaccard_by_window": (
            _mean_pairwise_value_distribution_distance_nested_maps(
                real.shared_isomorph_jaccard_by_window,
                synthetic.shared_isomorph_jaccard_by_window,
            )
        ),
        "local_alignment_ratio_by_pair": _pairwise_value_distribution_distance(
            real.local_alignment_ratio_by_pair,
            synthetic.local_alignment_ratio_by_pair,
        ),
    }

    weighted_total = 0.0
    for key, value in components.items():
        weighted_total += w.get(key, 1.0) * value

    return {
        **components,
        "total_weighted": float(weighted_total),
    }
