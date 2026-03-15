from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from ..corpus import Corpus
from ._common import CorpusLike, IntArray, _normalize_messages
from .dependence import mutual_information, repeat_rate_by_lag, self_follow_rate
from .entropy import conditional_entropy_k, unigram_entropy
from .isomorphs import isomorph_window_counts
from .sections import (
    normalized_longest_common_substring,
    pooled_ngram_counts,
    shared_isomorph_window_count,
    shared_isomorph_window_jaccard,
    shared_ngram_count,
    shared_ngram_jaccard,
)


@dataclass(slots=True)
class CorpusStats:
    alphabet_size: int
    num_messages: int
    total_length: int

    unigram_entropy: float
    conditional_entropy_by_k: dict[int, float]

    self_follow_rate: float
    repeat_rate_by_lag: dict[int, float]
    mutual_info_by_lag: dict[int, float]

    isomorph_counts_by_window: dict[int, dict[str, int]]
    exact_ngram_counts_by_n: dict[int, dict[str, int]]

    shared_exact_ngram_counts_by_n: dict[int, dict[str, int]]
    shared_isomorph_counts_by_window: dict[int, dict[str, int]]
    shared_exact_ngram_jaccard_by_n: dict[int, dict[str, float]]
    shared_isomorph_jaccard_by_window: dict[int, dict[str, float]]
    local_alignment_ratio_by_pair: dict[str, float]

    def to_json_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-friendly dictionary.
        """
        raw = asdict(self)
        raw["conditional_entropy_by_k"] = {
            str(k): v for k, v in self.conditional_entropy_by_k.items()
        }
        raw["repeat_rate_by_lag"] = {
            str(k): v for k, v in self.repeat_rate_by_lag.items()
        }
        raw["mutual_info_by_lag"] = {
            str(k): v for k, v in self.mutual_info_by_lag.items()
        }
        raw["isomorph_counts_by_window"] = {
            str(k): v for k, v in self.isomorph_counts_by_window.items()
        }
        raw["exact_ngram_counts_by_n"] = {
            str(k): v for k, v in self.exact_ngram_counts_by_n.items()
        }
        raw["shared_exact_ngram_counts_by_n"] = {
            str(k): v for k, v in self.shared_exact_ngram_counts_by_n.items()
        }
        raw["shared_isomorph_counts_by_window"] = {
            str(k): v for k, v in self.shared_isomorph_counts_by_window.items()
        }
        raw["shared_exact_ngram_jaccard_by_n"] = {
            str(k): v for k, v in self.shared_exact_ngram_jaccard_by_n.items()
        }
        raw["shared_isomorph_jaccard_by_window"] = {
            str(k): v for k, v in self.shared_isomorph_jaccard_by_window.items()
        }

        for key, value in list(raw.items()):
            if isinstance(value, dict):
                if all(str(item[0]).lstrip("-").isdigit() for item in value.items()):
                    raw[key] = dict(sorted(value.items(), key=lambda item: int(item[0])))
                else:
                    raw[key] = dict(sorted(value.items(), key=lambda item: item[0]))
        return raw


def _normalize_benchmark_input(
    data: Corpus | CorpusLike,
    alphabet_size: int | None,
) -> tuple[list[IntArray], list[str], int]:
    """
    Accept either a Corpus model or generic CorpusLike data.

    Returns:
        messages: normalized message arrays
        message_ids: ids aligned with messages
        alphabet_size: resolved alphabet size
    """
    if isinstance(data, Corpus):
        messages = [np.asarray(msg.symbols, dtype=np.int64) for msg in data.messages]
        message_ids = [msg.message_id for msg in data.messages]
        return messages, message_ids, data.alphabet_size

    if alphabet_size is None:
        raise ValueError(
            "alphabet_size must be provided when data is not a Corpus instance."
        )

    messages = _normalize_messages(data)
    message_ids = [f"msg_{i}" for i in range(len(messages))]
    return messages, message_ids, alphabet_size


def _encode_tuple_key(tokens: tuple[int, ...]) -> str:
    return ",".join(str(x) for x in tokens)


def _encode_count_map(counts: dict[tuple[int, ...], int]) -> dict[str, int]:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {_encode_tuple_key(k): v for k, v in items}


def _pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def compute_corpus_stats(
    data: Corpus | CorpusLike,
    alphabet_size: int | None = None,
    *,
    max_conditional_k: int = 4,
    max_lag: int = 20,
    isomorph_windows: range | tuple[int, ...] = tuple(range(4, 11)),
    pooled_ngram_ns: tuple[int, ...] = (2, 3, 4),
    shared_ngram_ns: tuple[int, ...] = (3, 4, 5),
    shared_isomorph_windows: tuple[int, ...] = (4, 5, 6, 7, 8),
) -> CorpusStats:
    """
    Compute pooled and pairwise benchmark statistics for a corpus of trigram-token
    messages.

    Notes:
        - All statistics operate on trigram-token sequences.
        - Message boundaries are respected.
        - `data` may be either a `Corpus` object or generic message arrays plus
          an explicit `alphabet_size`.
    """
    messages, message_ids, alphabet_size = _normalize_benchmark_input(
        data,
        alphabet_size,
    )

    total_length = int(sum(msg.size for msg in messages))
    num_messages = len(messages)

    conditional_entropy_by_k = {
        k: conditional_entropy_k(messages, alphabet_size, k=k)
        for k in range(1, max_conditional_k + 1)
    }

    repeat_by_lag = repeat_rate_by_lag(messages, max_lag=max_lag)
    mi_by_lag = {
        lag: mutual_information(messages, alphabet_size, lag=lag)
        for lag in range(1, max_lag + 1)
    }

    isomorph_counts_by_window = {
        window: _encode_count_map(isomorph_window_counts(messages, window))
        for window in isomorph_windows
    }

    exact_ngram_counts_by_n = {
        n: _encode_count_map(pooled_ngram_counts(messages, n)) for n in pooled_ngram_ns
    }

    pairwise_shared_exact: dict[int, dict[str, int]] = {n: {} for n in shared_ngram_ns}
    pairwise_shared_exact_jaccard: dict[int, dict[str, float]] = {
        n: {} for n in shared_ngram_ns
    }
    pairwise_shared_isomorph: dict[int, dict[str, int]] = {
        window: {} for window in shared_isomorph_windows
    }
    pairwise_shared_isomorph_jaccard: dict[int, dict[str, float]] = {
        window: {} for window in shared_isomorph_windows
    }
    local_alignment_ratio_by_pair: dict[str, float] = {}

    for (i, msg_a), (j, msg_b) in combinations(enumerate(messages), 2):
        id_a = message_ids[i]
        id_b = message_ids[j]
        key = _pair_key(id_a, id_b)

        a_list = msg_a.tolist()
        b_list = msg_b.tolist()

        for n in shared_ngram_ns:
            pairwise_shared_exact[n][key] = shared_ngram_count(a_list, b_list, n)
            pairwise_shared_exact_jaccard[n][key] = shared_ngram_jaccard(
                a_list,
                b_list,
                n,
            )

        for window in shared_isomorph_windows:
            pairwise_shared_isomorph[window][key] = shared_isomorph_window_count(
                a_list,
                b_list,
                window,
            )
            pairwise_shared_isomorph_jaccard[window][key] = (
                shared_isomorph_window_jaccard(
                    a_list,
                    b_list,
                    window,
                )
            )

        local_alignment_ratio_by_pair[key] = normalized_longest_common_substring(
            a_list,
            b_list,
        )

    return CorpusStats(
        alphabet_size=alphabet_size,
        num_messages=num_messages,
        total_length=total_length,
        unigram_entropy=unigram_entropy(messages, alphabet_size),
        conditional_entropy_by_k=conditional_entropy_by_k,
        self_follow_rate=self_follow_rate(messages),
        repeat_rate_by_lag=repeat_by_lag,
        mutual_info_by_lag=mi_by_lag,
        isomorph_counts_by_window=isomorph_counts_by_window,
        exact_ngram_counts_by_n=exact_ngram_counts_by_n,
        shared_exact_ngram_counts_by_n=pairwise_shared_exact,
        shared_isomorph_counts_by_window=pairwise_shared_isomorph,
        shared_exact_ngram_jaccard_by_n=pairwise_shared_exact_jaccard,
        shared_isomorph_jaccard_by_window=pairwise_shared_isomorph_jaccard,
        local_alignment_ratio_by_pair=local_alignment_ratio_by_pair,
    )
