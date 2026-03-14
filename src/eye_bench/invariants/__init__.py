from .benchmark import CorpusStats, compute_corpus_stats
from .counts import context_count_dict, transition_matrix, unigram_counts
from .dependence import mutual_information, repeat_rate_by_lag, self_follow_rate
from .entropy import (
    conditional_entropy_1,
    conditional_entropy_k,
    entropy_from_counts,
    entropy_from_probs,
    unigram_entropy,
)
from .isomorphs import (
    isomorph_window_counts,
    isomorph_windows,
    pattern_signature,
    shared_isomorph_window_count,
    shared_isomorph_window_jaccard,
    top_isomorph_patterns,
)
from .models import (
    cross_entropy_from_counts,
    cross_entropy_markov1,
    cross_entropy_unigram,
    fit_markov1_probs,
    fit_unigram_probs,
    perplexity,
)
from .sections import (
    ngram_count_map,
    ngrams,
    pooled_ngram_counts,
    shared_ngram_count,
    shared_ngram_jaccard,
)

__all__ = [
    "CorpusStats",
    "compute_corpus_stats",
    "context_count_dict",
    "transition_matrix",
    "unigram_counts",
    "self_follow_rate",
    "repeat_rate_by_lag",
    "mutual_information",
    "entropy_from_counts",
    "entropy_from_probs",
    "unigram_entropy",
    "conditional_entropy_1",
    "conditional_entropy_k",
    "pattern_signature",
    "isomorph_windows",
    "isomorph_window_counts",
    "top_isomorph_patterns",
    "shared_isomorph_window_count",
    "shared_isomorph_window_jaccard",
    "ngrams",
    "ngram_count_map",
    "pooled_ngram_counts",
    "shared_ngram_count",
    "shared_ngram_jaccard",
    "fit_unigram_probs",
    "fit_markov1_probs",
    "cross_entropy_from_counts",
    "cross_entropy_unigram",
    "cross_entropy_markov1",
    "perplexity",
]
