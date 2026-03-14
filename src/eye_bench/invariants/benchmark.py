from dataclasses import dataclass
import numpy as np


@dataclass
class CorpusStats:
    # number of unique trigrams
    alphabet_size: int

    total_length: int

    unigram_entropy: float
    conditional_entropy_1: float

    self_follow_rate: float
    repeat_rate_by_lag: dict[int, float]
    mutual_info_by_lag: dict[int, float]

    top_isomorph_patterns: dict[str, int]
    ngram_counts_2: dict[str, int]
    ngram_counts_3: dict[str, int]

    shared_section_scores: dict[str, float]
