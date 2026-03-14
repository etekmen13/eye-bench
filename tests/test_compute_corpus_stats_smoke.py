from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.invariants import compute_corpus_stats


def test_compute_corpus_stats_smoke():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    stats = compute_corpus_stats(corpus)

    assert stats.alphabet_size > 0
    assert stats.num_messages > 0
    assert stats.total_length > 0
    assert 1 in stats.conditional_entropy_by_k
    assert 1 in stats.repeat_rate_by_lag
    assert 1 in stats.mutual_info_by_lag
    assert 4 in stats.isomorph_counts_by_window
    assert 2 in stats.exact_ngram_counts_by_n
