from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.invariants import compute_corpus_stats


def test_corpus_stats_to_json_dict_preserves_scalars_and_stringifies_keys():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    stats = compute_corpus_stats(corpus)
    payload = stats.to_json_dict()

    assert payload["alphabet_size"] == stats.alphabet_size
    assert payload["num_messages"] == stats.num_messages
    assert payload["total_length"] == stats.total_length
    assert payload["unigram_entropy"] == stats.unigram_entropy

    assert "1" in payload["conditional_entropy_by_k"]
    assert "1" in payload["repeat_rate_by_lag"]
    assert "1" in payload["mutual_info_by_lag"]
    assert "4" in payload["isomorph_counts_by_window"]
    assert "2" in payload["exact_ngram_counts_by_n"]
    assert "3" in payload["shared_exact_ngram_counts_by_n"]
    assert "4" in payload["shared_isomorph_counts_by_window"]
    assert "3" in payload["shared_exact_ngram_jaccard_by_n"]
    assert "4" in payload["shared_isomorph_jaccard_by_window"]
    assert payload["local_alignment_ratio_by_pair"]
