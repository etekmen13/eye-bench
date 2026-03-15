from eye_bench.ciphers import sample_iid_unigram_corpus_like
from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.invariants import compute_corpus_stats
from eye_bench.scoring import corpus_stats_distance


def test_pairwise_overlap_distances_ignore_message_id_renames():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    renamed = load_corpus("data/processed/eyes.json")
    for i, message in enumerate(renamed.messages):
        message.message_id = f"renamed_{i}"

    reference_stats = compute_corpus_stats(corpus)
    renamed_stats = compute_corpus_stats(renamed)
    distance = corpus_stats_distance(reference_stats, renamed_stats)

    assert distance["shared_exact_ngram_counts_by_n"] == 0.0
    assert distance["shared_isomorph_counts_by_window"] == 0.0
    assert distance["shared_exact_ngram_jaccard_by_n"] == 0.0
    assert distance["shared_isomorph_jaccard_by_window"] == 0.0
    assert distance["local_alignment_ratio_by_pair"] == 0.0


def test_shared_isomorph_distance_is_not_forced_to_one_by_id_mismatch():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    reference_stats = compute_corpus_stats(corpus)
    synthetic_stats = compute_corpus_stats(sample_iid_unigram_corpus_like(corpus, seed=0))
    distance = corpus_stats_distance(reference_stats, synthetic_stats)

    assert distance["shared_isomorph_counts_by_window"] < 1.0
