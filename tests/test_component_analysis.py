from eye_bench.ciphers import (
    sample_iid_unigram_corpus_like,
    sample_markov1_corpus_like,
)
from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.scoring import GeneratorSpec, analyze_generator_components, benchmark_generators


def test_component_analysis_reports_markov1_deltas():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    comparison = benchmark_generators(
        corpus,
        [
            GeneratorSpec(
                name="markov1",
                generate=lambda ref, seed: sample_markov1_corpus_like(ref, seed=seed),
            ),
            GeneratorSpec(
                name="iid_unigram",
                generate=lambda ref, seed: sample_iid_unigram_corpus_like(ref, seed=seed),
            ),
        ],
        seeds=[0, 1],
    )
    report = analyze_generator_components(comparison, baseline_name="markov1")

    assert report.baseline_name == "markov1"
    assert report.spreads_by_component
    assert report.deltas_vs_baseline[0].generator_name == "iid_unigram"
    assert report.deltas_vs_baseline[0].total_weighted_delta >= 0.0
