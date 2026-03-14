import json
from pathlib import Path

from eye_bench.ciphers import sample_iid_unigram_corpus_like
from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.invariants import compute_corpus_stats
from eye_bench.scoring import corpus_stats_distance


def main() -> None:
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    real_stats = compute_corpus_stats(corpus)

    synthetic = sample_iid_unigram_corpus_like(corpus, seed=0)
    synthetic_stats = compute_corpus_stats(synthetic)

    distance = corpus_stats_distance(real_stats, synthetic_stats)

    out_path = Path("data/processed/iid_baseline_distance.json")
    out_path.write_text(
        json.dumps(distance, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(distance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
