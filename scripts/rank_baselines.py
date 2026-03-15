from __future__ import annotations

import json
from pathlib import Path

from eye_bench.ciphers import (
    sample_iid_unigram_corpus_like,
    sample_markov1_corpus_like,
    sample_markov2_backoff_corpus_like,
    sample_markov2_corpus_like,
    sample_perm_state_corpus_like,
    sample_perm_state_v2_corpus_like,
)
from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.scoring import GeneratorSpec, benchmark_generators


def main() -> None:
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    generators = [
        GeneratorSpec(
            name="iid_unigram",
            generate=lambda ref, seed: sample_iid_unigram_corpus_like(ref, seed=seed),
        ),
        GeneratorSpec(
            name="markov1",
            generate=lambda ref, seed: sample_markov1_corpus_like(ref, seed=seed),
        ),
        GeneratorSpec(
            name="markov2",
            generate=lambda ref, seed: sample_markov2_corpus_like(ref, seed=seed),
        ),
        GeneratorSpec(
            name="markov2_backoff",
            generate=lambda ref, seed: sample_markov2_backoff_corpus_like(
                ref,
                seed=seed,
                lambdas=(0.70, 0.20, 0.10),
            ),
        ),
        GeneratorSpec(
            name="perm_state_v1",
            generate=lambda ref, seed: sample_perm_state_corpus_like(
                ref,
                num_states=4,
                seed=seed,
            ),
        ),
        GeneratorSpec(
            name="perm_state_v2",
            generate=lambda ref, seed: sample_perm_state_v2_corpus_like(
                ref,
                num_states=4,
                delta_mode="bucketed",
                num_buckets=4,
                seed=seed,
            ),
        ),
    ]

    comparison = benchmark_generators(corpus, generators, seeds=list(range(50)))

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = processed_dir / "baseline_generator_ranking.json"
    ranking_path.write_text(
        json.dumps(comparison.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    noita_stats_path = processed_dir / "noita_corpus_stats.json"
    noita_stats_path.write_text(
        json.dumps(
            comparison.reference_stats.to_json_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("True Noita corpus stats:")
    print(
        json.dumps(
            comparison.reference_stats.to_json_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    print("\nRanking:")
    for summary in comparison.summaries:
        print(
            f"{summary.generator_name:18s} "
            f"mean_total={summary.mean_total_weighted:.6f} "
            f"std={summary.std_total_weighted:.6f}"
        )

    print("\nComponent means by generator:")
    for summary in comparison.summaries:
        print(f"\n{summary.generator_name}")
        for key, value in sorted(
            summary.mean_components.items(), key=lambda kv: -kv[1]
        ):
            print(f"{key:35s} {value:.6f}")

    print(f"\nWrote ranking JSON to: {ranking_path}")
    print(f"Wrote Noita corpus stats to: {noita_stats_path}")


if __name__ == "__main__":
    main()
