import json
from pathlib import Path

from eye_bench.ciphers import (
    analyze_markov2_context_sparsity,
    sample_markov2_backoff_corpus_like_with_diagnostics,
    sample_markov2_corpus_like_with_diagnostics,
)
from eye_bench.corpus import load_corpus, validate_corpus


def main() -> None:
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    sparsity = analyze_markov2_context_sparsity(corpus)

    _, markov2_diag = sample_markov2_corpus_like_with_diagnostics(corpus, seed=0)
    _, backoff_diag = sample_markov2_backoff_corpus_like_with_diagnostics(
        corpus,
        seed=0,
        lambdas=(0.70, 0.20, 0.10),
    )

    payload = {
        "context_sparsity": sparsity.to_json_dict(),
        "markov2_sampling": markov2_diag.to_json_dict(),
        "markov2_backoff_sampling": backoff_diag.to_json_dict(),
    }

    out_path = Path("data/processed/markov2_diagnostics.json")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
