from pathlib import Path
import json

from eye_bench.corpus import load_corpus, validate_corpus
from eye_bench.invariants import compute_corpus_stats


def main() -> None:
    corpus_path = Path("data/processed/eyes.json")
    output_path = Path("data/processed/baseline_stats.json")

    corpus = load_corpus(corpus_path)
    validate_corpus(corpus)

    stats = compute_corpus_stats(corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote baseline stats to {output_path}")


if __name__ == "__main__":
    main()
