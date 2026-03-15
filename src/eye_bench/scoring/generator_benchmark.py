from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any, Callable

from ..corpus import Corpus
from ..invariants import compute_corpus_stats
from ..invariants.benchmark import CorpusStats
from .feature_distance import corpus_stats_distance

GeneratorFn = Callable[[Corpus, int | None], Corpus]


@dataclass(slots=True)
class GeneratorSpec:
    name: str
    generate: GeneratorFn


@dataclass(slots=True)
class GeneratorRunResult:
    generator_name: str
    seed: int | None
    distance: dict[str, float]


@dataclass(slots=True)
class GeneratorSummary:
    generator_name: str
    num_runs: int
    mean_total_weighted: float
    std_total_weighted: float
    mean_components: dict[str, float]


@dataclass(slots=True)
class BenchmarkComparison:
    reference_stats: CorpusStats
    runs: list[GeneratorRunResult]
    summaries: list[GeneratorSummary]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reference_stats": self.reference_stats.to_json_dict(),
            "runs": [
                {
                    "generator_name": run.generator_name,
                    "seed": run.seed,
                    "distance": run.distance,
                }
                for run in self.runs
            ],
            "summaries": [asdict(summary) for summary in self.summaries],
        }


def _summarize_runs(runs: list[GeneratorRunResult]) -> list[GeneratorSummary]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        grouped.setdefault(run.generator_name, []).append(run.distance)

    summaries: list[GeneratorSummary] = []
    for generator_name, distances in grouped.items():
        totals = [d["total_weighted"] for d in distances]
        component_keys = sorted(distances[0].keys())

        mean_components = {
            key: mean(d[key] for d in distances) for key in component_keys
        }

        summaries.append(
            GeneratorSummary(
                generator_name=generator_name,
                num_runs=len(distances),
                mean_total_weighted=float(mean(totals)),
                std_total_weighted=float(pstdev(totals)) if len(totals) > 1 else 0.0,
                mean_components=mean_components,
            )
        )

    summaries.sort(key=lambda s: s.mean_total_weighted)
    return summaries


def benchmark_generators(
    reference: Corpus,
    generators: list[GeneratorSpec],
    *,
    seeds: list[int] | tuple[int, ...] = (0,),
    distance_weights: dict[str, float] | None = None,
    stats_kwargs: dict[str, Any] | None = None,
) -> BenchmarkComparison:
    """
    Compare and rank a list of generators against a reference corpus.

    Each generator is run once per seed. Results are summarized by mean distance
    and ranked by mean_total_weighted (smaller is better).
    """
    if not generators:
        raise ValueError("generators must not be empty.")
    if not seeds:
        raise ValueError("seeds must not be empty.")

    stats_kwargs = stats_kwargs or {}
    reference_stats = compute_corpus_stats(reference, **stats_kwargs)

    runs: list[GeneratorRunResult] = []

    for spec in generators:
        for seed in seeds:
            synthetic = spec.generate(reference, seed)
            synthetic_stats = compute_corpus_stats(synthetic, **stats_kwargs)
            distance = corpus_stats_distance(
                reference_stats,
                synthetic_stats,
                weights=distance_weights,
            )
            runs.append(
                GeneratorRunResult(
                    generator_name=spec.name,
                    seed=seed,
                    distance=distance,
                )
            )

    summaries = _summarize_runs(runs)
    return BenchmarkComparison(
        reference_stats=reference_stats,
        runs=runs,
        summaries=summaries,
    )
