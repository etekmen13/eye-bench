from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

from .generator_benchmark import BenchmarkComparison, GeneratorSummary


@dataclass(slots=True)
class ComponentRankingEntry:
    generator_name: str
    value: float


@dataclass(slots=True)
class ComponentSpread:
    component_name: str
    best_generator: str
    best_value: float
    worst_generator: str
    worst_value: float
    spread: float
    mean_value: float
    std_value: float


@dataclass(slots=True)
class BaselineDelta:
    generator_name: str
    baseline_name: str
    total_weighted_delta: float
    component_deltas: dict[str, float]
    improved_components: list[str]
    worsened_components: list[str]


@dataclass(slots=True)
class ComponentAnalysisReport:
    baseline_name: str
    rankings_by_component: dict[str, list[ComponentRankingEntry]]
    spreads_by_component: list[ComponentSpread]
    deltas_vs_baseline: list[BaselineDelta]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "rankings_by_component": {
                component: [asdict(entry) for entry in entries]
                for component, entries in self.rankings_by_component.items()
            },
            "spreads_by_component": [asdict(spread) for spread in self.spreads_by_component],
            "deltas_vs_baseline": [asdict(delta) for delta in self.deltas_vs_baseline],
        }


def _summary_by_name(
    summaries: list[GeneratorSummary],
) -> dict[str, GeneratorSummary]:
    return {summary.generator_name: summary for summary in summaries}


def analyze_generator_components(
    comparison: BenchmarkComparison,
    *,
    baseline_name: str = "markov1",
) -> ComponentAnalysisReport:
    summaries = comparison.summaries
    if not summaries:
        raise ValueError("comparison.summaries must not be empty.")

    by_name = _summary_by_name(summaries)
    if baseline_name not in by_name:
        raise ValueError(f"Unknown baseline_name: {baseline_name}")

    component_names = sorted(summaries[0].mean_components.keys())

    rankings_by_component: dict[str, list[ComponentRankingEntry]] = {}
    spreads_by_component: list[ComponentSpread] = []

    for component in component_names:
        entries = sorted(
            (
                ComponentRankingEntry(
                    generator_name=summary.generator_name,
                    value=float(summary.mean_components[component]),
                )
                for summary in summaries
            ),
            key=lambda entry: entry.value,
        )
        rankings_by_component[component] = entries

        values = [entry.value for entry in entries]
        spreads_by_component.append(
            ComponentSpread(
                component_name=component,
                best_generator=entries[0].generator_name,
                best_value=entries[0].value,
                worst_generator=entries[-1].generator_name,
                worst_value=entries[-1].value,
                spread=float(entries[-1].value - entries[0].value),
                mean_value=float(mean(values)),
                std_value=float(pstdev(values)) if len(values) > 1 else 0.0,
            )
        )

    spreads_by_component.sort(key=lambda item: item.spread, reverse=True)

    baseline = by_name[baseline_name]
    deltas_vs_baseline: list[BaselineDelta] = []
    for summary in summaries:
        if summary.generator_name == baseline_name:
            continue

        component_deltas = {
            component: float(summary.mean_components[component] - baseline.mean_components[component])
            for component in component_names
        }
        improved_components = sorted(
            [component for component, delta in component_deltas.items() if delta < 0.0]
        )
        worsened_components = sorted(
            [component for component, delta in component_deltas.items() if delta > 0.0]
        )

        deltas_vs_baseline.append(
            BaselineDelta(
                generator_name=summary.generator_name,
                baseline_name=baseline_name,
                total_weighted_delta=float(
                    summary.mean_total_weighted - baseline.mean_total_weighted
                ),
                component_deltas=component_deltas,
                improved_components=improved_components,
                worsened_components=worsened_components,
            )
        )

    deltas_vs_baseline.sort(key=lambda item: item.total_weighted_delta)

    return ComponentAnalysisReport(
        baseline_name=baseline_name,
        rankings_by_component=rankings_by_component,
        spreads_by_component=spreads_by_component,
        deltas_vs_baseline=deltas_vs_baseline,
    )
