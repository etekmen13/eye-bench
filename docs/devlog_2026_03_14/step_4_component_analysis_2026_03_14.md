# Step 4: Component-Wise Analysis Workflow

## Goal

The report explicitly asked for a workflow that answers:

* which components distinguish families?
* which components improve or worsen for a given family?

I implemented that as a reusable scoring-side analysis module and a persisted processed artifact.

## Code Changes

Files touched:

* `src/eye_bench/scoring/component_analysis.py`
* `src/eye_bench/scoring/__init__.py`
* `scripts/rank_baselines.py`
* `data/processed/baseline_component_analysis.json`

The analysis report includes:

* per-component generator rankings
* per-component spreads
* per-generator deltas against `markov1`
* improved and worsened component lists

## Devlog

I chose to build this on top of `BenchmarkComparison.summaries` rather than by scraping CLI output. That keeps the workflow deterministic and scriptable.

I also made the analysis write a JSON artifact instead of only printing text. The console summary is useful while iterating, but the JSON is the durable output that other scripts or later notebooks can consume.

## Result

Top discriminative components in the regenerated run:

* `mutual_info_by_lag`
* `shared_isomorph_counts_by_window`
* `exact_ngram_counts_by_n`
* `isomorph_counts_by_window`

Weak components that remain clustered:

* `shared_exact_ngram_counts_by_n`
* `shared_exact_ngram_jaccard_by_n`
* `local_alignment_ratio_by_pair`

This workflow turned an impression into an explicit artifact. The repo now has a direct way to tell whether a feature is carrying ranking signal or just adding noise.
