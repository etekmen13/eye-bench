# Step 1: Benchmark Refinement

## Goal

The report called out weak shared-structure metrics as the most important immediate issue. The existing benchmark had two concrete problems:

1. pairwise shared-overlap features were keyed by corpus-specific message-pair IDs, which made renaming messages look maximally different
2. the pairwise exact-overlap features were still close to flat across families, so they were adding more constant offset than discrimination

## Decisions

I changed the benchmark in the core scoring path rather than in one generator. The message-ID problem was a scorer bug, not a modeling result.

I kept the old pooled count-map features, but made the pairwise features explicitly distributional:

* pairwise shared count features are compared after discarding pair IDs
* pairwise normalized features are compared as sorted value distributions
* exact-overlap features that remained effectively saturated after the rerun were downweighted in `default_feature_weights()`

I also added new benchmark components instead of only reweighting the old ones:

* `shared_exact_ngram_jaccard_by_n`
* `shared_isomorph_jaccard_by_window`
* `local_alignment_ratio_by_pair`

That gives the benchmark more than one way to evaluate cross-message overlap structure.

## Code Changes

Files touched:

* `src/eye_bench/invariants/sections.py`
* `src/eye_bench/invariants/benchmark.py`
* `src/eye_bench/invariants/__init__.py`
* `src/eye_bench/scoring/feature_distance.py`

Concrete changes:

* added longest-common-substring and normalized local-alignment helpers
* extended `CorpusStats` with normalized pairwise-overlap maps and local-alignment maps
* fixed `CorpusStats.to_json_dict()` so scalar fields survive serialization
* made pairwise scorer paths invariant to message renaming
* lowered the default weight of `shared_exact_ngram_counts_by_n` and `shared_exact_ngram_jaccard_by_n` after the new component analysis showed they still cluster tightly

## Devlog

I first reproduced the original `shared_isomorph_counts_by_window == 1.0` failure and verified the real cause: the scorer compared maps like `{window: {east1__west2: count}}` against `{window: {iid_0__iid_1: count}}` as if the pair IDs were shared support. That made any renamed corpus score as a full mismatch.

After fixing that, I reran the baselines and checked which pairwise features still had weak spread. The shared-isomorph family became useful again, but the shared exact-overlap features were still almost constant across generators. I treated that as a calibration problem rather than pretending those features were informative.

The benchmark now has three layers for pairwise structure:

* raw shared-count distributions
* Jaccard-normalized overlap distributions
* local alignment ratios

## Result

The regenerated component analysis shows:

* `shared_isomorph_counts_by_window` is now discriminative again
* `shared_isomorph_jaccard_by_window` adds smaller but real spread
* `shared_exact_*` features remain weak on this corpus and are now lightly weighted rather than over-trusted

This step did not solve every benchmark weakness, but it corrected the broken scorer and made the benchmark materially more honest.
