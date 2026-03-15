# Step 5: Keep Markov-1 as the Baseline to Beat

## Goal

The report said new family work should now be judged against one question:

> Does this beat `markov1`, either overall or on structural components we care about?

I turned that from a verbal guideline into an explicit workflow artifact.

## Implementation

The component-analysis report is now parameterized by a named incumbent baseline. For this project, the default is `markov1`.

The regenerated outputs now include:

* `data/processed/baseline_component_analysis.json`
* console output showing `delta_total` versus `markov1`
* per-family lists of improved and worsened components versus `markov1`

## Devlog

I did not hard-code a pass/fail gate into the benchmark run itself because the ranking loop should remain usable for exploratory work. Instead, I made the “baseline to beat” comparison explicit in the analysis layer.

That keeps the core experiment loop simple while still making it impossible to avoid the main question during review.

## Result

Current 50-seed status versus `markov1`:

* no tested family beats `markov1` overall
* `static_sub` is closest at `+0.090494`
* `perm_state_v2` is the strongest stateful family in the current run at `+0.122547`
* `perm_state_v3` does not improve on `perm_state_v2`

Component-wise notes:

* `perm_state_v2` improves `mutual_info_by_lag` and `repeat_rate_by_lag`
* `perm_state_v3` only improves `repeat_rate_by_lag`
* `static_sub` only improves `conditional_entropy_by_k`, while losing on the more identity-sensitive distributional terms

This is the practical status checkpoint for future model work: the repo now automatically shows whether a new family actually clears the incumbent bar.
