# eye-bench: Results Analysis, `perm_state_v3` Review, and Recommended Next Steps

## Project purpose

`eye-bench` is a **benchmark-first cryptanalysis repo** for the Noita eye glyph puzzle.

The main idea is:

* represent the accepted Noita corpus as **trigram-token sequences**
* compute a structured set of benchmark statistics on the real corpus
* generate synthetic corpora from candidate model/cipher families
* rank those families by how closely they reproduce the real corpus’s structural fingerprints

This is a much stronger contribution than free-form plaintext guessing because it turns community observations into **reproducible model discrimination**.

---

## What has been established so far

## 1. The benchmark is operational

You now have a full loop:

* canonical corpus loading/validation
* feature extraction over trigram-token sequences
* family generators
* distance scoring
* repeated-seed evaluation
* ranked comparison output

That means the repo has moved beyond setup and into real experimental use.

## 2. The Noita corpus is not well explained by a memoryless source

The i.i.d. unigram baseline performs worst among the meaningful baselines.

So the corpus definitely contains:

* local dependence
* structured repetition behavior
* nontrivial sequential fingerprints

## 3. Markov-1 is the strongest shallow baseline

Over 50 seeds, `markov1` remains the best overall family among the tested baselines.

That is a credible and important result:

* shallow observed-token sequential dependence explains a lot
* but not everything

## 4. Higher-order observed-token Markov baselines are not the right frontier

The Markov-2 diagnostics showed severe sparsity:

* most observed order-2 contexts are singletons or near-singletons
* the sampler falls back on most steps
* even backoff/interpolation does not recover enough to beat Markov-1

So continuing to push pure observed-token Markov order upward is not a good use of effort.

## 5. Stateful permutation families are plausible, but not yet winning

`perm_state_v1` and `perm_state_v2` were both competitive, but neither beat `markov1`.

That means evolving remapping is not nonsense, but the currently tested versions are still too weak, too noisy, or not aligned with the benchmark.

---

## Current ranking picture

From the 50-seed runs, the ordering is:

1. `markov1`
2. `perm_state_v2`
3. `perm_state_v1`
4. `markov2_backoff`
5. `markov2`
6. `iid_unigram`

A key earlier result was that **`static_sub` came very close to `markov1`**. That was one of the most important findings in the repo so far.

The implication is:

> a large portion of the benchmark is currently **relabel-invariant** or close to it.

That is not automatically bad, but it means the benchmark is currently better at distinguishing **structural classes** than **exact surface token mechanisms**.

---

## What that means about the current benchmark

## 1. Much of the benchmark is still label-invariant

Because `static_sub` nearly tied `markov1`, a lot of the current score is effectively preserved under alphabet relabeling.

So the current benchmark is heavily rewarding:

* shape of repetition
* local sequential dependence
* overlap structure

more than:

* exact token identity reuse
* exact token-specific surface behavior

This is a central result, and it should shape the next phase of the repo.

## 2. Some benchmark components are weak discriminators

Several components either saturate or barely vary across families.

That means they are not currently pulling their weight in the total score.

This does **not** invalidate the benchmark, but it does mean the scoring should be refined before drawing stronger model-family conclusions.

## 3. `markov1` is the baseline to beat

At this point, every new family should be judged against a simple question:

> does it beat `markov1`, either overall or on the structural components we care most about?

That is the current bar.

---

# Thoughts on your `perm_state_v3` implementation

## High-level verdict

`perm_state_v3` is a reasonable attempt to increase expressivity, but it is **very likely too complex too quickly**.

Relative to v2, it adds several moving parts at once:

* block-rotation permutation family
* latent-token-driven state deltas
* emitted-token-driven state deltas
* message-specific initial states
* larger default number of states (`num_states=6`)

That means when it underperforms, it becomes hard to tell **which change hurt**.

So my main reaction is:

> the implementation is interesting, but it bundles too many interventions together to be a clean experiment.

---

## What I like about `perm_state_v3`

### 1. It is more truly stateful than v1/v2

The update rule

[
S_{t+1} = (S_t + \Delta_{\text{latent}}(Z_t) + \Delta_{\text{emitted}}(C_t)) \bmod K
]

is much more transducer-like than simple cyclic updates.

This moves closer to the kind of evolving cipher mechanism that people actually suspect.

### 2. Message-specific initial states are a good idea in principle

Allowing messages to begin in different states under a shared rule is plausible and more realistic than forcing every message to start identically.

### 3. Block-rotation permutation families are better motivated than arbitrary random permutations

This is a better direction than fully random state-specific permutations because it keeps the family structured and lower-variance.

---

## What worries me about `perm_state_v3`

### 1. Too many changes at once

This is the biggest problem.

Compared with v2, you changed:

* state update rule
* permutation family
* initial-state policy
* number of states

So if v3 underperforms, you cannot tell whether the issue is:

* too many states,
* emitted-token feedback,
* the block-rotation permutation family,
* or the message-level initial-state schedule.

That makes v3 harder to learn from.

### 2. Emitted-token feedback may inject too much chaos

Using both latent and emitted token deltas can make the state dynamics overly unstable.

The emitted token is already the result of:

* latent source
* current state
* current permutation

So feeding it back into the next state can create noisy self-reinforcing behavior that damages exact local motif matching.

That may be one reason v3 did not improve.

### 3. `num_states=6` may be too large for this corpus

Given how small the corpus is, adding more hidden states can easily increase variance without giving enough evidence to justify the extra complexity.

I would expect a smaller state set like:

* 3
* 4

to often behave better than 6 on this scale.

### 4. The initial-state schedule may be imposing structure that is not helpful

`make_cyclic_message_initial_states(...)` is clean, but it is also a pretty specific assumption.

If the real corpus does not behave like that schedule, it may just add mismatch.

### 5. There is a likely schema issue in the snippet

You construct:

```python
Message(
    message_id=...,
    length=...,
# What the results mean overall

## Strong conclusions

### 1. The repo is succeeding at its main mission

It is already uncovering important facts, especially:

* Markov-1 is the best shallow baseline
* Markov-2 is crippled by sparsity
* simple stateful remapping is competitive but not superior
* static substitution being near-tied reveals benchmark label-invariance

That is exactly the sort of thing the benchmark was meant to discover.

### 2. The bottleneck is now benchmark quality, not lack of candidate families

You already have enough families to learn from.

The biggest need now is:

* making the scoring more discriminative
* especially on exact surface/token-sensitive behavior and cross-message shared structure

### 3. Stateful models are still worth exploring

Even though v1/v2/v3 did not beat Markov-1, they remain the right conceptual direction if you want to test hypotheses about evolving alphabets / finite-state ciphers.

But the next experiments should be **more controlled**, not just more expressive.

---

# Recommended next steps

## 1. Split the benchmark into two explicit tracks

This is now the highest-value change.

Create:

### A. Relabel-invariant track

Metrics that should survive token relabeling:

* isomorph counts
* shared isomorph counts
* maybe lag repeat structure
* some dependence summaries

### B. Relabel-sensitive track

Metrics that care about exact token identity:

* exact n-gram counts
* exact repeated-section structure
* token-specific local alignment
* token-specific overlap rates

This will immediately clarify what each family is actually good at.

---

## 2. Prune or downweight weak benchmark components

Several components are currently not helping enough.

I would remove or strongly downweight any component that:

* is constant across families,
* saturates,
* or has near-zero spread.

That will make the total score more meaningful.

---

## 3. Add bootstrap confidence intervals

This is important now because some family gaps are relatively small.

You already have seed-based variability, but you should also quantify uncertainty in the scoring itself:

* bootstrap the real corpus statistics
* bootstrap pairwise-section metrics
* bootstrap total scores if possible

That will help separate real wins from noise.

---

## 4. Redesign the pairwise shared-section features

This is one of the clearest improvement areas.

The next benchmark version should emphasize features like:

* longest common substring over exact tokens
* longest common substring over isomorph windows
* nonzero overlap rate across message pairs
* stronger local alignment formulations

Your current shared exact-count features are too weak.

---

## 5. Move from “bigger v3/v4” to ablation-driven transducer design

Do **not** jump directly to `perm_state_v4` with even more moving pieces.

Instead, do controlled ablations.

For example, compare these families:

### A. v2 baseline

* cyclic rolled permutations
* token-driven latent deltas only
* fixed initial state
* 4 states

### B. v2 + message-specific initial states

Only change:

* initial state schedule

### C. v2 + block-rotation permutations

Only change:

* permutation family

### D. v2 + emitted-token delta

Only change:

* add emitted-token update

### E. v2 + more states

Only change:

* 4 → 6 states

That way you will know exactly what helps and what hurts.

This is much better than treating v3 as one monolithic new family.

---

## 6. Keep Markov-1 as the standing benchmark bar

This should remain the standard comparison family.

Every new model should answer:

* does it beat `markov1` overall?
* does it beat `markov1` on relabel-sensitive metrics?
* does it beat `markov1` on pairwise shared-section metrics?

That will keep the project focused.

---

# Recommended experimental roadmap

## Immediate

* clean up `perm_state_v3` into ablation components
* remove the `unigram=False` field if it is not in the schema
* build benchmark v2 with relabel-invariant / relabel-sensitive totals
* prune dead metrics

## Near-term

* run ablations:

  * block permutations only
  * emitted-token delta only
  * message-specific initial states only
  * state-count sweep
* add bootstrap intervals
* improve pairwise shared-section scoring

## Medium-term

* build a more principled finite-state transducer family
* possibly parameterize:

  * state update rule class
  * permutation family class
  * initial-state family class

## Long-term

* use the benchmark as a model-selection engine over a library of compact cipher/transducer families
* then use the best-performing family classes to guide actual cryptanalytic search

---

# Final summary

The repo is in a strong place.

You have already established that:

* Noita is not well modeled as i.i.d.
* Markov-1 is the best shallow baseline
* higher-order observed-token Markov models are crippled by sparsity
* stateful permutation families are plausible, but current versions do not yet beat Markov-1
* the benchmark is currently too label-invariant, as shown by static substitution nearly tying the best shallow model

Your `perm_state_v3` implementation is conceptually interesting, but as an experiment it is too bundled. The next phase should emphasize:

* **benchmark refinement**
* **controlled ablations**
* **two-track scoring (relabel-invariant vs relabel-sensitive)**
* **more informative shared-section features**

That is the path most likely to convert the current infrastructure into genuinely strong evidence about which cipher families are still viable.
