# Step 3: Classical Sanity-Check Baselines

## Goal

The report asked for one or two classical cipher families to calibrate the benchmark. I added two:

* `static_sub`
* `vigenere_like`

These are not intended as serious claims about the Noita messages. They are sanity baselines that show how the benchmark reacts to familiar structured remapping families.

## Design

Both generators use the same fitted latent Markov-1 source so the difference is isolated to the emission mechanism:

* `static_sub`: one fixed substitution permutation over the token alphabet
* `vigenere_like`: periodic additive key over the token alphabet modulo `alphabet_size`

That keeps the comparison clean:

* same latent source class
* different classical-style observation rules

## Code Changes

Files touched:

* `src/eye_bench/ciphers/classical.py`
* `src/eye_bench/ciphers/__init__.py`
* `scripts/rank_baselines.py`

## Devlog

I chose not to implement these as direct transforms of the real corpus. That would have produced toy “encrypt the dataset itself” baselines rather than synthetic families. Instead, they sample a latent source and then apply a classical-style emission rule, which keeps them comparable to the rest of the synthetic benchmark pipeline.

I also stopped at two families because that was enough to answer the calibration question:

* does the benchmark keep `markov1` as the incumbent?
* where do simple remapping families land?

## Result

The 50-seed ranking now includes:

* `static_sub` at `3.038068`
* `vigenere_like` at `3.122377`

Observed behavior:

* `static_sub` is surprisingly strong, finishing just behind `markov1`
* `vigenere_like` is competitive but worse than the top shallow and stateful families

My interpretation is that this benchmark, on the trigram-token representation, is still fairly tolerant of families that preserve shallow sequence structure while changing label semantics. That is exactly why adding these families was useful: they reveal where the current benchmark is permissive.
