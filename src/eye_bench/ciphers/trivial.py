from collections.abc import Sequence

import numpy as np

from ..corpus import Corpus
from ..corpus.schema import Message
from ..invariants.models import fit_unigram_probs


def sample_iid_unigram_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    probs: np.ndarray | None = None,
    seed: int | None = None,
    message_prefix: str = "iid",
) -> Corpus:
    """
    Sample an i.i.d. unigram-token corpus.

    This is the simplest possible synthetic baseline:
        C_t ~ q independently for all positions t
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    rng = np.random.default_rng(seed)

    if probs is None:
        probs = np.full(alphabet_size, 1.0 / alphabet_size, dtype=np.float64)
    else:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or probs.shape[0] != alphabet_size:
            raise ValueError("probs must be a 1D array of length alphabet_size.")
        if np.any(probs < 0):
            raise ValueError("probs must be nonnegative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("probs must sum to a positive value.")
        probs = probs / total

    messages: list[Message] = []
    for i, length in enumerate(message_lengths):
        symbols = rng.choice(alphabet_size, size=int(length), p=probs).astype(np.int64)
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=int(length),
                symbols=symbols.tolist(),
                unigram=False,
            )
        )

    return Corpus(
        alphabet_size=alphabet_size,
        messages=messages,
    )


def sample_iid_unigram_corpus_like(
    reference: Corpus,
    *,
    alpha: float = 0.5,
    seed: int | None = None,
    message_prefix: str = "iid",
) -> Corpus:
    """
    Fit a smoothed unigram model to the reference corpus, then sample an i.i.d.
    synthetic corpus with the same alphabet size and message lengths.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    probs = fit_unigram_probs(
        reference_messages,
        alphabet_size=reference.alphabet_size,
        alpha=alpha,
    )

    return sample_iid_unigram_corpus(
        alphabet_size=reference.alphabet_size,
        message_lengths=message_lengths,
        probs=probs,
        seed=seed,
        message_prefix=message_prefix,
    )
