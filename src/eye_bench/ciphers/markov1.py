from collections.abc import Sequence

import numpy as np

from ..corpus import Corpus
from ..corpus.schema import Message
from ..invariants.models import fit_markov1_probs


def _fit_initial_probs_from_corpus(
    reference: Corpus,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Fit a smoothed distribution over message-initial tokens.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    counts = np.zeros(reference.alphabet_size, dtype=np.float64)
    for msg in reference.messages:
        if not msg.symbols:
            continue
        counts[msg.symbols[0]] += 1.0

    probs = counts + alpha
    total = probs.sum()
    if total <= 0:
        raise ValueError("Cannot fit initial-token distribution.")
    return probs / total


def sample_markov1_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_probs: np.ndarray,
    trans_probs: np.ndarray,
    seed: int | None = None,
    message_prefix: str = "markov1",
) -> Corpus:
    """
    Sample a synthetic corpus from a first-order Markov token model.

    For each message:
        C_0 ~ q_init
        C_t | C_{t-1} ~ q_trans[prev, :]
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    initial_probs = np.asarray(initial_probs, dtype=np.float64)
    if initial_probs.ndim != 1 or initial_probs.shape[0] != alphabet_size:
        raise ValueError("initial_probs must be a 1D array of length alphabet_size.")
    if np.any(initial_probs < 0):
        raise ValueError("initial_probs must be nonnegative.")
    init_total = initial_probs.sum()
    if init_total <= 0:
        raise ValueError("initial_probs must sum to a positive value.")
    initial_probs = initial_probs / init_total

    trans_probs = np.asarray(trans_probs, dtype=np.float64)
    if trans_probs.ndim != 2 or trans_probs.shape != (alphabet_size, alphabet_size):
        raise ValueError(
            "trans_probs must be a square matrix with shape "
            "(alphabet_size, alphabet_size)."
        )
    if np.any(trans_probs < 0):
        raise ValueError("trans_probs must be nonnegative.")

    row_sums = trans_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every row of trans_probs must sum to a positive value.")
    trans_probs = trans_probs / row_sums

    rng = np.random.default_rng(seed)

    messages: list[Message] = []
    for i, length in enumerate(message_lengths):
        length = int(length)
        symbols = np.empty(length, dtype=np.int64)

        symbols[0] = rng.choice(alphabet_size, p=initial_probs)
        for t in range(1, length):
            prev = int(symbols[t - 1])
            symbols[t] = rng.choice(alphabet_size, p=trans_probs[prev])

        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=length,
                symbols=symbols.tolist(),
                unigram=False,
            )
        )

    return Corpus(
        alphabet_size=alphabet_size,
        messages=messages,
    )


def sample_markov1_corpus_like(
    reference: Corpus,
    *,
    alpha: float = 0.5,
    init_alpha: float = 0.5,
    seed: int | None = None,
    message_prefix: str = "markov1",
) -> Corpus:
    """
    Fit a smoothed first-order Markov model to the reference corpus, then sample
    a synthetic corpus with the same alphabet size and message lengths.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if init_alpha < 0:
        raise ValueError("init_alpha must be >= 0.")

    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    initial_probs = _fit_initial_probs_from_corpus(reference, alpha=init_alpha)
    trans_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=reference.alphabet_size,
        alpha=alpha,
    )

    return sample_markov1_corpus(
        alphabet_size=reference.alphabet_size,
        message_lengths=message_lengths,
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        seed=seed,
        message_prefix=message_prefix,
    )
