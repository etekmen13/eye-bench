from collections.abc import Sequence

import numpy as np

from ..corpus import Corpus
from ..corpus.schema import Message
from ..invariants.models import fit_markov1_probs


def _fit_initial_probs_from_corpus(
    reference: Corpus,
    alpha: float = 0.5,
) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    counts = np.zeros(reference.alphabet_size, dtype=np.float64)
    for msg in reference.messages:
        if msg.length >= 1:
            counts[msg.symbols[0]] += 1.0

    probs = counts + alpha
    total = probs.sum()
    if total <= 0:
        raise ValueError("Cannot fit initial probabilities from empty counts.")
    return probs / total


def _sample_latent_markov1_messages(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_probs: np.ndarray,
    trans_probs: np.ndarray,
    seed: int | None = None,
) -> list[np.ndarray]:
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    initial_probs = np.asarray(initial_probs, dtype=np.float64)
    trans_probs = np.asarray(trans_probs, dtype=np.float64)

    rng = np.random.default_rng(seed)
    messages: list[np.ndarray] = []

    for length in message_lengths:
        length = int(length)
        latent = np.empty(length, dtype=np.int64)
        latent[0] = rng.choice(alphabet_size, p=initial_probs)
        for t in range(1, length):
            latent[t] = rng.choice(alphabet_size, p=trans_probs[latent[t - 1]])
        messages.append(latent)

    return messages


def sample_static_substitution_corpus_like(
    reference: Corpus,
    *,
    alpha: float = 0.5,
    structure_seed: int | None = None,
    seed: int | None = None,
    message_prefix: str = "static_sub",
) -> Corpus:
    """
    Sample a Markov-1 latent source and emit it through one fixed substitution.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    if structure_seed is None:
        structure_seed = seed

    alphabet_size = reference.alphabet_size
    reference_messages = [msg.symbols for msg in reference.messages]
    message_lengths = [msg.length for msg in reference.messages]

    initial_probs = _fit_initial_probs_from_corpus(reference, alpha=alpha)
    trans_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha,
    )
    latent_messages = _sample_latent_markov1_messages(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        seed=seed,
    )

    permutation = np.random.default_rng(structure_seed).permutation(alphabet_size)

    messages: list[Message] = []
    for i, latent in enumerate(latent_messages):
        emitted = permutation[latent]
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=int(latent.size),
                symbols=emitted.astype(np.int64).tolist(),
                unigram=False,
            )
        )

    return Corpus(alphabet_size=alphabet_size, messages=messages)


def _make_vigenere_key(
    *,
    alphabet_size: int,
    key_period: int,
    seed: int | None = None,
) -> np.ndarray:
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if key_period <= 0:
        raise ValueError("key_period must be positive.")

    rng = np.random.default_rng(seed)
    return rng.integers(0, alphabet_size, size=key_period, dtype=np.int64)


def sample_vigenere_like_corpus_like(
    reference: Corpus,
    *,
    key_period: int = 5,
    alpha: float = 0.5,
    structure_seed: int | None = None,
    seed: int | None = None,
    message_prefix: str = "vigenere_like",
) -> Corpus:
    """
    Sample a Markov-1 latent source and emit it with a periodic additive key.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if key_period <= 0:
        raise ValueError("key_period must be positive.")

    if structure_seed is None:
        structure_seed = seed

    alphabet_size = reference.alphabet_size
    reference_messages = [msg.symbols for msg in reference.messages]
    message_lengths = [msg.length for msg in reference.messages]

    initial_probs = _fit_initial_probs_from_corpus(reference, alpha=alpha)
    trans_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha,
    )
    latent_messages = _sample_latent_markov1_messages(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_probs=initial_probs,
        trans_probs=trans_probs,
        seed=seed,
    )

    key = _make_vigenere_key(
        alphabet_size=alphabet_size,
        key_period=key_period,
        seed=structure_seed,
    )

    messages: list[Message] = []
    for i, latent in enumerate(latent_messages):
        positions = np.arange(latent.size, dtype=np.int64)
        emitted = (latent + key[positions % key_period]) % alphabet_size
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=int(latent.size),
                symbols=emitted.astype(np.int64).tolist(),
                unigram=False,
            )
        )

    return Corpus(alphabet_size=alphabet_size, messages=messages)
