from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np

from ..corpus import Corpus
from ..corpus.schema import Message
from ..invariants.counts import context_count_dict
from ..invariants.models import fit_markov1_probs, fit_markov_k_probs, fit_unigram_probs


@dataclass(slots=True)
class Markov2ContextSparsity:
    alphabet_size: int
    possible_contexts: int
    observed_contexts: int
    observed_context_fraction: float

    total_context_emissions: int

    mean_context_count: float
    median_context_count: float
    max_context_count: int

    singleton_contexts: int
    singleton_context_fraction: float
    emissions_from_singletons: int
    emission_fraction_from_singletons: float

    low_support_contexts_le_2: int
    low_support_context_fraction_le_2: float
    emissions_from_low_support_le_2: int
    emission_fraction_from_low_support_le_2: float

    def to_json_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class Markov2SamplingDiagnostics:
    total_generated_tokens: int
    total_context_steps: int

    context_seen_steps: int
    context_unseen_steps: int
    context_unseen_rate: float

    unique_contexts_encountered: int
    unique_unseen_contexts_encountered: int

    mode_step_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_markov2_context_sparsity(
    reference: Corpus,
) -> Markov2ContextSparsity:
    """
    Diagnose how sparse the observed order-2 context space is in the reference corpus.
    """
    messages = [msg.symbols for msg in reference.messages]
    counts_by_context, total_transitions = context_count_dict(
        messages,
        alphabet_size=reference.alphabet_size,
        k=2,
    )

    context_counts = np.array(
        [int(counts.sum()) for counts in counts_by_context.values()],
        dtype=np.int64,
    )

    possible_contexts = int(reference.alphabet_size**2)
    observed_contexts = int(context_counts.size)

    if observed_contexts == 0:
        return Markov2ContextSparsity(
            alphabet_size=reference.alphabet_size,
            possible_contexts=possible_contexts,
            observed_contexts=0,
            observed_context_fraction=0.0,
            total_context_emissions=int(total_transitions),
            mean_context_count=0.0,
            median_context_count=0.0,
            max_context_count=0,
            singleton_contexts=0,
            singleton_context_fraction=0.0,
            emissions_from_singletons=0,
            emission_fraction_from_singletons=0.0,
            low_support_contexts_le_2=0,
            low_support_context_fraction_le_2=0.0,
            emissions_from_low_support_le_2=0,
            emission_fraction_from_low_support_le_2=0.0,
        )

    singleton_mask = context_counts == 1
    low_support_mask = context_counts <= 2

    singleton_contexts = int(np.sum(singleton_mask))
    emissions_from_singletons = int(np.sum(context_counts[singleton_mask]))

    low_support_contexts_le_2 = int(np.sum(low_support_mask))
    emissions_from_low_support_le_2 = int(np.sum(context_counts[low_support_mask]))

    return Markov2ContextSparsity(
        alphabet_size=reference.alphabet_size,
        possible_contexts=possible_contexts,
        observed_contexts=observed_contexts,
        observed_context_fraction=float(observed_contexts / possible_contexts),
        total_context_emissions=int(total_transitions),
        mean_context_count=float(np.mean(context_counts)),
        median_context_count=float(np.median(context_counts)),
        max_context_count=int(np.max(context_counts)),
        singleton_contexts=singleton_contexts,
        singleton_context_fraction=float(singleton_contexts / observed_contexts),
        emissions_from_singletons=emissions_from_singletons,
        emission_fraction_from_singletons=float(
            emissions_from_singletons / total_transitions
        )
        if total_transitions > 0
        else 0.0,
        low_support_contexts_le_2=low_support_contexts_le_2,
        low_support_context_fraction_le_2=float(
            low_support_contexts_le_2 / observed_contexts
        ),
        emissions_from_low_support_le_2=emissions_from_low_support_le_2,
        emission_fraction_from_low_support_le_2=float(
            emissions_from_low_support_le_2 / total_transitions
        )
        if total_transitions > 0
        else 0.0,
    )


def _fit_initial_pair_probs_from_corpus(
    reference: Corpus,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Fit a smoothed joint distribution over the first two symbols of each message.

    Shape:
        (alphabet_size, alphabet_size)
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    alphabet_size = reference.alphabet_size
    pair_counts = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)

    for msg in reference.messages:
        if msg.length >= 2:
            pair_counts[msg.symbols[0], msg.symbols[1]] += 1.0

    pair_probs = pair_counts + alpha
    total = pair_probs.sum()
    if total <= 0:
        raise ValueError("Cannot fit initial pair distribution.")
    return pair_probs / total


def _fit_first_symbol_probs_from_corpus(
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
        raise ValueError("Cannot fit first-symbol distribution.")
    return probs / total


def _sample_initial_pair(
    rng: np.random.Generator,
    pair_probs: np.ndarray,
) -> tuple[int, int]:
    alphabet_size = pair_probs.shape[0]
    flat_idx = int(rng.choice(alphabet_size * alphabet_size, p=pair_probs.ravel()))
    a, b = divmod(flat_idx, alphabet_size)
    return a, b


def _normalize_prob_vector(
    probs: np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != expected_size:
        raise ValueError(f"{name} must be a 1D array of length {expected_size}.")
    if np.any(probs < 0):
        raise ValueError(f"{name} must be nonnegative.")
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value.")
    return probs / total


def _normalize_prob_matrix(
    probs: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if np.any(probs < 0):
        raise ValueError(f"{name} must be nonnegative.")
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value.")
    return probs / total


def _normalize_context_prob_map(
    trans_probs_by_context: dict[tuple[int, int], np.ndarray],
    alphabet_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    normalized: dict[tuple[int, int], np.ndarray] = {}
    for context, probs in trans_probs_by_context.items():
        normalized[context] = _normalize_prob_vector(
            probs,
            expected_size=alphabet_size,
            name=f"trans_probs_by_context[{context}]",
        )
    return normalized


def _mix_distributions(
    weighted_parts: list[tuple[float, np.ndarray | None]],
    *,
    alphabet_size: int,
) -> np.ndarray:
    """
    Mix available distributions and renormalize over the active weights only.
    """
    out = np.zeros(alphabet_size, dtype=np.float64)
    active_weight = 0.0

    for weight, probs in weighted_parts:
        if weight <= 0.0 or probs is None:
            continue
        out += float(weight) * probs
        active_weight += float(weight)

    if active_weight <= 0.0:
        raise ValueError("No active distributions available for interpolation.")

    out /= active_weight
    out /= out.sum()
    return out


def sample_markov2_corpus_with_diagnostics(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_pair_probs: np.ndarray,
    first_symbol_probs: np.ndarray,
    trans_probs_by_context: dict[tuple[int, int], np.ndarray],
    fallback_probs: np.ndarray | None = None,
    seed: int | None = None,
    message_prefix: str = "markov2",
) -> tuple[Corpus, Markov2SamplingDiagnostics]:
    """
    Sample a synthetic corpus from a sparse second-order Markov model and report
    how often unseen contexts forced fallback.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    initial_pair_probs = _normalize_prob_matrix(
        initial_pair_probs,
        expected_shape=(alphabet_size, alphabet_size),
        name="initial_pair_probs",
    )
    first_symbol_probs = _normalize_prob_vector(
        first_symbol_probs,
        expected_size=alphabet_size,
        name="first_symbol_probs",
    )
    trans_probs_by_context = _normalize_context_prob_map(
        trans_probs_by_context,
        alphabet_size=alphabet_size,
    )

    if fallback_probs is None:
        fallback_probs = np.full(alphabet_size, 1.0 / alphabet_size, dtype=np.float64)
    else:
        fallback_probs = _normalize_prob_vector(
            fallback_probs,
            expected_size=alphabet_size,
            name="fallback_probs",
        )

    rng = np.random.default_rng(seed)

    messages: list[Message] = []

    total_generated_tokens = 0
    total_context_steps = 0
    context_seen_steps = 0
    context_unseen_steps = 0

    encountered_contexts: set[tuple[int, int]] = set()
    unseen_contexts: set[tuple[int, int]] = set()

    mode_step_counts = {
        "order2_only": 0,
        "fallback_unigram": 0,
    }

    for i, length in enumerate(message_lengths):
        length = int(length)
        symbols = np.empty(length, dtype=np.int64)

        if length == 1:
            symbols[0] = rng.choice(alphabet_size, p=first_symbol_probs)
        else:
            a, b = _sample_initial_pair(rng, initial_pair_probs)
            symbols[0] = a
            symbols[1] = b

            for t in range(2, length):
                context = (int(symbols[t - 2]), int(symbols[t - 1]))
                encountered_contexts.add(context)
                total_context_steps += 1

                probs = trans_probs_by_context.get(context)
                if probs is None:
                    unseen_contexts.add(context)
                    context_unseen_steps += 1
                    mode_step_counts["fallback_unigram"] += 1
                    probs = fallback_probs
                else:
                    context_seen_steps += 1
                    mode_step_counts["order2_only"] += 1

                symbols[t] = rng.choice(alphabet_size, p=probs)

        total_generated_tokens += length
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=length,
                symbols=symbols.tolist(),
                unigram=False,
            )
        )

    diagnostics = Markov2SamplingDiagnostics(
        total_generated_tokens=total_generated_tokens,
        total_context_steps=total_context_steps,
        context_seen_steps=context_seen_steps,
        context_unseen_steps=context_unseen_steps,
        context_unseen_rate=float(context_unseen_steps / total_context_steps)
        if total_context_steps > 0
        else 0.0,
        unique_contexts_encountered=len(encountered_contexts),
        unique_unseen_contexts_encountered=len(unseen_contexts),
        mode_step_counts=mode_step_counts,
    )

    return (
        Corpus(alphabet_size=alphabet_size, messages=messages),
        diagnostics,
    )


def sample_markov2_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_pair_probs: np.ndarray,
    first_symbol_probs: np.ndarray,
    trans_probs_by_context: dict[tuple[int, int], np.ndarray],
    fallback_probs: np.ndarray | None = None,
    seed: int | None = None,
    message_prefix: str = "markov2",
) -> Corpus:
    corpus, _ = sample_markov2_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_pair_probs=initial_pair_probs,
        first_symbol_probs=first_symbol_probs,
        trans_probs_by_context=trans_probs_by_context,
        fallback_probs=fallback_probs,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus


def sample_markov2_corpus_like_with_diagnostics(
    reference: Corpus,
    *,
    alpha: float = 0.5,
    init_alpha: float = 0.5,
    fallback_alpha: float = 0.5,
    seed: int | None = None,
    message_prefix: str = "markov2",
) -> tuple[Corpus, Markov2SamplingDiagnostics]:
    """
    Fit a sparse second-order Markov model to the reference corpus, then sample
    a synthetic corpus and return fallback diagnostics.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if init_alpha < 0:
        raise ValueError("init_alpha must be >= 0.")
    if fallback_alpha < 0:
        raise ValueError("fallback_alpha must be >= 0.")

    alphabet_size = reference.alphabet_size
    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    initial_pair_probs = _fit_initial_pair_probs_from_corpus(
        reference, alpha=init_alpha
    )
    first_symbol_probs = _fit_first_symbol_probs_from_corpus(
        reference, alpha=init_alpha
    )

    trans_probs_by_context = fit_markov_k_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        k=2,
        alpha=alpha,
    )

    fallback_probs = fit_unigram_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=fallback_alpha,
    )

    return sample_markov2_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_pair_probs=initial_pair_probs,
        first_symbol_probs=first_symbol_probs,
        trans_probs_by_context=trans_probs_by_context,
        fallback_probs=fallback_probs,
        seed=seed,
        message_prefix=message_prefix,
    )


def sample_markov2_corpus_like(
    reference: Corpus,
    *,
    alpha: float = 0.5,
    init_alpha: float = 0.5,
    fallback_alpha: float = 0.5,
    seed: int | None = None,
    message_prefix: str = "markov2",
) -> Corpus:
    corpus, _ = sample_markov2_corpus_like_with_diagnostics(
        reference,
        alpha=alpha,
        init_alpha=init_alpha,
        fallback_alpha=fallback_alpha,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus


def sample_markov2_backoff_corpus_with_diagnostics(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_pair_probs: np.ndarray,
    first_symbol_probs: np.ndarray,
    order2_probs_by_context: dict[tuple[int, int], np.ndarray],
    order1_probs: np.ndarray,
    order0_probs: np.ndarray,
    lambdas: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int | None = None,
    message_prefix: str = "markov2_backoff",
) -> tuple[Corpus, Markov2SamplingDiagnostics]:
    """
    Sample from an interpolated/backoff Markov-2 model:

        P(x_t | x_{t-2}, x_{t-1})
            = λ2 P2(x_t | x_{t-2}, x_{t-1})
            + λ1 P1(x_t | x_{t-1})
            + λ0 P0(x_t)

    If the order-2 context is unseen, only the available lower-order components
    are mixed and renormalized over active weights.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    lambda2, lambda1, lambda0 = map(float, lambdas)
    if min(lambda2, lambda1, lambda0) < 0.0:
        raise ValueError("All lambdas must be nonnegative.")
    if (lambda2 + lambda1 + lambda0) <= 0.0:
        raise ValueError("At least one lambda must be positive.")

    initial_pair_probs = _normalize_prob_matrix(
        initial_pair_probs,
        expected_shape=(alphabet_size, alphabet_size),
        name="initial_pair_probs",
    )
    first_symbol_probs = _normalize_prob_vector(
        first_symbol_probs,
        expected_size=alphabet_size,
        name="first_symbol_probs",
    )
    order1_probs = _normalize_prob_matrix(
        order1_probs,
        expected_shape=(alphabet_size, alphabet_size),
        name="order1_probs",
    )
    order0_probs = _normalize_prob_vector(
        order0_probs,
        expected_size=alphabet_size,
        name="order0_probs",
    )
    order2_probs_by_context = _normalize_context_prob_map(
        order2_probs_by_context,
        alphabet_size=alphabet_size,
    )

    rng = np.random.default_rng(seed)

    messages: list[Message] = []

    total_generated_tokens = 0
    total_context_steps = 0
    context_seen_steps = 0
    context_unseen_steps = 0

    encountered_contexts: set[tuple[int, int]] = set()
    unseen_contexts: set[tuple[int, int]] = set()

    mode_step_counts = {
        "order2+order1+order0": 0,
        "order1+order0": 0,
    }

    for i, length in enumerate(message_lengths):
        length = int(length)
        symbols = np.empty(length, dtype=np.int64)

        if length == 1:
            symbols[0] = rng.choice(alphabet_size, p=first_symbol_probs)
        else:
            a, b = _sample_initial_pair(rng, initial_pair_probs)
            symbols[0] = a
            symbols[1] = b

            for t in range(2, length):
                prev2 = int(symbols[t - 2])
                prev1 = int(symbols[t - 1])
                context = (prev2, prev1)

                encountered_contexts.add(context)
                total_context_steps += 1

                order2 = order2_probs_by_context.get(context)
                order1 = order1_probs[prev1]

                if order2 is None:
                    unseen_contexts.add(context)
                    context_unseen_steps += 1
                    mode_step_counts["order1+order0"] += 1
                    probs = _mix_distributions(
                        [
                            (lambda1, order1),
                            (lambda0, order0_probs),
                        ],
                        alphabet_size=alphabet_size,
                    )
                else:
                    context_seen_steps += 1
                    mode_step_counts["order2+order1+order0"] += 1
                    probs = _mix_distributions(
                        [
                            (lambda2, order2),
                            (lambda1, order1),
                            (lambda0, order0_probs),
                        ],
                        alphabet_size=alphabet_size,
                    )

                symbols[t] = rng.choice(alphabet_size, p=probs)

        total_generated_tokens += length
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=length,
                symbols=symbols.tolist(),
                unigram=False,
            )
        )

    diagnostics = Markov2SamplingDiagnostics(
        total_generated_tokens=total_generated_tokens,
        total_context_steps=total_context_steps,
        context_seen_steps=context_seen_steps,
        context_unseen_steps=context_unseen_steps,
        context_unseen_rate=float(context_unseen_steps / total_context_steps)
        if total_context_steps > 0
        else 0.0,
        unique_contexts_encountered=len(encountered_contexts),
        unique_unseen_contexts_encountered=len(unseen_contexts),
        mode_step_counts=mode_step_counts,
    )

    return (
        Corpus(alphabet_size=alphabet_size, messages=messages),
        diagnostics,
    )


def sample_markov2_backoff_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    initial_pair_probs: np.ndarray,
    first_symbol_probs: np.ndarray,
    order2_probs_by_context: dict[tuple[int, int], np.ndarray],
    order1_probs: np.ndarray,
    order0_probs: np.ndarray,
    lambdas: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int | None = None,
    message_prefix: str = "markov2_backoff",
) -> Corpus:
    corpus, _ = sample_markov2_backoff_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_pair_probs=initial_pair_probs,
        first_symbol_probs=first_symbol_probs,
        order2_probs_by_context=order2_probs_by_context,
        order1_probs=order1_probs,
        order0_probs=order0_probs,
        lambdas=lambdas,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus


def sample_markov2_backoff_corpus_like_with_diagnostics(
    reference: Corpus,
    *,
    alpha2: float = 0.5,
    alpha1: float = 0.5,
    alpha0: float = 0.5,
    init_alpha: float = 0.5,
    lambdas: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int | None = None,
    message_prefix: str = "markov2_backoff",
) -> tuple[Corpus, Markov2SamplingDiagnostics]:
    """
    Fit an interpolated/backoff Markov-2 model to the reference corpus, then
    sample a synthetic corpus and return diagnostics.
    """
    if alpha2 < 0 or alpha1 < 0 or alpha0 < 0 or init_alpha < 0:
        raise ValueError("All alpha values must be >= 0.")

    alphabet_size = reference.alphabet_size
    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    initial_pair_probs = _fit_initial_pair_probs_from_corpus(
        reference, alpha=init_alpha
    )
    first_symbol_probs = _fit_first_symbol_probs_from_corpus(
        reference, alpha=init_alpha
    )

    order2_probs_by_context = fit_markov_k_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        k=2,
        alpha=alpha2,
    )
    order1_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha1,
    )
    order0_probs = fit_unigram_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha0,
    )

    return sample_markov2_backoff_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        initial_pair_probs=initial_pair_probs,
        first_symbol_probs=first_symbol_probs,
        order2_probs_by_context=order2_probs_by_context,
        order1_probs=order1_probs,
        order0_probs=order0_probs,
        lambdas=lambdas,
        seed=seed,
        message_prefix=message_prefix,
    )


def sample_markov2_backoff_corpus_like(
    reference: Corpus,
    *,
    alpha2: float = 0.5,
    alpha1: float = 0.5,
    alpha0: float = 0.5,
    init_alpha: float = 0.5,
    lambdas: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int | None = None,
    message_prefix: str = "markov2_backoff",
) -> Corpus:
    corpus, _ = sample_markov2_backoff_corpus_like_with_diagnostics(
        reference,
        alpha2=alpha2,
        alpha1=alpha1,
        alpha0=alpha0,
        init_alpha=init_alpha,
        lambdas=lambdas,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus
