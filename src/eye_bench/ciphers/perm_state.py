from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np

from ..corpus import Corpus
from ..corpus.schema import Message
from ..invariants.models import fit_markov1_probs


@dataclass(slots=True)
class PermStateSamplingDiagnostics:
    num_states: int
    total_generated_tokens: int
    state_visit_counts: dict[int, int]
    state_transition_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state_visit_counts"] = {
            str(k): v for k, v in self.state_visit_counts.items()
        }
        return payload


def _fit_initial_probs_from_corpus(
    reference: Corpus,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Fit a smoothed distribution over message-initial latent tokens.
    """
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


def _validate_state_permutations(
    state_permutations: Sequence[np.ndarray],
    alphabet_size: int,
) -> list[np.ndarray]:
    """
    Validate and normalize state permutations.

    Each permutation is a length-A array where perm[z] = emitted token for latent z.
    """
    if not state_permutations:
        raise ValueError("state_permutations must not be empty.")

    normalized: list[np.ndarray] = []
    target = np.arange(alphabet_size, dtype=np.int64)

    for i, perm in enumerate(state_permutations):
        arr = np.asarray(perm, dtype=np.int64)
        if arr.ndim != 1 or arr.shape[0] != alphabet_size:
            raise ValueError(
                f"Permutation {i} must be a 1D array of length {alphabet_size}."
            )
        if np.any(arr < 0) or np.any(arr >= alphabet_size):
            raise ValueError(
                f"Permutation {i} contains values outside [0, {alphabet_size})."
            )
        if not np.array_equal(np.sort(arr), target):
            raise ValueError(f"Permutation {i} is not a valid permutation.")
        normalized.append(arr)

    return normalized


def make_cyclic_state_permutations(
    alphabet_size: int,
    num_states: int,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Create a v1 bank of state permutations.

    Construction:
        - sample one base permutation over the alphabet
        - derive each state permutation by cyclically rolling that base array

    This yields a simple, low-parameter evolving-remapping family.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if num_states <= 0:
        raise ValueError("num_states must be positive.")

    rng = np.random.default_rng(seed)
    base = rng.permutation(alphabet_size).astype(np.int64)

    return [np.roll(base, shift=s).copy() for s in range(num_states)]


def sample_perm_state_corpus_with_diagnostics(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    latent_init_probs: np.ndarray,
    latent_trans_probs: np.ndarray,
    state_permutations: Sequence[np.ndarray],
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v1",
) -> tuple[Corpus, PermStateSamplingDiagnostics]:
    """
    Sample a corpus from the v1 permutation-state family.

    Model:
        Z_0 ~ q_init
        Z_t | Z_{t-1} ~ q_latent(. | Z_{t-1})

        C_t = pi_{S_t}(Z_t)

        S_0 = initial_state
        S_{t+1} = (S_t + 1) mod K

    Notes:
        - the hidden state resets to `initial_state` at the start of each message
        - state is advanced before emitting the next position, so message positions
          use states:
              0, 1, 2, ..., K-1, 0, 1, ...
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    latent_init_probs = np.asarray(latent_init_probs, dtype=np.float64)
    if latent_init_probs.ndim != 1 or latent_init_probs.shape[0] != alphabet_size:
        raise ValueError(
            "latent_init_probs must be a 1D array of length alphabet_size."
        )
    if np.any(latent_init_probs < 0):
        raise ValueError("latent_init_probs must be nonnegative.")
    init_total = latent_init_probs.sum()
    if init_total <= 0:
        raise ValueError("latent_init_probs must sum to a positive value.")
    latent_init_probs = latent_init_probs / init_total

    latent_trans_probs = np.asarray(latent_trans_probs, dtype=np.float64)
    if latent_trans_probs.ndim != 2 or latent_trans_probs.shape != (
        alphabet_size,
        alphabet_size,
    ):
        raise ValueError(
            "latent_trans_probs must have shape (alphabet_size, alphabet_size)."
        )
    if np.any(latent_trans_probs < 0):
        raise ValueError("latent_trans_probs must be nonnegative.")
    row_sums = latent_trans_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "Every row of latent_trans_probs must sum to a positive value."
        )
    latent_trans_probs = latent_trans_probs / row_sums

    state_permutations = _validate_state_permutations(state_permutations, alphabet_size)
    num_states = len(state_permutations)

    rng = np.random.default_rng(seed)

    state_visit_counts = {s: 0 for s in range(num_states)}
    state_transition_counts: dict[str, int] = {}
    total_generated_tokens = 0

    messages: list[Message] = []

    for i, length in enumerate(message_lengths):
        length = int(length)
        latent = np.empty(length, dtype=np.int64)
        emitted = np.empty(length, dtype=np.int64)

        state = int(initial_state) % num_states

        latent[0] = rng.choice(alphabet_size, p=latent_init_probs)
        emitted[0] = state_permutations[state][latent[0]]
        state_visit_counts[state] += 1

        for t in range(1, length):
            next_state = (state + 1) % num_states
            key = f"{state}->{next_state}"
            state_transition_counts[key] = state_transition_counts.get(key, 0) + 1
            state = next_state

            latent[t] = rng.choice(alphabet_size, p=latent_trans_probs[latent[t - 1]])
            emitted[t] = state_permutations[state][latent[t]]
            state_visit_counts[state] += 1

        total_generated_tokens += length
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=length,
                symbols=emitted.tolist(),
                unigram=False,
            )
        )

    diagnostics = PermStateSamplingDiagnostics(
        num_states=num_states,
        total_generated_tokens=total_generated_tokens,
        state_visit_counts=state_visit_counts,
        state_transition_counts=state_transition_counts,
    )

    return Corpus(alphabet_size=alphabet_size, messages=messages), diagnostics


def sample_perm_state_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    latent_init_probs: np.ndarray,
    latent_trans_probs: np.ndarray,
    state_permutations: Sequence[np.ndarray],
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v1",
) -> Corpus:
    corpus, _ = sample_perm_state_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        latent_init_probs=latent_init_probs,
        latent_trans_probs=latent_trans_probs,
        state_permutations=state_permutations,
        initial_state=initial_state,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus


def sample_perm_state_corpus_like_with_diagnostics(
    reference: Corpus,
    *,
    num_states: int = 4,
    alpha: float = 0.5,
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v1",
) -> tuple[Corpus, PermStateSamplingDiagnostics]:
    """
    Fit the latent source model from the reference corpus, then sample a v1
    permutation-state synthetic corpus with matching message lengths.

    v1 family:
        - latent source: fitted Markov-1
        - state update: cyclic
        - emission mapping: one rolled permutation per state
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    alphabet_size = reference.alphabet_size
    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    latent_init_probs = _fit_initial_probs_from_corpus(reference, alpha=alpha)
    latent_trans_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha,
    )
    state_permutations = make_cyclic_state_permutations(
        alphabet_size=alphabet_size,
        num_states=num_states,
        seed=seed,
    )

    return sample_perm_state_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        latent_init_probs=latent_init_probs,
        latent_trans_probs=latent_trans_probs,
        state_permutations=state_permutations,
        initial_state=initial_state,
        seed=seed,
        message_prefix=message_prefix,
    )


def sample_perm_state_corpus_like(
    reference: Corpus,
    *,
    num_states: int = 4,
    alpha: float = 0.5,
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v1",
) -> Corpus:
    corpus, _ = sample_perm_state_corpus_like_with_diagnostics(
        reference,
        num_states=num_states,
        alpha=alpha,
        initial_state=initial_state,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus


@dataclass(slots=True)
class PermStateV2SamplingDiagnostics:
    num_states: int
    update_mode: str
    total_generated_tokens: int
    total_state_transitions: int
    state_visit_counts: dict[int, int]
    state_transition_counts: dict[str, int]
    delta_usage_counts: dict[int, int]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state_visit_counts"] = {
            str(k): v for k, v in self.state_visit_counts.items()
        }
        payload["delta_usage_counts"] = {
            str(k): v for k, v in self.delta_usage_counts.items()
        }
        return payload


def _validate_state_delta_by_token(
    state_delta_by_token: np.ndarray,
    *,
    alphabet_size: int,
    num_states: int,
) -> np.ndarray:
    arr = np.asarray(state_delta_by_token, dtype=np.int64)
    if arr.ndim != 1 or arr.shape[0] != alphabet_size:
        raise ValueError(
            f"state_delta_by_token must be a 1D array of length {alphabet_size}."
        )
    if np.any(arr < 0) or np.any(arr >= num_states):
        raise ValueError(
            f"state_delta_by_token must contain integers in [0, {num_states})."
        )
    return arr


def make_bucketed_state_delta_by_token(
    alphabet_size: int,
    num_states: int,
    *,
    num_buckets: int | None = None,
    seed: int | None = None,
    shuffle_tokens: bool = True,
    shuffle_bucket_deltas: bool = True,
) -> np.ndarray:
    """
    Create a low-variance token->state-shift map.

    Tokens are partitioned into a small number of buckets. Every token in the same
    bucket gets the same state shift.

    This is intentionally more stable than assigning a fully random shift to every
    token independently.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if num_states <= 0:
        raise ValueError("num_states must be positive.")

    if num_buckets is None:
        num_buckets = min(num_states, 4)

    if not (1 <= num_buckets <= num_states):
        raise ValueError("num_buckets must satisfy 1 <= num_buckets <= num_states.")

    rng = np.random.default_rng(seed)

    tokens = np.arange(alphabet_size, dtype=np.int64)
    if shuffle_tokens:
        rng.shuffle(tokens)

    buckets = np.array_split(tokens, num_buckets)

    if shuffle_bucket_deltas:
        bucket_deltas = rng.permutation(num_states)[:num_buckets].astype(np.int64)
    else:
        bucket_deltas = np.arange(num_buckets, dtype=np.int64)

    state_delta_by_token = np.zeros(alphabet_size, dtype=np.int64)
    for bucket_tokens, delta in zip(buckets, bucket_deltas, strict=False):
        state_delta_by_token[bucket_tokens] = int(delta)

    return state_delta_by_token


def make_random_state_delta_by_token(
    alphabet_size: int,
    num_states: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """
    Create a fully random token->state-shift map.

    This is more expressive than the bucketed version, but usually noisier.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if num_states <= 0:
        raise ValueError("num_states must be positive.")

    rng = np.random.default_rng(seed)
    return rng.integers(
        low=0,
        high=num_states,
        size=alphabet_size,
        dtype=np.int64,
    )


def make_state_delta_by_token(
    alphabet_size: int,
    num_states: int,
    *,
    mode: str = "bucketed",
    num_buckets: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Dispatcher for v2 token-driven state updates.

    Supported modes:
        - "bucketed"
        - "random"
    """
    if mode == "bucketed":
        return make_bucketed_state_delta_by_token(
            alphabet_size=alphabet_size,
            num_states=num_states,
            num_buckets=num_buckets,
            seed=seed,
        )
    if mode == "random":
        return make_random_state_delta_by_token(
            alphabet_size=alphabet_size,
            num_states=num_states,
            seed=seed,
        )
    raise ValueError(f"Unknown delta mode: {mode}")


def sample_perm_state_v2_corpus_with_diagnostics(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    latent_init_probs: np.ndarray,
    latent_trans_probs: np.ndarray,
    state_permutations: Sequence[np.ndarray],
    state_delta_by_token: np.ndarray,
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v2",
    update_mode: str = "token_driven",
) -> tuple[Corpus, PermStateV2SamplingDiagnostics]:
    """
    Sample a corpus from the v2 permutation-state family.

    Model:
        Z_0 ~ q_init
        Z_t | Z_{t-1} ~ q_latent(. | Z_{t-1})

        C_t = pi_{S_t}(Z_t)

        S_0 = initial_state
        S_{t+1} = (S_t + delta(Z_t)) mod K

    Notes:
        - state update is driven by the latent token just emitted
        - hidden state resets to `initial_state` at the start of each message
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if any(length <= 0 for length in message_lengths):
        raise ValueError("All message lengths must be positive.")

    latent_init_probs = np.asarray(latent_init_probs, dtype=np.float64)
    if latent_init_probs.ndim != 1 or latent_init_probs.shape[0] != alphabet_size:
        raise ValueError(
            "latent_init_probs must be a 1D array of length alphabet_size."
        )
    if np.any(latent_init_probs < 0):
        raise ValueError("latent_init_probs must be nonnegative.")
    init_total = latent_init_probs.sum()
    if init_total <= 0:
        raise ValueError("latent_init_probs must sum to a positive value.")
    latent_init_probs = latent_init_probs / init_total

    latent_trans_probs = np.asarray(latent_trans_probs, dtype=np.float64)
    if latent_trans_probs.ndim != 2 or latent_trans_probs.shape != (
        alphabet_size,
        alphabet_size,
    ):
        raise ValueError(
            "latent_trans_probs must have shape (alphabet_size, alphabet_size)."
        )
    if np.any(latent_trans_probs < 0):
        raise ValueError("latent_trans_probs must be nonnegative.")
    row_sums = latent_trans_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "Every row of latent_trans_probs must sum to a positive value."
        )
    latent_trans_probs = latent_trans_probs / row_sums

    state_permutations = _validate_state_permutations(state_permutations, alphabet_size)
    num_states = len(state_permutations)

    state_delta_by_token = _validate_state_delta_by_token(
        state_delta_by_token,
        alphabet_size=alphabet_size,
        num_states=num_states,
    )

    rng = np.random.default_rng(seed)

    state_visit_counts = {s: 0 for s in range(num_states)}
    state_transition_counts: dict[str, int] = {}
    delta_usage_counts = {d: 0 for d in range(num_states)}

    total_generated_tokens = 0
    total_state_transitions = 0

    messages: list[Message] = []

    for i, length in enumerate(message_lengths):
        length = int(length)
        latent = np.empty(length, dtype=np.int64)
        emitted = np.empty(length, dtype=np.int64)

        state = int(initial_state) % num_states

        latent[0] = rng.choice(alphabet_size, p=latent_init_probs)
        emitted[0] = state_permutations[state][latent[0]]
        state_visit_counts[state] += 1

        for t in range(1, length):
            delta = int(state_delta_by_token[latent[t - 1]])
            delta_usage_counts[delta] += 1

            next_state = (state + delta) % num_states
            key = f"{state}->{next_state}"
            state_transition_counts[key] = state_transition_counts.get(key, 0) + 1
            total_state_transitions += 1

            state = next_state

            latent[t] = rng.choice(alphabet_size, p=latent_trans_probs[latent[t - 1]])
            emitted[t] = state_permutations[state][latent[t]]
            state_visit_counts[state] += 1

        total_generated_tokens += length
        messages.append(
            Message(
                message_id=f"{message_prefix}_{i}",
                length=length,
                symbols=emitted.tolist(),
                unigram=False,
            )
        )

    diagnostics = PermStateV2SamplingDiagnostics(
        num_states=num_states,
        update_mode=update_mode,
        total_generated_tokens=total_generated_tokens,
        total_state_transitions=total_state_transitions,
        state_visit_counts=state_visit_counts,
        state_transition_counts=state_transition_counts,
        delta_usage_counts=delta_usage_counts,
    )

    return Corpus(alphabet_size=alphabet_size, messages=messages), diagnostics


def sample_perm_state_v2_corpus(
    *,
    alphabet_size: int,
    message_lengths: Sequence[int],
    latent_init_probs: np.ndarray,
    latent_trans_probs: np.ndarray,
    state_permutations: Sequence[np.ndarray],
    state_delta_by_token: np.ndarray,
    initial_state: int = 0,
    seed: int | None = None,
    message_prefix: str = "perm_state_v2",
    update_mode: str = "token_driven",
) -> Corpus:
    corpus, _ = sample_perm_state_v2_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        latent_init_probs=latent_init_probs,
        latent_trans_probs=latent_trans_probs,
        state_permutations=state_permutations,
        state_delta_by_token=state_delta_by_token,
        initial_state=initial_state,
        seed=seed,
        message_prefix=message_prefix,
        update_mode=update_mode,
    )
    return corpus


def sample_perm_state_v2_corpus_like_with_diagnostics(
    reference: Corpus,
    *,
    num_states: int = 4,
    alpha: float = 0.5,
    initial_state: int = 0,
    delta_mode: str = "bucketed",
    num_buckets: int | None = None,
    structure_seed: int | None = None,
    seed: int | None = None,
    message_prefix: str = "perm_state_v2",
) -> tuple[Corpus, PermStateV2SamplingDiagnostics]:
    """
    Fit the latent source model from the reference corpus, then sample a v2
    permutation-state synthetic corpus with matching message lengths.

    v2 family:
        - latent source: fitted Markov-1
        - state update: token-driven
        - emission mapping: one rolled permutation per state
        - token->state-shift map: bucketed or random
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    if structure_seed is None:
        structure_seed = seed

    alphabet_size = reference.alphabet_size
    message_lengths = [msg.length for msg in reference.messages]
    reference_messages = [msg.symbols for msg in reference.messages]

    latent_init_probs = _fit_initial_probs_from_corpus(reference, alpha=alpha)
    latent_trans_probs = fit_markov1_probs(
        reference_messages,
        alphabet_size=alphabet_size,
        alpha=alpha,
    )

    state_permutations = make_cyclic_state_permutations(
        alphabet_size=alphabet_size,
        num_states=num_states,
        seed=structure_seed,
    )

    state_delta_by_token = make_state_delta_by_token(
        alphabet_size=alphabet_size,
        num_states=num_states,
        mode=delta_mode,
        num_buckets=num_buckets,
        seed=structure_seed,
    )

    return sample_perm_state_v2_corpus_with_diagnostics(
        alphabet_size=alphabet_size,
        message_lengths=message_lengths,
        latent_init_probs=latent_init_probs,
        latent_trans_probs=latent_trans_probs,
        state_permutations=state_permutations,
        state_delta_by_token=state_delta_by_token,
        initial_state=initial_state,
        seed=seed,
        message_prefix=message_prefix,
        update_mode=f"token_driven:{delta_mode}",
    )


def sample_perm_state_v2_corpus_like(
    reference: Corpus,
    *,
    num_states: int = 4,
    alpha: float = 0.5,
    initial_state: int = 0,
    delta_mode: str = "bucketed",
    num_buckets: int | None = None,
    structure_seed: int | None = None,
    seed: int | None = None,
    message_prefix: str = "perm_state_v2",
) -> Corpus:
    corpus, _ = sample_perm_state_v2_corpus_like_with_diagnostics(
        reference,
        num_states=num_states,
        alpha=alpha,
        initial_state=initial_state,
        delta_mode=delta_mode,
        num_buckets=num_buckets,
        structure_seed=structure_seed,
        seed=seed,
        message_prefix=message_prefix,
    )
    return corpus
