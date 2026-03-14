import numpy as np

from .schema import Corpus


def corpus_to_arrays(corpus: Corpus) -> dict[str, np.ndarray]:
    return {
        msg.message_id: np.asarray(msg.symbols, dtype=np.int64)
        for msg in corpus.messages
    }


def corpus_to_message_arrays(corpus: Corpus) -> list[np.ndarray]:
    return [np.asarray(msg.symbols, dtype=np.int64) for msg in corpus.messages]
