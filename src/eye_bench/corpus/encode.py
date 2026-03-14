import numpy as np
from .schema import Corpus


def corpus_to_arrays(corpus: Corpus) -> dict[str, np.ndarray]:
    return {
        msg.message_id: np.asarray(msg.symbols, dtype=np.int16)
        for msg in corpus.messages
    }
