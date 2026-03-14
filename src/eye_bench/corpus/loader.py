import orjson
from pathlib import Path
from .schema import Corpus


def load_corpus(path: str | Path) -> Corpus:
    data = orjson.loads(Path(path).read_bytes())
    return Corpus.model_validate(data)
