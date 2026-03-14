from .schema import Corpus


def validate_corpus(corpus: Corpus) -> None:
    seen = set()
    for msg in corpus.messages:
        assert msg.message_id not in seen
        seen.add(msg.message_id)
        assert msg.length == len(msg.symbols)
        assert msg.length > 0
        for x in msg.symbols:
            assert 0 <= x < corpus.alphabet_size
