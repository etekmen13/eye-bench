from eye_bench.ciphers import (
    sample_perm_state_v3_corpus_like,
    sample_static_substitution_corpus_like,
    sample_vigenere_like_corpus_like,
)
from eye_bench.corpus import load_corpus, validate_corpus


def test_extended_generators_preserve_corpus_shape():
    corpus = load_corpus("data/processed/eyes.json")
    validate_corpus(corpus)

    generators = [
        sample_static_substitution_corpus_like(corpus, seed=0),
        sample_vigenere_like_corpus_like(corpus, seed=0),
        sample_perm_state_v3_corpus_like(corpus, seed=0),
    ]

    reference_lengths = [message.length for message in corpus.messages]
    for generated in generators:
        validate_corpus(generated)
        assert generated.alphabet_size == corpus.alphabet_size
        assert [message.length for message in generated.messages] == reference_lengths
