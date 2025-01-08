from hybridseq2seq.data import WordVocabulary

from .conftest import unique_word_vocab_sents, split_vocab_sents


def test_empty_vocab():
    empty_vocab = WordVocabulary()
    # sos, pad, unk, eos special tokens only
    assert len(empty_vocab) == len(empty_vocab.special_tokens)


def test_init_vocab(unique_word_vocab_sents):
    vocab = WordVocabulary(unique_word_vocab_sents)
    assert len(vocab) == 1 + len(vocab.special_tokens)
    assert vocab[len(vocab.special_tokens)] == "test"


def test_split_sents_punctuation(split_vocab_sents):
    vocab = WordVocabulary(split_vocab_sents, split_punctuation=True)
    assert len(vocab) == 3 + len(vocab.special_tokens)


def test_split_sents_whitespaces(split_vocab_sents):
    vocab = WordVocabulary(split_vocab_sents, split_punctuation=False)
    assert len(vocab) == 1 + len(vocab.special_tokens)


def test_merge_vocabs(unique_word_vocab_sents):
    vocab1 = WordVocabulary(unique_word_vocab_sents)
    vocab2 = WordVocabulary(unique_word_vocab_sents)
    vocab = vocab1 + vocab2
    assert len(vocab) == 1 + len(vocab.special_tokens)
    assert vocab[len(vocab.special_tokens)] == "test"


def test_encoding(unique_word_vocab_sents):
    vocab = WordVocabulary(unique_word_vocab_sents)
    assert (
        vocab.get_string(vocab(unique_word_vocab_sents[0]))
        == unique_word_vocab_sents[0]
    )


def test_oov_tokens(unique_word_vocab_sents):
    vocab = WordVocabulary(unique_word_vocab_sents, allow_any_word=True)
    assert vocab("word") == [vocab.unk_idx]
