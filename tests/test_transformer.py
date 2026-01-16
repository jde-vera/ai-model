import pytest
from transformer import Transformer
from collections import Counter


def test_tokenize_basic_sentence():
    transformer = Transformer()
    s = "Miley Cyrus - Party In The U.S.A"

    tokens = transformer.tokenize(s)

    assert tokens == [
        "Miley", "Cyrus", "-", "Party", "In", "The", "U", ".", "S", ".", "A"
    ]


def test_tokenize_counter_counts_tokens():
    transformer = Transformer()
    s = "Miley Cyrus - Party In The U.S.A"

    tokens = transformer.tokenize(s)

    counter = Counter()
    counter.update(tokens)

    assert counter["Miley"] == 1
    assert counter["."] == 2
    assert counter["-"] == 1


def test_build_vocab_creates_vocab():
    transformer = Transformer()

    texts = ["Miley Cyrus - Party In The U.S.A"]
    transformer.build_vocab(texts)

    vocab = transformer.vocab

    # special tokens
    assert "[PAD]" in vocab
    assert "[UNK]" in vocab

    # regular tokens
    assert "Miley" in vocab
    assert "Cyrus" in vocab
    assert "." in vocab

def test_pad_and_mask():
    t = Transformer()
    t.vocab = {"[PAD]": 0, "[UNK]": 1}  # minimal vocab for this test

    ids, mask = t.pad_and_mask([5, 6, 7], max_len=5)
    assert ids == [5, 6, 7, 0, 0]
    assert mask == [1, 1, 1, 0, 0]
