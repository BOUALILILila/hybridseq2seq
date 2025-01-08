import tempfile

import numpy as np

from .conftest import (
    dummy_text_dataset,
    dummy_test_text_dataset,
    dummy_train_text_dataset,
    dummy_train_text_dataset_sos,
    dummy_train_text_dataset_eos,
    dummy_text_dataset_shared_vocab,
    dummy_text_dataset_oov,
    train_src_seq,
    train_tgt_seq,
)


def test_dataset_no_sos_eos(dummy_train_text_dataset):
    data = dummy_train_text_dataset[0]
    assert (
        data["in_seq_ids"][-1] != dummy_train_text_dataset._cache.in_vocabulary.eos_idx
    )
    assert (
        data["out_seq_ids"][-1]
        == dummy_train_text_dataset._cache.out_vocabulary.eos_idx
    )
    assert (
        data["in_seq_ids"][0] != dummy_train_text_dataset._cache.in_vocabulary.sos_idx
    )


def test_dataset_add_sos(dummy_train_text_dataset_sos):
    data = dummy_train_text_dataset_sos[0]
    assert (
        data["in_seq_ids"][0]
        == dummy_train_text_dataset_sos._cache.in_vocabulary.sos_idx
    )


def test_dataset_add_eos(dummy_train_text_dataset_eos):
    data = dummy_train_text_dataset_eos[0]
    assert (
        data["in_seq_ids"][-1]
        == dummy_train_text_dataset_eos._cache.in_vocabulary.eos_idx
    )
    assert (
        data["out_seq_ids"][-1]
        == dummy_train_text_dataset_eos._cache.out_vocabulary.eos_idx
    )


def test_dataset_encoding_train_split(
    dummy_train_text_dataset, train_src_seq, train_tgt_seq
):
    data = dummy_train_text_dataset[0]
    vocab_in = dummy_train_text_dataset._cache.in_vocabulary
    vocab_out = dummy_train_text_dataset._cache.out_vocabulary
    assert vocab_in.get_string(data["in_seq_ids"]) == train_src_seq
    assert vocab_out.get_string(data["out_seq_ids"]) == train_tgt_seq


def test_dataset_encoding_test_split(
    dummy_test_text_dataset, test_src_seq, test_tgt_seq
):
    data = dummy_test_text_dataset[0]
    vocab_in = dummy_test_text_dataset._cache.in_vocabulary
    vocab_out = dummy_test_text_dataset._cache.out_vocabulary
    assert vocab_in.get_string(data["in_seq_ids"]) == test_src_seq
    assert vocab_out.get_string(data["out_seq_ids"]) == test_tgt_seq


def test_dataset_shared_vocab(dummy_text_dataset_shared_vocab, train_tgt_seq):
    in_enc = dummy_text_dataset_shared_vocab.in_vocabulary(train_tgt_seq)
    out_enc = dummy_text_dataset_shared_vocab.out_vocabulary(train_tgt_seq)
    assert in_enc == out_enc


def test_dataset_separate_vocab(dummy_text_dataset_oov, train_tgt_seq):
    in_enc = dummy_text_dataset_oov.in_vocabulary(train_tgt_seq)
    out_enc = dummy_text_dataset_oov.out_vocabulary(train_tgt_seq)
    assert dummy_text_dataset_oov.in_vocabulary.unk_idx in in_enc
    assert in_enc != out_enc


def test_load_cache(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        dataset = dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=False,
            add_eos_token=False,
        )
        data = dataset[0]

        dataset._cache = dataset._load_dataset()
        data_rec = dataset[0]

        assert np.all(data["in_seq_ids"] == data_rec["in_seq_ids"]) and np.all(
            data["out_seq_ids"] == data_rec["out_seq_ids"]
        )


def test_load_cache_diff_syn_parser(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        gold_dataset = dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=True,
            add_eos_token=True,
            syntax_parser="gold",
        )
        data = gold_dataset[0]
        print(data)

        # raise value error
        # pred_dataset = dummy_text_dataset(
        #     sets=["train"],
        #     cache_dir=tmp_directory,
        #     add_sos_token=False,
        #     add_eos_token=False,
        #     syntax_parser=None,
        # )
        # pred_data = pred_dataset[0]
