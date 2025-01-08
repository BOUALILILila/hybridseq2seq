import pytest

from hybridseq2seq.data import COGSDataModule


def test_train_data_loader():
    d = COGSDataModule(
        train_batch_size=4,
        valid_batch_size=2,
        train_num_workers=1,
        max_seq_length=5,
        data_dir="./tests/resources/cache",
    )
    d.setup(stage="fit")
    assert len(d.train_set) == 32
    assert len(d.valid_set) == 16

    d.setup(stage="test")
    assert len(d.test_set) == 4
    assert d.test_set.in_vocabulary == d.test_set.out_vocabulary
    assert len(d.test_set.in_vocabulary) == len(d.train_set.in_vocabulary)

    features = d.test_set[0]
    assert features["in_seq_ids"][0] == d.test_set.in_vocabulary.sos_idx
    assert features["in_seq_ids"][-1] == d.test_set.in_vocabulary.eos_idx
    assert features["out_seq_ids"][-1] == d.test_set.out_vocabulary.eos_idx
    assert features["in_dist_matrix"].shape[0] == features["in_len"]


def test_train_data_loader_no_shared():
    # No shared vocab
    d = COGSDataModule(
        train_batch_size=4,
        valid_batch_size=2,
        train_num_workers=1,
        max_seq_length=5,
        data_dir="./tests/resources/cache",
        shared_vocab=False,
    )
    d.setup(stage="fit")
    assert len(d.train_set) == 32
    assert len(d.valid_set) == 16

    d.setup(stage="test")
    assert len(d.test_set) == 4
    assert d.test_set.in_vocabulary != d.test_set.out_vocabulary
    assert len(d.test_set.in_vocabulary) == len(d.train_set.in_vocabulary)


def test_train_data_loader_add_eos():
    # Do not add eos
    d = COGSDataModule(
        train_batch_size=4,
        valid_batch_size=2,
        train_num_workers=1,
        max_seq_length=5,
        data_dir="./tests/resources/cache",
        add_eos_token=False,
    )
    d.setup(stage="fit")
    assert len(d.train_set) == 32
    assert len(d.valid_set) == 16

    d.setup(stage="test")
    assert len(d.test_set) == 4

    features = d.test_set[0]
    assert features["in_seq_ids"][0] == d.test_set.in_vocabulary.sos_idx
    assert features["in_seq_ids"][-1] != d.test_set.in_vocabulary.eos_idx
    assert features["out_seq_ids"][-1] == d.test_set.out_vocabulary.eos_idx
    assert features["in_dist_matrix"].shape[0] == features["in_len"]
    assert len(features["in_seq_ids"]) == features["in_len"]
    assert len(features["out_seq_ids"]) == features["out_len"]


def test_train_data_loader_no_sos_no_eos():
    # Do not add eos nor sos
    d = COGSDataModule(
        train_batch_size=4,
        valid_batch_size=2,
        train_num_workers=1,
        max_seq_length=5,
        data_dir="./tests/resources/cache",
        add_sos_token=False,
        add_eos_token=False,
    )

    d.setup(stage="test")
    assert len(d.test_set) == 4

    features = d.test_set[0]
    assert features["in_seq_ids"][0] != d.test_set.in_vocabulary.sos_idx
    assert features["in_seq_ids"][-1] != d.test_set.in_vocabulary.eos_idx
    assert features["out_seq_ids"][-1] == d.test_set.out_vocabulary.eos_idx
    assert features["in_dist_matrix"].shape[0] == features["in_len"]


def test_train_data_loader_gold_parser():
    # Do not add eos nor sos
    d = COGSDataModule(
        train_batch_size=4,
        valid_batch_size=2,
        train_num_workers=1,
        max_seq_length=5,
        data_dir="./tests/resources/syntax_cache",
        add_sos_token=True,
        add_eos_token=True,
        syntax_parser="gold",
    )

    d.setup(stage="test")
    assert len(d.test_set) == 4

    features = d.test_set[0]
    assert features["in_seq_ids"][0] == d.test_set.in_vocabulary.sos_idx
    assert features["in_seq_ids"][-1] == d.test_set.in_vocabulary.eos_idx
    assert features["out_seq_ids"][-1] == d.test_set.out_vocabulary.eos_idx
    assert features["in_dist_matrix"].shape[0] == features["in_len"]
    assert len(features["in_seq_tokens"]) == features["in_len"]
    assert len(features["in_seq_ids"]) == features["in_len"]
