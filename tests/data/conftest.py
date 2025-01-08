import pytest

import tempfile

from hybridseq2seq.data.datasets.dataset import TextDataset, TextDatasetCache


@pytest.fixture
def unique_word_vocab_sents():
    return ["test test"]


@pytest.fixture
def split_vocab_sents():
    return ["test.vocab"]


@pytest.fixture
def train_src_seq():
    return "train"


@pytest.fixture
def train_tgt_seq():
    return "train target sentence"


@pytest.fixture
def test_src_seq():
    return "test"


@pytest.fixture
def test_tgt_seq():
    return "test target sentence"


@pytest.fixture
def dummy_text_dataset(train_src_seq, train_tgt_seq, test_src_seq, test_tgt_seq):
    class DummyTextDataset(TextDataset):
        def build_cache(self):
            index_table = {"train": [0], "test": [1]}
            in_sents = [train_src_seq, test_src_seq]
            out_sents = [train_tgt_seq, test_tgt_seq]
            syn_trees = ["( N train )", "( V test )"]
            return TextDatasetCache().build(
                index_table={"simple": index_table},
                in_sentences=in_sents,
                out_sentences=out_sents,
                syn_trees_in_sentences=syn_trees,
                syntax_parser=self.syntax_parser,
                syntax_parser_type=self.syntax_parser_type,
            )

    return DummyTextDataset


@pytest.fixture
def dummy_train_text_dataset(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        return dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=False,
            add_eos_token=False,
        )


@pytest.fixture
def dummy_train_text_dataset_sos(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        return dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=True,
            add_eos_token=False,
        )


@pytest.fixture
def dummy_train_text_dataset_eos(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        return dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=False,
            add_eos_token=True,
        )


@pytest.fixture
def dummy_test_text_dataset(dummy_text_dataset):
    tmp_directory = tempfile.TemporaryDirectory()
    return dummy_text_dataset(sets=["test"], cache_dir=tmp_directory.name)


@pytest.fixture
def dummy_text_dataset_shared_vocab(dummy_text_dataset):
    with tempfile.TemporaryDirectory() as tmp_directory:
        return dummy_text_dataset(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=False,
            add_eos_token=False,
            shared_vocabulary=True,
        )


@pytest.fixture
def dummy_text_dataset_allow_oov(
    train_src_seq, train_tgt_seq, test_src_seq, test_tgt_seq
):
    class DummyTextDatasetAllowOOV(TextDataset):
        def build_cache(self):
            index_table = {"train": [0], "test": [1]}
            in_sents = [train_src_seq, test_src_seq]
            out_sents = [train_tgt_seq, test_tgt_seq]
            return TextDatasetCache().build(
                index_table={"simple": index_table},
                in_sentences=in_sents,
                out_sentences=out_sents,
                allow_any_word=True,
                syntax_parser=self.syntax_parser,
                syntax_parser_type=self.syntax_parser_type,
            )

    return DummyTextDatasetAllowOOV


@pytest.fixture
def dummy_text_dataset_oov(dummy_text_dataset_allow_oov):
    with tempfile.TemporaryDirectory() as tmp_directory:
        return dummy_text_dataset_allow_oov(
            sets=["train"],
            cache_dir=tmp_directory,
            add_sos_token=False,
            add_eos_token=False,
            shared_vocabulary=False,
        )
