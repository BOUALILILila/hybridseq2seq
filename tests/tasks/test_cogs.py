import shutil

import pytest
import torch

from hybridseq2seq.data import WordVocabulary
from hybridseq2seq.tasks.cogs import COGSAccuracy, COGSTask


@pytest.fixture
def config_path():
    return "tests/resources/config.yaml"


@pytest.fixture
def syntax_config_path():
    return "tests/resources/syntax_config.yaml"


def test_task_init(config_path):
    task = COGSTask(config_file=config_path, no_cuda=True)
    assert len(task.out_vocab) == task.config.model.vocab_size


def test_task_train(config_path):
    shutil.rmtree("tests/resources/output/COGS")
    task = COGSTask(config_file=config_path, no_cuda=False)
    task.run_train()


def test_task_train_gold(syntax_config_path):
    shutil.rmtree("tests/resources/syntax_output/COGS")
    task = COGSTask(config_file=syntax_config_path, no_cuda=False)
    task.run_train()


def test_metric_reset():
    metric = COGSAccuracy(type_names=["type 1", "type 2"], max_bad_samples=2)

    assert metric.correct == 0
    assert metric.total == 0
    assert len(metric.bad_sequences) == 0
    assert (
        len(metric.correct_per_type.keys()) == 2
        and metric.correct_per_type[0] == 0
        and metric.correct_per_type[1] == 0
    )
    assert (
        len(metric.total_per_type.keys()) == 2
        and metric.total_per_type[0] == 0
        and metric.total_per_type[1] == 0
    )
    assert metric.type_names[0] == "type 1"
    assert metric.type_names[1] == "type 2"

    metric = COGSAccuracy(type_names=["type 1", "type 2"], max_bad_samples=2)
    assert metric.correct == 0
    assert metric.total == 0
    assert len(metric.bad_sequences) == 0
    assert (
        len(metric.correct_per_type.keys()) == 2
        and metric.correct_per_type[0] == 0
        and metric.correct_per_type[1] == 0
    )
    assert (
        len(metric.total_per_type.keys()) == 2
        and metric.total_per_type[0] == 0
        and metric.total_per_type[1] == 0
    )
    assert metric.type_names[0] == "type 1"
    assert metric.type_names[1] == "type 2"


def test_metric_update_compute():
    metric = COGSAccuracy(type_names=["type 1", "type 2"], max_bad_samples=4)

    in_seq = [
        "paula paint a cake",
        "zoe think",
        "the princess teleport",
        "a cockroach eat a pretzel",
    ]
    in_seq = [seq.split(" ") for seq in in_seq]

    out_seq = [
        "paint . agent ( x _ 1 , paula ) and paint . theme ( x _ 1 , x _ 3 ) and cake ( x _ 3 ) </s>",
        "think . agent ( x _ 1 , Zoe ) </s> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>",
        "* princess ( x _ 1 ) ; teleport . agent ( x _ 2 , x _ 1 ) </s> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>",
        "cockroach ( x _ 1 ) and eat . agent ( x _ 2 , x _ 1 ) and pretzel ( x _ 4 ) </s> <pad> <pad> <pad> <pad>",
    ]

    pred_seq = [
        "paint . agent ( x _ 1 , zoe ) and paint . theme ( x _ 1 , x _ 3 ) and cake ( x _ 3 ) </s> <pad>",  # paula -> zoe
        "think . agent ( x _ 1 , Zoe ) </s> <pad> <pad> ) <pad> x <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>",  # correct, tokens after eos
        "* princess ( x _ 1 ) ; teleport . agent ( x _ 2 , x _ 1 ) </s> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>",  # correct all
        "cockroach ( x _ 1 ) and eat . agent ( x _ 2 , x _ 1 ) and pretzel ( x _ 4 ) ) </s> <pad> <pad> <pad> <pad>",  # additional ) at the end
    ]

    vocab = WordVocabulary(in_seq, split_punctuation=False) + WordVocabulary(
        out_seq, split_punctuation=False
    )

    targets = torch.tensor([vocab(seq) for seq in out_seq])
    len_targets = torch.tensor([31, 11, 21, 27])

    preds = torch.tensor([vocab(seq) for seq in pred_seq])
    len_preds = torch.tensor([31, 11, 21, 28])

    types = torch.tensor([0, 1, 0, 0])

    metric.update(
        preds=preds[:2],
        len_preds=len_preds[:2],
        targets=targets[:2],
        len_targets=len_targets[:2],
        decode=lambda batched_token_ids: [
            vocab(seq.tolist()) for seq in batched_token_ids
        ],
        data={
            "source_token_seqs": in_seq[:2],
            "type": types[:2],
        },
    )

    assert metric.total == 2
    assert metric.correct == 1
    assert metric.total_per_type[0] == 1
    assert metric.correct_per_type[0] == 0
    assert metric.total_per_type[1] == 1
    assert metric.correct_per_type[1] == 1

    metric.update(
        preds=preds[2:],
        len_preds=len_preds[2:],
        targets=targets[2:],
        len_targets=len_targets[2:],
        decode=lambda batched_token_ids: [
            vocab(seq.tolist()) for seq in batched_token_ids
        ],
        data={
            "source_token_seqs": in_seq[2:],
            "type": types[2:],
        },
    )

    assert metric.total == 4
    assert metric.correct == 2
    assert metric.total_per_type[0] == 3
    assert metric.correct_per_type[0] == 1
    assert metric.total_per_type[1] == 1
    assert metric.correct_per_type[1] == 1

    metrics = metric.compute("test")
    assert metrics["test/accuracy/type 1"] == 1 / 3
    assert metrics["test/accuracy/type 2"] == 1
    assert metrics["test/accuracy"] == 2 / 4
