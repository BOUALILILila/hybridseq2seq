import pytest

import numpy as np

import torch

from hybridseq2seq.data.parsers.semantic_parser import SemanticParser


@pytest.fixture
def source_tokens():
    return "<s> mary like the cat </s>".split()


@pytest.fixture
def source_tokens_no_sos():
    return "mary like the cat </s>".split()


@pytest.fixture
def target_tokens():
    return "<s> * cat ( x _ 3 ) ; like . agent ( x _ 1 , Mary ) AND like . theme ( x _ 1 , x _ 3 ) </s>".split()


@pytest.fixture
def source_distance_matrix():
    return torch.tensor(
        [
            [0.0, 2.0, 1.0, 3.0, 2.0],
            [2.0, 0.0, 1.0, 3.0, 2.0],
            [1.0, 1.0, 0.0, 2.0, 1.0],
            [3.0, 3.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def source_distance_matrix_no_sos():
    return torch.tensor(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 0.0, 2.0, 1.0],
            [3.0, 2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def gold_cross_distance_matrix_agent():
    #     D = torch.tensor([
    #                       *     cat             3          like     agent          1       mary
    #         [0, 1, 3, 2,  3.25, 2.25, 0, 0, 0, 2.25, 0, 0, 1.25, 0, 0.5, 0, 0, 0, 1.25, 0, 0.25],
    #         [0, 0, 2, 1,  2.25, 1.25, 0, 0, 0, 1.25, 0, 0, 0.25, 0, 0.5, 0, 0, 0, 0.25, 0, 1.25],
    #         [0, 0, 0, 1,  0.25, 1.25, 0, 0, 0, 1.25, 0, 0, 2.25, 0, 2.5, 0, 0, 0, 2.25, 0, 3.25],
    #         [0, 0, 0, 0,  1.25, 0.25, 0, 0, 0, 0.25, 0, 0, 1.25, 0, 1.5, 0, 0, 0, 1.25, 0, 2.25],
    #     *   [0, 0, 0, 0,  0   , 1   , 0, 0, 0, 1   , 0, 0, 2   , 0, 2.5, 0, 0, 0, 2   , 0, 3   ],
    #     cat [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 1   , 0, 1.5, 0, 0, 0, 1   , 0, 2   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #      3  [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 1   , 0, 1.5, 0, 0, 0, 1   , 0, 2   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #    like [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0.5, 0, 0, 0, 0   , 0, 1   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #   agent [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0.5 , 0, 0.5 ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #    1    [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 1   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #         [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0  , 0, 0, 0, 0   , 0, 0   ],
    #     ])
    # D = torch.tensor([
    #     [0, 1, 3, 2,  3.25, 2.25, 0, 0, 0, 2.25, 0, 0, 1.25, 0, 0.75, 0, 0, 0, 1.25, 0, 0.25],
    #     [0, 0, 2, 1,  2.25, 1.25, 0, 0, 0, 1.25, 0, 0, 0.25, 0, 0.75, 0, 0, 0, 0.25, 0, 1.25],
    #     [0, 0, 0, 1,  0.25, 1.25, 0, 0, 0, 1.25, 0, 0, 2.25, 0, 2.75, 0, 0, 0, 2.25, 0, 3.25],
    #     [0, 0, 0, 0,  1.25, 0.25, 0, 0, 0, 0.25, 0, 0, 1.25, 0, 1.75, 0, 0, 0, 1.25, 0, 2.25],
    #     [0, 0, 0, 0,  0   , 1   , 0, 0, 0, 1   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0.5, 0, 0, 0, 0   , 0, 1   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0.5 , 0, 0.5 ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 1   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    #     [0, 0, 0, 0,  0   , 0   , 0, 0, 0, 0   , 0, 0, 0   , 0, 0   , 0, 0, 0, 0   , 0, 0   ],
    # ])
    D = torch.tensor(
        [
            [
                0.0000,
                2.0000,
                1.0000,
                3.0000,
                2.0000,
                0.2500,
                3.2500,
                2.2500,
                0.0000,
                0.0000,
                0.0000,
                2.2500,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                1.7500,
                0.0000,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                2.2500,
            ],
            [
                2.0000,
                0.0000,
                1.0000,
                3.0000,
                2.0000,
                2.2500,
                3.2500,
                2.2500,
                0.0000,
                0.0000,
                0.0000,
                2.2500,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                0.7500,
                0.0000,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                0.2500,
            ],
            [
                1.0000,
                1.0000,
                0.0000,
                2.0000,
                1.0000,
                1.2500,
                2.2500,
                1.2500,
                0.0000,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                0.0000,
                0.2500,
                0.0000,
                0.7500,
                0.0000,
                0.0000,
                0.0000,
                0.2500,
                0.0000,
                1.2500,
            ],
            [
                3.0000,
                3.0000,
                2.0000,
                0.0000,
                1.0000,
                3.2500,
                0.2500,
                1.2500,
                0.0000,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                0.0000,
                2.2500,
                0.0000,
                2.7500,
                0.0000,
                0.0000,
                0.0000,
                2.2500,
                0.0000,
                3.2500,
            ],
            [
                2.0000,
                2.0000,
                1.0000,
                1.0000,
                0.0000,
                2.2500,
                1.2500,
                0.2500,
                0.0000,
                0.0000,
                0.0000,
                0.2500,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                1.7500,
                0.0000,
                0.0000,
                0.0000,
                1.2500,
                0.0000,
                2.2500,
            ],
            [
                0.2500,
                2.2500,
                1.2500,
                3.2500,
                2.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                3.2500,
                3.2500,
                2.2500,
                0.2500,
                1.2500,
                0.0000,
                0.0000,
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                2.2500,
                2.2500,
                1.2500,
                1.2500,
                0.2500,
                0.0000,
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                2.2500,
                2.2500,
                1.2500,
                1.2500,
                0.2500,
                0.0000,
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                1.2500,
                1.2500,
                0.2500,
                2.2500,
                1.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                1.7500,
                0.7500,
                0.7500,
                2.7500,
                1.7500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.5000,
                0.0000,
                0.5000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                1.2500,
                1.2500,
                0.2500,
                2.2500,
                1.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                2.2500,
                0.2500,
                1.2500,
                3.2500,
                2.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                0.0000,
                1.0000,
                0.0000,
                0.0000,
            ],
        ]
    )
    return D


def test_cogs_logic_form_parsing(source_tokens, target_tokens):
    parser = SemanticParser(epsilon=0.25, add_sos_token=True)

    target_seq = " ".join(target_tokens[:3])  # "<s> * cat"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4
    assert [dep[0].value in ("def", "*", "cat") for dep in dep_tuples]
    assert [dep[1].position == -1 for dep in dep_tuples]

    target_seq = " ".join(target_tokens[:7])  # "<s> * cat ( x _ 3 )"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4
    assert [dep[0].value in ("def", "*", "cat") for dep in dep_tuples]
    assert [dep[1].position > -1 for dep in dep_tuples]

    target_seq = " ".join(target_tokens[:10])  # "... like"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4 + 1
    for dep in dep_tuples:
        if dep[0].value == "like":
            assert dep[0].position > -1
            assert dep[1].value == 2 and dep[1].position == -1

    target_seq = " ".join(target_tokens[:12])  # "... like . agent"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4 + 2
    for dep in dep_tuples:
        print(dep)
        if dep[0].value == "like":
            assert dep[0].position > -1 and dep[1].position == -1
            assert dep[1].value == 2
        elif dep[0].value == "agent":
            assert dep[0].position > -1 and dep[1].position == -1
            assert dep[1].value == 2

    target_seq = " ".join(target_tokens[:16])  # "... like . agent ( x _ 1"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4 + 2
    for dep in dep_tuples:
        if dep[0].value == "like":
            assert dep[0].position > -1 and dep[1].position > 0
            assert dep[1].value == 2
        elif dep[0].value == "agent":
            assert dep[0].position > -1 and dep[1].position > 0
            assert dep[1].value == 2

    target_seq = " ".join(target_tokens[:19])  # "... like . agent ( x _ 1 , Mary )"
    dep_tuples = parser.get_dependency_tuples(source_tokens, target_seq)
    assert len(dep_tuples) == 4 + 3
    for dep in dep_tuples:
        if dep[0].value == "like":
            assert dep[0].position > -1 and dep[1].position > 0
            assert dep[1].value == 2
        elif dep[0].value == "Mary":
            assert dep[0].position > -1 and dep[1].position == -1
            assert dep[1].value == 1
        elif dep[0].value == "agent":
            assert (
                dep[0].position > -1 and dep[1].position > 0 and dep[2].position == -1
            )
            assert dep[1].value == 2 and dep[2].value == 1


def test_distance_matrix_stop_words(
    source_tokens_no_sos, target_tokens, source_distance_matrix_no_sos
):
    source_tokens = source_tokens_no_sos
    source_distance_matrix = source_distance_matrix_no_sos

    parser = SemanticParser(epsilon=0.25, add_sos_token=False)

    i = 1
    mask = torch.ones(1, len(target_tokens), dtype=source_distance_matrix.dtype)
    src_seq_len = source_distance_matrix.shape[-1]
    D, attn_mask = parser.get_distance_matrix(
        step=i,
        previous_cross_distance_matrix=source_distance_matrix.unsqueeze(0),
        source_distance_matrix=source_distance_matrix.unsqueeze(0),
        source_tokens_list=[source_tokens],
        previous_target_tokens_list=[target_tokens[:i]],  # "<s>"
        target_attention_mask_step=mask[:, :i],
    )
    assert torch.allclose(D[0], D[0].T, rtol=1e-5, atol=1e-8)
    assert torch.allclose(
        D[0, :src_seq_len, :src_seq_len], source_distance_matrix, rtol=1e-5, atol=1e-8
    )
    assert attn_mask[0, -1] == torch.zeros_like(attn_mask[0, -1])

    i = 2
    previous_attention_mask = torch.ones_like(mask[:, :i])
    previous_attention_mask[:, :-1] = attn_mask
    D, attn_mask = parser.get_distance_matrix(
        step=i,
        previous_cross_distance_matrix=D,
        source_distance_matrix=source_distance_matrix.unsqueeze(0),
        source_tokens_list=[source_tokens],
        previous_target_tokens_list=[target_tokens[:i]],  # "<s> *"
        target_attention_mask_step=previous_attention_mask,
    )

    assert torch.allclose(D[0], D[0].T, rtol=1e-5, atol=1e-8)
    assert torch.allclose(
        D[0, :src_seq_len, :src_seq_len], source_distance_matrix, rtol=1e-5, atol=1e-8
    )
    assert attn_mask[0, -1] == torch.zeros_like(attn_mask[0, -1])

    i = 3
    previous_attention_mask = torch.ones_like(mask[:, :i])
    previous_attention_mask[:, :-1] = attn_mask
    D, attn_mask = parser.get_distance_matrix(
        step=i,
        previous_cross_distance_matrix=D,
        source_distance_matrix=source_distance_matrix.unsqueeze(0),
        source_tokens_list=[source_tokens],
        previous_target_tokens_list=[target_tokens[:i]],  # "<s> * cat"
        target_attention_mask_step=previous_attention_mask,
    )
    assert torch.allclose(D[0], D[0].T, rtol=1e-5, atol=1e-8)
    assert torch.allclose(
        D[0, :src_seq_len, :src_seq_len], source_distance_matrix, rtol=1e-5, atol=1e-8
    )
    assert torch.allclose(
        attn_mask[0, 1:], torch.ones_like(attn_mask[0, 1:]), rtol=1e-8, atol=1e-8
    )
    assert torch.allclose(
        attn_mask[0, 0], torch.zeros_like(attn_mask[0, 0]), rtol=1e-8, atol=1e-8
    )
    assert torch.allclose(
        D[0, src_seq_len + 1, :src_seq_len],
        D[0, 2, :src_seq_len] + parser.epsilon,
        rtol=1e-5,
        atol=1e-8,
    )  # d("*") == d("the") + epsilon
    assert torch.allclose(
        D[0, src_seq_len + 2, :src_seq_len],
        D[0, 3, :src_seq_len] + parser.epsilon,
        rtol=1e-5,
        atol=1e-8,
    )  # d("cat") == d("cat") + epsilon
    assert torch.allclose(
        D[0, src_seq_len + 2, src_seq_len],
        D[0, src_seq_len, src_seq_len + 2],
        rtol=1e-8,
        atol=1e-8,
    )  # d("*", "cat") == 1


def test_distance_matrix(
    source_tokens,
    target_tokens,
    source_distance_matrix,
    gold_cross_distance_matrix_agent,
):
    parser = SemanticParser(epsilon=0.25, add_sos_token=True)
    mask = torch.ones(1, len(target_tokens), dtype=source_distance_matrix.dtype)
    attn_mask = mask[:, 0]
    D = source_distance_matrix
    for i in range(1, 19):
        print(target_tokens[:i])
        print(source_tokens)
        previous_attention_mask = torch.ones_like(mask[:, :i])
        previous_attention_mask[:, :-1] = attn_mask
        D, attn_mask = parser.get_distance_matrix(
            step=i,
            previous_cross_distance_matrix=D,
            source_distance_matrix=source_distance_matrix.unsqueeze(0),
            source_tokens_list=[source_tokens],
            previous_target_tokens_list=[target_tokens[:i]],
            target_attention_mask_step=previous_attention_mask,
        )
        assert torch.allclose(D[0], D[0].T, rtol=1e-5, atol=1e-8)
    print(D[0])
    assert torch.allclose(D[0], gold_cross_distance_matrix_agent, rtol=1e-8, atol=1e-8)
