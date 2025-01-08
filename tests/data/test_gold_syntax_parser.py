import pytest
import numpy as np
from hybridseq2seq.data.parsers.gold_syntax_parser import GoldSyntaxParser


@pytest.fixture
def sample():
    return (
        "A rose was help by a dog .",
        "( S ( NP ( Det A ) ( N rose ) ) ( VP ( AUX was ) ( V helped ) ( BY by ) ( NP ( Det a ) ( N dog ) ) ) )",
        np.array(
            [
                [0.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0],
                [3.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0],
                [2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0],
                [2.0, 3.0, 2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0],
                [1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                [2.0, 3.0, 2.0, 2.0, 1.0, 0.0, 3.0, 2.0, 2.0, 2.0],
                [3.0, 4.0, 3.0, 3.0, 2.0, 3.0, 0.0, 1.0, 3.0, 3.0],
                [2.0, 3.0, 2.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0],
                [2.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.0, 2.0],
                [2.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 0.0],
            ]
        ),
    )


@pytest.fixture
def primitive():
    return (
        "like",
        "( V like )",
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
    )


def test_gold_syntax_tree_parsing(sample):
    s, st, _ = sample
    parser = GoldSyntaxParser()
    graph, lem = parser.get_dependency_tree(st)
    assert len(lem) == len(s.split(" "))
    assert len(graph._graph) == len(lem) + 1  # for the end token
    assert s.split(" ") == lem


def test_gold_syntax_tree_distances(sample):
    s, st, gold_D = sample
    parser = GoldSyntaxParser()
    D, lem = parser.get_distance_matrix(st)  # adds the start token
    assert s.split(" ") == lem
    assert D.shape[0] == len(lem) + 2  # for the start and end tokens
    assert np.allclose(D, gold_D)


def test_gold_syntax_parser_primitive(primitive):
    s, st, gold_D = primitive
    parser = GoldSyntaxParser()
    D, lem = parser.get_distance_matrix(st)
    assert s.split(" ") == lem
    assert D.shape[0] == len(lem) + 2  # for the start and end tokens
    assert np.allclose(D, gold_D)
