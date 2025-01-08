import pytest

import numpy as np

from hybridseq2seq.data.parsers.syntactic_parser import SyntacticParser


@pytest.fixture
def simple_sentence():
    return "Mary reads a book ."


@pytest.fixture
def primitive():
    return "Mary"


def test_dependency_tree_contruction(simple_sentence):
    parser = SyntacticParser()
    g, vocab, _ = parser.get_dependency_tree(simple_sentence)
    assert g.number_of_nodes() == len(vocab)
    assert len(vocab) == len(simple_sentence.split()) + 2


def test_distance_matrix_construction(simple_sentence):
    parser = SyntacticParser()
    D, lem_sent = parser.get_distance_matrix(simple_sentence)
    assert np.allclose(D, D.T, rtol=1e-5, atol=1e-8)
    assert np.allclose(
        np.diagonal(D), np.zeros(D.shape[0]).astype(D.dtype), rtol=1e-5, atol=1e-8
    )
    assert D.shape[0] == len(lem_sent) + 2


def test_distance_matrix_primitive(primitive):
    parser = SyntacticParser()
    D, lem_sent = parser.get_distance_matrix(primitive)
    D_ = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    assert np.allclose(D, D_, rtol=1e-5, atol=1e-8)
    assert " ".join(lem_sent).strip() == primitive.lower()
