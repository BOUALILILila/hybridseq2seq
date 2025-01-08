import pytest

from hybridseq2seq.models.encoder import EuclideanTranformerEncoderModel
from ..conftest import prepare_config_and_inputs


def test_euclidean_transformer_encoder(prepare_config_and_inputs):
    encoder = EuclideanTranformerEncoderModel(prepare_config_and_inputs[0])
    out = encoder(**prepare_config_and_inputs[1])
    assert out[0].shape == prepare_config_and_inputs[1]["input_ids"].shape + (
        prepare_config_and_inputs[0].hidden_size,
    )
