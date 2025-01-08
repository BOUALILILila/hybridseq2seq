import pytest

from hybridseq2seq.layers.transformer_encoder import TransformerEncoder
from .conftest import prepare_config_and_hidden_states


def test_transformer_encoder_block(prepare_config_and_hidden_states):
    trans_block = TransformerEncoder(prepare_config_and_hidden_states[0])
    out = trans_block(**prepare_config_and_hidden_states[1])
    assert out[0].shape == prepare_config_and_hidden_states[1]["hidden_states"].shape
