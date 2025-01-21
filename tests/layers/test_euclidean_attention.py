import pytest

from hybridseq2seq.layers.euclidean_attention import (
    EuclideanAttention,
    MultiHeadAttention,
)
from .conftest import (
    num_heads_wrong,
    get_config,
    prepare_config_and_hidden_states,
)


def test_raise_error_number_attn_heads(num_heads_wrong):
    with pytest.raises(ValueError):
        MultiHeadAttention(num_heads_wrong)


def test_attn_head_size(get_config):
    attn = MultiHeadAttention(get_config)
    assert (
        attn.attention_head_size
        == get_config.hidden_size / get_config.num_attention_heads
    )


def test_relative_position_embedding_attn(get_config):
    attn = MultiHeadAttention(get_config, position_embedding_type="relative_key")
    assert attn.distance_embedding is not None


def test_absolute_position_attn(get_config):
    attn = MultiHeadAttention(get_config, position_embedding_type="learned")
    assert not hasattr(attn, "distance_embedding")


def test_attention_output_hidden_shape(prepare_config_and_hidden_states):
    config = prepare_config_and_hidden_states[0]
    attn = EuclideanAttention(config)
    out = attn(**prepare_config_and_hidden_states[1])
    assert out[0].shape == prepare_config_and_hidden_states[1]["hidden_states"].shape


def test_attention_output_attention_scores(prepare_config_and_hidden_states):
    config = prepare_config_and_hidden_states[0]
    attn = EuclideanAttention(config)
    out = attn(
        **prepare_config_and_hidden_states[1],
        output_attentions=True,
    )
    batch_size, seq_length = prepare_config_and_hidden_states[1]["hidden_states"].shape[
        :2
    ]
    n_heads = config.num_attention_heads
    assert len(out) == 2
    assert out[1].shape == (batch_size, n_heads, seq_length, seq_length)
