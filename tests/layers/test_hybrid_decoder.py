import pytest

import torch

from hybridseq2seq.layers import HybridTransformerDecoder
from hybridseq2seq.manifolds import PoincareBall
from .conftest import prepare_config_and_hidden_states


@pytest.fixture
def hybrid_decoder_inputs(prepare_config_and_hidden_states):
    manifold = PoincareBall()
    config = prepare_config_and_hidden_states[0]
    euclidean_hs = prepare_config_and_hidden_states[1]["hidden_states"]
    euclidean_mask = torch.ones_like(
        prepare_config_and_hidden_states[1]["attention_mask"]
    )
    hyperbolic_hs = manifold.expmap0(euclidean_hs)
    hyperbolic_mask = prepare_config_and_hidden_states[1]["attention_mask"].squeeze(-2)
    encoder_euclidean_hs = prepare_config_and_hidden_states[1]["hidden_states"]
    encoder_euclidean_mask = torch.ones_like(
        prepare_config_and_hidden_states[1]["attention_mask"]
    )
    encoder_hyperbolic_hs = manifold.expmap0(euclidean_hs)
    encoder_hyperbolic_mask = prepare_config_and_hidden_states[1][
        "attention_mask"
    ].squeeze(-2)
    return (
        manifold,
        config,
        (
            euclidean_hs,
            euclidean_mask,
            hyperbolic_hs,
            hyperbolic_mask,
        ),
        (
            encoder_euclidean_hs,
            encoder_euclidean_mask,
            encoder_hyperbolic_hs,
            encoder_hyperbolic_mask,
        ),
    )


def test_hybrid_transformer_decoder_block(hybrid_decoder_inputs):
    manifold = hybrid_decoder_inputs[0]
    config = hybrid_decoder_inputs[1]
    decoder_inputs = hybrid_decoder_inputs[2]
    encoder_outputs = hybrid_decoder_inputs[3]

    config.combine_euclidean_hyperbolic_attention_scores = True
    decoder = HybridTransformerDecoder(config, manifold)
    out = decoder(
        euclidean_hidden_states=decoder_inputs[0],
        hyperbolic_hidden_states=decoder_inputs[2],
        euclidean_attention_mask=decoder_inputs[1],
        hyperbolic_attention_mask=decoder_inputs[3],
        head_mask=None,
        euclidean_encoder_hidden_states=encoder_outputs[0],
        hyperbolic_encoder_hidden_states=encoder_outputs[2],
        euclidean_encoder_attention_mask=encoder_outputs[1],
        hyperbolic_encoder_attention_mask=encoder_outputs[3],
    )
    assert out[0].shape == decoder_inputs[0].shape

    config.combine_euclidean_hyperbolic_attention_scores = False
    decoder = HybridTransformerDecoder(config, manifold)
    out = decoder(
        euclidean_hidden_states=decoder_inputs[0],
        hyperbolic_hidden_states=decoder_inputs[2],
        euclidean_attention_mask=decoder_inputs[1],
        hyperbolic_attention_mask=decoder_inputs[3],
        head_mask=None,
        euclidean_encoder_hidden_states=encoder_outputs[0],
        hyperbolic_encoder_hidden_states=encoder_outputs[2],
        euclidean_encoder_attention_mask=encoder_outputs[1],
        hyperbolic_encoder_attention_mask=encoder_outputs[3],
        output_attentions=True,
    )
    assert len(out) == 3
    assert len(out[-1]) == 2
    assert out[0].shape == decoder_inputs[0].shape

    config.combine_euclidean_hyperbolic_attention_scores = True
    decoder = HybridTransformerDecoder(config, manifold)
    out = decoder(
        euclidean_hidden_states=decoder_inputs[0],
        hyperbolic_hidden_states=decoder_inputs[2],
        euclidean_attention_mask=decoder_inputs[1],
        hyperbolic_attention_mask=None,
        head_mask=None,
        euclidean_encoder_hidden_states=encoder_outputs[0],
        hyperbolic_encoder_hidden_states=None,
        euclidean_encoder_attention_mask=encoder_outputs[1],
        hyperbolic_encoder_attention_mask=encoder_outputs[3],
        output_attentions=True,
    )
    assert len(out[-1]) == 1  # only hyperbolic self attn scores, no cross attn
    assert out[0].shape == decoder_inputs[0].shape

    config.combine_euclidean_hyperbolic_attention_scores = False
    decoder = HybridTransformerDecoder(config, manifold)
    out = decoder(
        euclidean_hidden_states=decoder_inputs[0],
        hyperbolic_hidden_states=None,
        euclidean_attention_mask=decoder_inputs[1],
        hyperbolic_attention_mask=None,
        head_mask=None,
        euclidean_encoder_hidden_states=encoder_outputs[0],
        hyperbolic_encoder_hidden_states=encoder_outputs[2],
        euclidean_encoder_attention_mask=encoder_outputs[1],
        hyperbolic_encoder_attention_mask=encoder_outputs[3],
        output_attentions=True,
    )
    assert len(out[-1]) == 0  # No hyperbolic attention
    assert out[0].shape == decoder_inputs[0].shape
