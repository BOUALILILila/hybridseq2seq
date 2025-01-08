import torch

from hybridseq2seq.manifolds import PoincareBall

from hybridseq2seq.layers.hyperbolic_attention import (
    HyperbolicAttention,
    hyperbolic_distance_scores,
)

from .conftest import prepare_config_and_hidden_states


def test_attention_output_hidden(prepare_config_and_hidden_states):
    config = prepare_config_and_hidden_states[0]
    assert config.hidden_size == config.hyperbolic_hidden_size
    manifold = PoincareBall()

    attn = HyperbolicAttention(config, manifold)

    # Project dummy tensor into the manifold
    hidden_states = manifold.expmap0(
        prepare_config_and_hidden_states[1]["hidden_states"]
    )
    attn_mask = prepare_config_and_hidden_states[1]["attention_mask"]
    out = attn(hidden_states, attn_mask)

    assert out[0].shape == prepare_config_and_hidden_states[1]["hidden_states"].shape
    assert manifold._check_point_on_manifold(out[0])


def test_hyperbolic_distance_scoring():
    manifold = PoincareBall()
    hidden_states = torch.rand(2, 128, 64)
    hidden_states = manifold.expmap0(hidden_states)

    attn_scores = hyperbolic_distance_scores(manifold, hidden_states, normalize=False)
    assert attn_scores.shape == hidden_states.shape[:-1] + (hidden_states.shape[-2],)
    self_dist = torch.diagonal(attn_scores, dim1=-2, dim2=-1)
    assert torch.allclose(self_dist, torch.zeros_like(self_dist), rtol=1e-5, atol=1e-8)

    attn_scores = hyperbolic_distance_scores(manifold, hidden_states, normalize=True)
    assert attn_scores.shape == hidden_states.shape[:-1] + (hidden_states.shape[-2],)
    sum_scores = attn_scores.sum(-1)
    assert torch.allclose(sum_scores, torch.ones_like(sum_scores), rtol=1e-5, atol=1e-8)
