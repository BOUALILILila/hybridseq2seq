import torch

from hybridseq2seq.models.base import BaseModule


def test_extend_mask():
    model = BaseModule()

    # encoder: self attention mask (bs, src_seq_len)
    attn_mask = torch.tensor([[1, 1, 1, 0.0], [1, 1, 0.0, 0.0]])
    extended_attn_mask = model.get_extended_attention_mask(
        attn_mask, input_shape=attn_mask.shape
    )
    assert extended_attn_mask.shape == (attn_mask.shape[0], 1, 1, attn_mask.shape[1])

    # decoder: euclidean self attention mask (bs, tgt_seq_len)
    attn_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0.0, 0.0]])
    extended_attn_mask = model.get_extended_attention_mask(
        attn_mask, input_shape=attn_mask.shape, causal=True
    )
    assert extended_attn_mask.shape == (
        attn_mask.shape[0],  # bs
        1,  # num_heads
        attn_mask.shape[1],  # tgt_seq_len
        attn_mask.shape[1],  # tgt_seq_len
    )

    # decoder: euclidean cross attention mask (bs, src_seq_len)
    attn_mask = torch.tensor([[1, 1, 1, 0.0], [1, 1, 0.0, 0.0]])
    extended_attn_mask = model.get_extended_attention_mask(
        attn_mask, input_shape=attn_mask.shape
    )
    assert extended_attn_mask.shape == (
        attn_mask.shape[0],  # bs
        1,  # num_heads
        1,  # tgt_seq_len
        attn_mask.shape[1],  # src_seq_len
    )


def test_extend_mask_hyperbolic():
    model = BaseModule()

    # decoder: hyperbolic self attention mask from (bs, tgt_seq_len)
    attn_mask = torch.tensor([[1, 1, 1, 0.0], [1, 1, 0.0, 0.0]])
    extended_attn_mask = model.get_extended_hyperbolic_attention_mask(
        attn_mask, input_shape=attn_mask.shape, causal=True
    )
    assert extended_attn_mask.shape == (
        attn_mask.shape[0],  # bs
        attn_mask.shape[1],  # tgt_seq_len
        attn_mask.shape[1],  # src_seq_len
    )

    # decoder: hyperbolic cross attention mask from (bs, src_seq_len)
    attn_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0.0, 0.0]])
    extended_attn_mask = model.get_extended_hyperbolic_attention_mask(
        attn_mask, input_shape=attn_mask.shape
    )
    assert extended_attn_mask.shape == (
        attn_mask.shape[0],  # bs
        1,  # tgt_seq_len
        attn_mask.shape[1],  # src_seq_len
    )
