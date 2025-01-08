import pytest
import torch

from hybridseq2seq.layers import HMDSEmbeddingFromCrossDistances
from hybridseq2seq.manifolds import PoincareBall

from ..conftest import floats_tensor, get_config


@pytest.fixture
def inputs(get_config):
    config = get_config
    batch_size = 1
    seq_length = 5
    config.semantic_parser_epsilon = 0.01
    config.semantic_parser_relation_edge_weight = 0.5
    config.add_sos_token = False
    config.default_max_distance = 7.0
    config.hyperbolic_scale = 1.0

    source_token_seqs = ["emma eat a hammer .".split()]

    vocab = dict()
    inv_vocab = dict()
    inv_vocab[0] = "<s>"
    vocab["<s>"] = 0
    i = 1
    for s in source_token_seqs:
        for w in s:
            if w not in vocab:
                vocab[w] = i
                inv_vocab[i] = w
                i += 1

    target_tokens = "eat . agent ( x _ 1 , Emma ) AND eat . theme ( x _ 1 , x _ 3 ) AND hammer ( x _ 3 )".split()
    for w in target_tokens:
        if w not in vocab:
            vocab[w] = i
            inv_vocab[i] = w
            i += 1

    inv_vocab[i] = "<pad>"
    vocab["<pad>"] = i

    config.sos_idx = vocab["<s>"]

    target_input_ids = torch.tensor(
        [[vocab["<s>"]] + [vocab[w] for w in target_tokens]]
    )

    source_attention_mask = torch.ones(batch_size, seq_length + 1).float()
    source_attention_mask[:, -1] = torch.zeros(batch_size, 1)

    decode = lambda x: [[inv_vocab[w] for w in token_ids] for token_ids in x.tolist()]

    source_distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 3.0, 2.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 1.0, 1.0, 0.0],
            [3.0, 2.0, 0.0, 1.0, 3.0, 0.0],
            [2.0, 1.0, 1.0, 0.0, 2.0, 0.0],
            [2.0, 1.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    return (
        config,
        target_input_ids,
        source_attention_mask,
        source_token_seqs,
        source_distance_matrix.unsqueeze(0),
        decode,
    )


def test_embed(inputs):
    config = inputs[0]
    hyperbolic_embedding = HMDSEmbeddingFromCrossDistances(
        epsilon=config.semantic_parser_epsilon,
        relation_edge_weight=config.semantic_parser_relation_edge_weight,
        default_max_distance=config.default_max_distance,
        add_sos_token=config.add_sos_token,
        hyperbolic_embedding_size=config.hyperbolic_hidden_size,
        hmds_scale=config.hyperbolic_scale,
        manifold=PoincareBall(),
    )

    target_ids = inputs[1]
    encoder_attention_mask = inputs[2]
    source_token_seqs = inputs[3]
    source_distance_matrix = inputs[4]
    decode = inputs[5]

    cross_distance_matrices_i = source_distance_matrix
    hyperbolic_attention_mask_i = torch.ones_like(target_ids[:, :1])

    for i in range(1, target_ids.shape[-1] + 1):
        previous_token_seqs_i = decode(target_ids[:, :i])

        previous_hyperbolic_attention_mask = torch.ones_like(target_ids[:, :i]).float()
        previous_hyperbolic_attention_mask[:, :-1] = hyperbolic_attention_mask_i

        (
            hyperbolic_cross_embeddings_i,
            hyperbolic_attention_mask_i,
            cross_distance_matrices_i,
        ) = hyperbolic_embedding.embed(
            step=i,
            previous_cross_distance_matrix=cross_distance_matrices_i,  # (bs, step-1, step-1)
            source_distance_matrix=source_distance_matrix,  # (bs, src_seq_length, src_seq_length)
            source_tokens_list=source_token_seqs,  # (bs, src_seq_length)
            source_attention_mask=encoder_attention_mask,
            previous_target_tokens_list=previous_token_seqs_i,  # (bs, step)
            previous_target_hyperbolic_attention_mask=previous_hyperbolic_attention_mask,  # (bs, step)
        )

        assert hyperbolic_cross_embeddings_i.shape == (
            encoder_attention_mask.shape[0],
            encoder_attention_mask.shape[1] + i,
            config.hyperbolic_hidden_size,
        )

        reverse_mask = 1 - torch.cat(
            [encoder_attention_mask[0], hyperbolic_attention_mask_i[0]]
        )
        assert torch.allclose(
            hyperbolic_cross_embeddings_i[0]
            .masked_select(reverse_mask.view(-1, 1).bool())
            .view(-1, config.hyperbolic_hidden_size),
            torch.zeros(reverse_mask.sum().int().item(), config.hyperbolic_hidden_size),
        )
    # print(hyperbolic_cross_embeddings_i)
    # print(cross_distance_matrices_i)
    # print(hyperbolic_attention_mask_i)
