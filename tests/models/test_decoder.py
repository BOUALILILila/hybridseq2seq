import pytest
import torch

from hybridseq2seq.manifolds import PoincareBall
from hybridseq2seq.models.decoder import HybridDecoderModel

from ..conftest import floats_tensor, get_config


@pytest.fixture
def hybrid_decoder_inputs(get_config):
    config = get_config
    batch_size = 1
    seq_length = 5
    config.semantic_parser_epsilon = 0.01
    config.semantic_parser_relation_edge_weight = 0.5
    config.add_sos_token = False
    config.tie_input_output_emb = True

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
    labels = torch.tensor([[vocab[w] for w in target_tokens] + [vocab["<pad>"]]])
    encoder_hidden_states = floats_tensor(
        shape=[batch_size, seq_length, config.hidden_size]
    )
    source_attention_mask = torch.ones(batch_size, seq_length)

    decode = lambda x: [[inv_vocab[w] for w in token_ids] for token_ids in x.tolist()]

    source_distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 3.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 1.0, 3.0],
            [2.0, 1.0, 1.0, 0.0, 2.0],
            [2.0, 1.0, 3.0, 2.0, 0.0],
        ]
    )
    return (
        config,
        target_input_ids,
        encoder_hidden_states,
        source_attention_mask,
        source_token_seqs,
        source_distance_matrix.unsqueeze(0),
        decode,
        labels,
    )


def test_hybrid_decoder(hybrid_decoder_inputs):
    config = hybrid_decoder_inputs[0]
    decoder = HybridDecoderModel(config)

    # use input_ids
    input_ids = hybrid_decoder_inputs[1]
    decoder_output = decoder(
        input_ids=input_ids,
        encoder_hidden_states=hybrid_decoder_inputs[2],
        encoder_attention_mask=hybrid_decoder_inputs[3],
        source_token_seqs=hybrid_decoder_inputs[4],
        source_distance_matrix=hybrid_decoder_inputs[5],
        decode=hybrid_decoder_inputs[6],
    )
    assert decoder_output.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        config.vocab_size,
    )

    # use labels
    decoder_output = decoder(
        labels=hybrid_decoder_inputs[7],
        encoder_hidden_states=hybrid_decoder_inputs[2],
        encoder_attention_mask=hybrid_decoder_inputs[3],
        source_token_seqs=hybrid_decoder_inputs[4],
        source_distance_matrix=hybrid_decoder_inputs[5],
        decode=hybrid_decoder_inputs[6],
    )
    assert decoder_output.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        config.vocab_size,
    )


def test_hybrid_decoder_eh_score_comb(hybrid_decoder_inputs):
    config = hybrid_decoder_inputs[0]
    config.combine_euclidean_hyperbolic_attention_scores = True
    decoder = HybridDecoderModel(config)

    # check tie emb
    input_ids = hybrid_decoder_inputs[1]

    decoder_output = decoder(
        input_ids=input_ids,
        encoder_hidden_states=hybrid_decoder_inputs[2],
        encoder_attention_mask=hybrid_decoder_inputs[3],
        source_token_seqs=hybrid_decoder_inputs[4],
        source_distance_matrix=hybrid_decoder_inputs[5],
        decode=hybrid_decoder_inputs[6],
    )
    assert decoder_output.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        config.vocab_size,
    )


def test_hybrid_decoder_one_step(hybrid_decoder_inputs):
    config = hybrid_decoder_inputs[0]
    decoder = HybridDecoderModel(config)

    assert id(decoder.embeddings.word_embeddings.weight) == id(
        decoder.prediction_head.decoder.weight
    )

    input_ids = hybrid_decoder_inputs[1]

    decoder_outputs = decoder.forward_one_step(
        input_ids=input_ids[:, :1],
        encoder_hidden_states=hybrid_decoder_inputs[2],
        encoder_attention_mask=hybrid_decoder_inputs[3],
        source_token_seqs=hybrid_decoder_inputs[4],
        cross_distance_matrix=hybrid_decoder_inputs[5],
        source_distance_matrix=hybrid_decoder_inputs[5],
        decode=hybrid_decoder_inputs[6],
        hyperbolic_attention_mask=None,
    )
    output = decoder_outputs[0]
    cross_dist_mat = decoder_outputs[2]
    assert output.shape == (input_ids.shape[0], 1, config.vocab_size)
    assert cross_dist_mat.shape == (
        input_ids.shape[0],
        hybrid_decoder_inputs[3].shape[1] + 1,
        hybrid_decoder_inputs[3].shape[1] + 1,
    )

    # next step
    previous_hyperbolic_attention_mask = torch.ones_like(input_ids[:, :2]).float()
    previous_hyperbolic_attention_mask[:, -1:] = decoder_outputs[1]

    decoder_outputs = decoder.forward_one_step(
        input_ids=input_ids[:, :2],
        encoder_hidden_states=hybrid_decoder_inputs[2],
        encoder_attention_mask=hybrid_decoder_inputs[3],
        source_token_seqs=hybrid_decoder_inputs[4],
        cross_distance_matrix=cross_dist_mat,
        source_distance_matrix=hybrid_decoder_inputs[5],
        decode=hybrid_decoder_inputs[6],
        hyperbolic_attention_mask=previous_hyperbolic_attention_mask,
    )
    output = decoder_outputs[0]
    cross_dist_mat = decoder_outputs[2]
    assert output.shape == (input_ids.shape[0], 2, config.vocab_size)
    assert cross_dist_mat.shape == (
        input_ids.shape[0],
        hybrid_decoder_inputs[3].shape[1] + 2,
        hybrid_decoder_inputs[3].shape[1] + 2,
    )
