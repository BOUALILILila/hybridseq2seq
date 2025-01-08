from hybridseq2seq.layers import TransformerEmbeddings

from .conftest import prepare_config_and_inputs


def test_transformer_embeddings(prepare_config_and_inputs):
    config = prepare_config_and_inputs[0]
    emb = TransformerEmbeddings(config)
    out = emb(prepare_config_and_inputs[1]["input_ids"])
    assert out.shape == prepare_config_and_inputs[1]["input_ids"].shape + (
        config.hidden_size,
    )
