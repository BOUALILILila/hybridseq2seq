import random
from dataclasses import dataclass, field

import pytest
import torch

from hybridseq2seq.tasks.config import Config

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return (
        torch.tensor(data=values, dtype=torch.long, device=torch_device)
        .view(shape)
        .contiguous()
    )


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None) * 1.0
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return (
        torch.tensor(data=values, dtype=torch.float, device=torch_device)
        .view(shape)
        .contiguous()
    )


@dataclass
class Config:
    hidden_size: int = field(
        default=768, metadata={"help": "Euclidean Transformer hidden size"}
    )
    hyperbolic_hidden_size: int = field(
        default=768, metadata={"help": "Hyperbolic Transformer hidden size"}
    )
    hyperbolic_scale: float = field(
        default=1.0, metadata={"help": "Hyperbolic scale factor for hmds"}
    )
    num_attention_heads: int = field(
        default=8, metadata={"help": "Total attention heads"}
    )
    hyperbolic_num_attention_heads: int = field(
        default=8, metadata={"help": "Total attention heads in hyperbolic attention"}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout rate applied inside attention"}
    )
    hyperbolic_attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied inside hyperbolic attention"},
    )
    max_position_embeddings: int = field(
        default=256, metadata={"help": "Maximum input sequence length"}
    )

    layer_norm_eps: float = field(
        default=1e-12, metadata={"help": "Layer norm layers initialization"}
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout rate applied on hidden states"}
    )
    hyperbolic_hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied on hyperbolic hidden states"},
    )
    num_hidden_layers: int = field(
        default=2, metadata={"help": "Number of layers used in the encoder/decoder"}
    )
    intermediate_size: int = field(
        default=768,
        metadata={"help": "Tensor dimension in the FFN of the transormer layers"},
    )
    hidden_act: str = field(
        default="relu",
        metadata={"help": "Hidden activation for the FFN in transformer layers"},
    )
    vocab_size: int = field(
        default=None, metadata={"help": "Vocabulary size (total number of tokens)"}
    )
    pad_idx: int = field(
        default=0, metadata={"help": "Padding token index in the vocabulary"}
    )
    position_embedding_type: str = field(
        default="learned",
        metadata={
            "help": "Type of the position embeddings: ['learned', 'sinusoidal', 'relative_key', 'relative_key_query']"
        },
    )
    scale_mode: str = field(
        default="none",
        metadata={
            "help": "Scaling method for input embeddings if position embeddings are 'sinusoidal': ['none','down', 'up']"
        },
    )
    init_embeddings: str = field(
        default="normal",
        metadata={
            "help": "Input embedding initialization: ['normal', 'xavier', 'kaiming']"
        },
    )
    combine_euclidean_hyperbolic_attention_scores: bool = field(
        default=False,
        metadata={
            "help": "Combine the hyperbolic attention sscores with the euclidean attention scores in decoder attention if True, else concatenate the resulting value vectors"
        },
    )
    default_max_distance: bool = field(
        default=30.0,
        metadata={"help": "Default maximum distance for the hyperbolic embeddings"},
    )

    def __post_init__(self):
        # super().__post_init__()
        assert self.init_embeddings in (
            "normal",
            "xavier",
            "kaiming",
        ), f"init_embeddings must be in ['normal', 'xavier', 'kaiming'] but {self.init_embeddings} was given"
        assert self.scale_mode in (
            "none",
            "down",
            "up",
        ), f"scale_mode must be in ['none','down', 'up'] but {self.scale_mode} was given"


@pytest.fixture
def get_config():
    data = {
        "hidden_size": 64,
        "hyperbolic_hidden_size": 64,
        "init_embeddings": "normal",
        "vocab_size": 50,
        "default_max_distance": 7.0,
    }
    return Config(**data)


@pytest.fixture
def prepare_config_and_inputs(get_config):
    batch_size = 2
    seq_length = 128
    return (
        get_config,
        {
            "input_ids": ids_tensor(
                shape=[batch_size, seq_length], vocab_size=get_config.vocab_size
            ),
            "attention_mask": random_attention_mask(shape=[batch_size, seq_length]),
        },
    )


@pytest.fixture
def prepare_config_and_hidden_states(get_config):
    batch_size = 2
    seq_length = 128
    hidden_size = get_config.hidden_size
    mask = random_attention_mask(shape=[batch_size, seq_length])[:, None, None, :]
    return (
        get_config,
        {
            "hidden_states": floats_tensor(shape=[batch_size, seq_length, hidden_size]),
            "attention_mask": (1.0 - mask) * torch.finfo(mask.dtype).min,
        },
    )
