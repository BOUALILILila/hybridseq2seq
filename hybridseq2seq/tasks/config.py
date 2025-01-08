from dataclasses import dataclass, field

import yaml

from ..utils import get_logger

logger = get_logger(__name__)


class DictObj(object):
    def __init__(self, d):
        self.d = d
        for key in d:
            if isinstance(d[key], dict):
                self.__dict__[key] = DictObj(d[key])
            else:
                self.__dict__[key] = d[key]

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return None

    def __repr__(self) -> str:
        return self.d.__repr__()


def load_config_from_yaml(path):
    logger.info(f"Loading configuration from path: {path}")
    with open(path) as f:
        all_config = yaml.safe_load(f.read())
    return DictObj(all_config)


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
    # is_decoder: bool = field(
    #     default=False, metadata={
    #         "help": "Indicate if we are in a decoder transformer layer if true, else an enocder layer (full attention span)"
    #     }
    # )
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
