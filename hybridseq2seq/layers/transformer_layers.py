import math

import numpy as np
import torch
from torch import nn

from ..utils import ACT2FN


class TransformerIntermediate(nn.Module):
    # config: intermediate_size, hidden_size, hidden_act
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # default: relu
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerOutput(nn.Module):
    # config: intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


class TransformerEmbeddings(nn.Module):
    """Construct the input embeddings from word and position embeddings."""

    def __init__(self, config) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_idx
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

        self.init_weights(config)

        self.scale = 1.0
        if self.position_embedding_type == "sinusoidal":
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings,
                dim=config.hidden_size,
                out=self.position_embeddings.weight,
            )
            if config.scale_mode == "down":
                self.position_embeddings.weight *= 1.0 / math.sqrt(config.hidden_size)
            elif config.scale_mode == "up":
                self.scale = math.sqrt(config.hidden_size)
            elif config.scale_mode != "none":
                raise ValueError(
                    f"Unknown scale_mode {config.scale_mode}. Expected one of ['none','down', 'up']"
                )

    def init_weights(self, config):
        if config.init_embeddings == "xavier":
            torch.nn.init.xavier_uniform_(self.word_embeddings.weight)
        elif config.init_embeddings == "kaiming":
            torch.nn.init.kaiming_normal_(self.word_embeddings.weight)
        elif config.init_embeddings == "normal":
            torch.nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(
                f"Unknown init_embeddings mode {config.init_embeddings}. Expected one of ['normal', 'xavier', 'kaiming']"
            )

        if self.word_embeddings.padding_idx is not None:
            self.word_embeddings.weight.data[self.word_embeddings.padding_idx].zero_()

        if self.position_embedding_type != "sinusoidal":
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        embeddings = self.word_embeddings(input_ids)

        if self.position_embedding_type in ("absolute", "sinusoidal"):
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings * self.scale + position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
        return embeddings
