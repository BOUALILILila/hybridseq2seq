from typing import Optional, Tuple

import torch
from torch import nn

from .euclidean_attention import EuclideanAttention
from .transformer_layers import TransformerIntermediate, TransformerOutput

from ..utils import get_logger

logger = get_logger(__name__)

# class TransformerEncoderIntermediate(nn.Module):
#     # config: intermediate_size, hidden_size, hidden_act
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         # default: relu
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class TransformerEncoderOutput(nn.Module):
#     # config: intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # self.seq_len_dim = 1

        self.attention = EuclideanAttention(config)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # transformers/pytorch_utils.py use function apply_chunking_to_forward
        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        return outputs

    # def feed_forward_chunk(self, attention_output):
    #     intermediate_output = self.intermediate(attention_output)
    #     layer_output = self.output(intermediate_output, attention_output)
    #     return layer_output


class TransformerEncoder(nn.Module):
    # config: num_hidden_layers
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer(
                hidden_states, attention_mask, layer_head_mask, output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )
