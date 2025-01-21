from typing import Optional, Tuple

import torch
from torch import nn

from ..utils import ACT2FN, get_logger
from .euclidean_attention import EuclideanAttention
from .transformer_layers import TransformerIntermediate, TransformerOutput

logger = get_logger(__name__)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Euclidean components
        self.self_attention = EuclideanAttention(config, is_decoder=True)
        self.cross_attention = EuclideanAttention(config, is_decoder=True)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Euclidean Self-attention with euclidean mulit-head attention
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # Euclidean Cross_attention
        cross_attention_outputs = self.cross_attention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            # cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = (
            outputs + cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights

        # Euclidean intermediate and output layer
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class TransformerDecoder(nn.Module):
    # config: num_hidden_layers
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Decoder
        assert (
            encoder_hidden_states is not None
        ), "euclidean_encoder_hidden_states=None, cannot perform decoding"

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_attentions,
            ]
            if v is not None
        )
