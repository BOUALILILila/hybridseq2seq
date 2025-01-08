from typing import Optional, Tuple

import torch
from torch import nn

from .hyperbolic_attention import HyperbolicAttention
from .hyperbolic_transformer_layers import (
    HyperbolicTransformerIntermediate,
    HyperbolicTransformerOutput,
)

from ..utils import get_logger

logger = get_logger(__name__)


class HyperbolicTransformerEncoderLayer(nn.Module):
    def __init__(self, config, manifold, is_decoder: bool = False):
        super().__init__()
        self.manifold = manifold
        self.is_decoder = is_decoder

        self.self_attention = HyperbolicAttention(config, manifold, is_decoder=True)
        if self.is_decoder:
            self.cross_attention = HyperbolicAttention(
                config, manifold, is_decoder=True
            )
        self.intermediate = HyperbolicTransformerIntermediate(config, manifold)
        self.output = HyperbolicTransformerOutput(config, manifold)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Hyperbolic Self-attention
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

        # Hyperbolic Cross_attention
        if self.is_decoder:
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

        # Hyperbolic intermediate and output layer
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class HyperbolicTransformerEncoder(nn.Module):
    # config: num_hidden_layers
    def __init__(self, config, manifold, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.layers = nn.ModuleList(
            [
                HyperbolicTransformerEncoderLayer(
                    config, self.manifold, is_decoder=is_decoder
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
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
