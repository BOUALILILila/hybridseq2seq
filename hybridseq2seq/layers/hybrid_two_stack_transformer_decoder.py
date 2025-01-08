from typing import Optional, Tuple

import torch
from torch import nn

from .euclidean_attention import EuclideanAttention
from .hyperbolic_attention import HyperbolicAttention
from .transformer_layers import TransformerIntermediate, TransformerOutput
from .hyperbolic_transformer_layers import (
    HyperbolicTransformerIntermediate,
    HyperbolicTransformerOutput,
)

from ..utils import get_logger, ACT2FN

logger = get_logger(__name__)


class HybridTwoStackTransformerDecoderLayer(nn.Module):
    def __init__(self, config, manifold):
        super().__init__()
        # Euclidean components
        self.euclidean_self_attention = EuclideanAttention(config, is_decoder=True)
        self.euclidean_cross_attention = EuclideanAttention(config, is_decoder=True)
        self.euclidean_intermediate = TransformerIntermediate(config)
        self.euclidean_output = TransformerOutput(config)
        # Hyperbolic components
        self.hyperbolic_self_attention = HyperbolicAttention(
            config, manifold, is_decoder=True
        )
        self.hyperbolic_cross_attention = HyperbolicAttention(
            config, manifold, is_decoder=True
        )
        self.hyperbolic_intermediate = HyperbolicTransformerIntermediate(
            config, manifold
        )
        self.hyperbolic_output = HyperbolicTransformerOutput(config, manifold)

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        hyperbolic_hidden_states: torch.Tensor,
        euclidean_attention_mask: Optional[torch.Tensor] = None,
        hyperbolic_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        euclidean_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        euclidean_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Euclidean Self-attention
        self_attention_outputs = self.euclidean_self_attention(
            euclidean_hidden_states,
            euclidean_attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        euclidean_attention_output = self_attention_outputs[0]
        euclidean_outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights
        # Hyperbolic Self-attention
        hyperbolic_self_attention_outputs = self.hyperbolic_self_attention(
            hyperbolic_hidden_states,
            hyperbolic_attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        hyperbolic_attention_output = hyperbolic_self_attention_outputs[0]
        hyperbolic_outputs = hyperbolic_self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights
        # Euclidean Cross_attention
        cross_attention_outputs = self.euclidean_cross_attention(
            euclidean_attention_output,
            euclidean_attention_mask,
            head_mask,
            euclidean_encoder_hidden_states,
            euclidean_encoder_attention_mask,
            # cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        euclidean_attention_output = cross_attention_outputs[0]
        euclidean_outputs = (
            euclidean_outputs + cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights
        # Hyperbolic Cross_attention
        hyperbolic_cross_attention_outputs = self.hyperbolic_cross_attention(
            hyperbolic_attention_output,
            hyperbolic_attention_mask,
            head_mask,
            hyperbolic_encoder_hidden_states,
            hyperbolic_encoder_attention_mask,
            # cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        hyperbolic_attention_output = hyperbolic_cross_attention_outputs[0]
        hyperbolic_outputs = (
            hyperbolic_outputs + hyperbolic_cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights
        # Euclidean intermediate and output layer
        euclidean_intermediate_output = self.euclidean_intermediate(
            euclidean_attention_output
        )
        euclidean_layer_output = self.euclidean_output(
            euclidean_intermediate_output, euclidean_attention_output
        )
        euclidean_outputs = (euclidean_layer_output,) + euclidean_outputs
        # Hyperbolic intermediate and output layer
        hyperbolic_intermediate_output = self.hyperbolic_intermediate(
            hyperbolic_attention_output
        )
        hyperbolic_layer_output = self.hyperbolic_output(
            hyperbolic_intermediate_output, hyperbolic_attention_output
        )
        hyperbolic_outputs = (hyperbolic_layer_output,) + hyperbolic_outputs

        return euclidean_outputs, hyperbolic_outputs


class HybridTwoStackTransformerDecoder(nn.Module):
    # config: num_hidden_layers
    def __init__(self, config, manifold):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.layers = nn.ModuleList(
            [
                HybridTwoStackTransformerDecoderLayer(config, self.manifold)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        hyperbolic_hidden_states: torch.Tensor,
        euclidean_attention_mask: Optional[torch.Tensor] = None,
        hyperbolic_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        euclidean_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        euclidean_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        all_euclidean_hidden_states, all_hyperbolic_hidden_states = (
            (),
            () if output_hidden_states else None,
        )
        all_euclidean_attentions, all_hyperbolic_attentions = (
            (),
            () if output_attentions else None,
        )

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_euclidean_hidden_states = all_euclidean_hidden_states + (
                    euclidean_hidden_states,
                )
                all_hyperbolic_hidden_states = all_hyperbolic_hidden_states + (
                    hyperbolic_hidden_states,
                )

            layer_head_mask = head_mask[i] if head_mask is not None else None

            euclidean_layer_outputs, hyperbolic_layer_outputs = layer(
                euclidean_hidden_states,
                hyperbolic_hidden_states,
                euclidean_attention_mask,
                hyperbolic_attention_mask,
                layer_head_mask,
                euclidean_encoder_hidden_states,
                hyperbolic_encoder_hidden_states,
                euclidean_encoder_attention_mask,
                hyperbolic_encoder_attention_mask,
                past_key_value,
                output_attentions=output_attentions,
            )

            euclidean_hidden_states = euclidean_layer_outputs[0]
            hyperbolic_hidden_states = hyperbolic_layer_outputs[0]

            if output_attentions:
                all_euclidean_attentions = all_euclidean_attentions + (
                    euclidean_layer_outputs[1],
                )
                all_hyperbolic_attentions = all_hyperbolic_attentions + (
                    hyperbolic_layer_outputs[1],
                )

        if output_hidden_states:
            all_euclidean_hidden_states = all_euclidean_hidden_states + (
                euclidean_hidden_states,
            )
            all_hyperbolic_hidden_states = all_hyperbolic_hidden_states + (
                hyperbolic_hidden_states,
            )

        return tuple(
            v
            for v in [
                euclidean_hidden_states,
                hyperbolic_hidden_states,
                all_euclidean_hidden_states,
                all_hyperbolic_hidden_states,
                all_euclidean_attentions,
                all_hyperbolic_attentions,
            ]
            if v is not None
        )


class TwoStackDecoderPredictionHeadCatTransform(nn.Module):
    def __init__(self, config, manifold):
        super().__init__()
        self.manifold = manifold
        self.dense = nn.Linear(
            config.hidden_size + config.hyperbolic_hidden_size, config.hidden_size
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        hyperbolic_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = torch.cat(
            [euclidean_hidden_states, self.manifold.logmap0(hyperbolic_hidden_states)],
            dim=2,
        )
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TwoStackDecoderPredictionHead(nn.Module):
    def __init__(self, config, manifold) -> None:
        super().__init__()
        self.manifold = manifold
        self.transform = TwoStackDecoderPredictionHeadCatTransform(
            config, manifold=self.manifold
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        hyperbolic_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.transform(
            euclidean_hidden_states, hyperbolic_hidden_states
        )
        hidden_states = self.decoder(hidden_states)
        return hidden_states
