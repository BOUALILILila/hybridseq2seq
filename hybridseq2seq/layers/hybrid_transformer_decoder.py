from typing import Optional, Tuple

import torch
from torch import nn

from .euclidean_attention import EuclideanAttention, HybridEuclideanHyperbolicAttention
from .hyperbolic_attention import hyperbolic_distance_scores
from .transformer_layers import TransformerIntermediate, TransformerOutput

from ..utils import get_logger, ACT2FN

logger = get_logger(__name__)

# from bert self attention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
from typing import Optional, Tuple

import math

import torch
from torch import nn

from ..utils import get_logger

logger = get_logger(__name__)


class HybridTransformerDecoderLayer(nn.Module):
    def __init__(self, config, attention_cls):
        super().__init__()
        # Euclidean components
        self.euclidean_self_attention = attention_cls(config, is_decoder=True)
        self.euclidean_cross_attention = attention_cls(config, is_decoder=True)
        self.euclidean_intermediate = TransformerIntermediate(config)
        self.euclidean_output = TransformerOutput(config)

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        euclidean_attention_mask: torch.Tensor,
        euclidean_encoder_hidden_states: torch.Tensor,
        euclidean_encoder_attention_mask: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_self_attention_scores: Optional[torch.Tensor] = None,
        hyperbolic_cross_attention_scores: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Euclidean Self-attention with euclidean mulit-head attention
        self_attention_outputs = self.euclidean_self_attention(
            euclidean_hidden_states,
            euclidean_attention_mask,
            head_mask,
            output_attentions=output_attentions,
            hyperbolic_attention_scores=hyperbolic_self_attention_scores,
        )
        euclidean_attention_output = self_attention_outputs[0]
        euclidean_outputs = self_attention_outputs[
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
            hyperbolic_attention_scores=hyperbolic_cross_attention_scores,
        )
        euclidean_attention_output = cross_attention_outputs[0]
        euclidean_outputs = (
            euclidean_outputs + cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights

        # Euclidean intermediate and output layer
        euclidean_intermediate_output = self.euclidean_intermediate(
            euclidean_attention_output
        )
        euclidean_layer_output = self.euclidean_output(
            euclidean_intermediate_output, euclidean_attention_output
        )
        euclidean_outputs = (euclidean_layer_output,) + euclidean_outputs

        return euclidean_outputs


class HybridTransformerDecoder(nn.Module):
    # config: num_hidden_layers
    def __init__(self, config, manifold):
        super().__init__()
        self.config = config
        self.manifold = manifold
        if self.config.combine_euclidean_hyperbolic_attention_scores:
            attn_cls = EuclideanAttention
        else:
            attn_cls = HybridEuclideanHyperbolicAttention
        self.layers = nn.ModuleList(
            [
                HybridTransformerDecoderLayer(config, attn_cls)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        euclidean_hidden_states: torch.Tensor,
        euclidean_attention_mask: torch.Tensor,
        euclidean_encoder_hidden_states: torch.Tensor,
        euclidean_encoder_attention_mask: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_hidden_states: Optional[torch.Tensor] = None,
        hyperbolic_attention_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        hyperbolic_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        all_euclidean_hidden_states = () if output_hidden_states else None
        all_euclidean_attentions, all_hyperbolic_attentions = (
            (),
            () if output_attentions else None,
        )

        # Decoder
        assert (
            euclidean_encoder_hidden_states is not None
        ), "euclidean_encoder_hidden_states=None, cannot to perform decoding"

        # Hyperbolic attention scores
        if hyperbolic_hidden_states is not None:
            # self attn scores
            hyperbolic_self_attention_scores = hyperbolic_distance_scores(
                self.manifold,
                hyperbolic_hidden_states,
                hyperbolic_attention_mask,
                normalize=(
                    not self.config.combine_euclidean_hyperbolic_attention_scores
                ),
            )
            if output_attentions:
                all_hyperbolic_attentions = (hyperbolic_self_attention_scores,)
        else:
            hyperbolic_self_attention_scores = None

        if (hyperbolic_hidden_states is not None) and (
            hyperbolic_encoder_hidden_states is not None
        ):
            # cross attn scores
            hyperbolic_cross_attention_scores = hyperbolic_distance_scores(
                self.manifold,
                hyperbolic_hidden_states,
                hyperbolic_attention_mask,
                hyperbolic_encoder_hidden_states,
                hyperbolic_encoder_attention_mask,
                normalize=(
                    not self.config.combine_euclidean_hyperbolic_attention_scores
                ),
            )
            if output_attentions:
                all_hyperbolic_attentions = all_hyperbolic_attentions + (
                    hyperbolic_cross_attention_scores,
                )
        else:
            hyperbolic_cross_attention_scores = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_euclidean_hidden_states = all_euclidean_hidden_states + (
                    euclidean_hidden_states,
                )
                # all_hyperbolic_hidden_states = all_hyperbolic_hidden_states + (hyperbolic_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            euclidean_layer_outputs = layer(
                euclidean_hidden_states=euclidean_hidden_states,
                euclidean_attention_mask=euclidean_attention_mask,
                euclidean_encoder_hidden_states=euclidean_encoder_hidden_states,
                euclidean_encoder_attention_mask=euclidean_encoder_attention_mask,
                head_mask=layer_head_mask,
                hyperbolic_self_attention_scores=hyperbolic_self_attention_scores,
                hyperbolic_cross_attention_scores=hyperbolic_cross_attention_scores,
                # past_key_value = past_key_value,
                output_attentions=output_attentions,
            )

            euclidean_hidden_states = euclidean_layer_outputs[0]
            # hyperbolic_hidden_states = hyperbolic_layer_outputs[0]

            if output_attentions:
                all_euclidean_attentions = all_euclidean_attentions + (
                    euclidean_layer_outputs[1],
                )
                # all_hyperbolic_attentions = all_hyperbolic_attentions + (hyperbolic_layer_outputs[1],)

        if output_hidden_states:
            all_euclidean_hidden_states = all_euclidean_hidden_states + (
                euclidean_hidden_states,
            )
            # all_hyperbolic_hidden_states = all_hyperbolic_hidden_states + (hyperbolic_hidden_states,)

        return tuple(
            v
            for v in [
                euclidean_hidden_states,
                # hyperbolic_hidden_states,
                all_euclidean_hidden_states,
                # all_hyperbolic_hidden_states,
                all_euclidean_attentions,
                all_hyperbolic_attentions,
            ]
            if v is not None
        )


class DecoderOutputCatEHoutputs(nn.Module):
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


class DecoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderPredictionHead(nn.Module):
    def __init__(self, config, input_embed) -> None:
        super().__init__()
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        if config.tie_input_output_emb:
            self.decoder.weight = input_embed.weight
        else:
            nn.init.normal_(self.decoder.weight, mean=0, std=config.hidden_size**-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.decoder(hidden_states)
        return hidden_states
