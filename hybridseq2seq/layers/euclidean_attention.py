# from bert self attention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
import math
from typing import Optional, Tuple

import torch
from torch import nn

from ..utils import get_logger

logger = get_logger(__name__)


class MultiHeadAttention(nn.Module):
    """Compute multi-head attention.
    From "Attention Is All You Need".
    Relative position embeddings option.
    """

    # config info: is_decoder, hidden_size, num_attention_heads, attention_probs_dropout_prob, max_position_embeddings, position_embedding_type[optional]
    def __init__(
        self, config, position_embedding_type: str = None, is_decoder: bool = False
    ):
        """position_embedding_type: str, default = None
        Specify the position embedding type:
            absolute (default if not specified by position_embedding_type and in config.position_embedding_type)
            relative_key : relative position embedding for the key vectors
            relative_key_query : relative position embedding for the key and query vectors
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Projection Layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Position Embedding, default = absolute
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        # Causal self-attention
        self.is_decoder = is_decoder

        self.hyperbolic_attention_score_weight = (
            config.hyperbolic_attention_score_weight
        )

        # self.init_weights()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        hyperbolic_attention_scores: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Compute new hidden_states using attention mechanism.

        Parameters
        ----------
        hidden_states: torch.Tensor
            The previous hidden_states (input or output of previous layer).
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the hidden_states tensor (0 indicate padding tokens).
        head_mask: Optional[torch.FloatTensor]
            Mask to nullify selected attention heads.
        encoder_hidden_states: Optional[torch.FloatTensor]
            The hidden_states from the encoder model if encoder-decoder attention.
        encoder_attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]
            Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors
            of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and optionally
            if config.is_cross_attention=True 2 additional tensors
            of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
        output_attentions: Optional[bool], default = False
            Return attention scores if True.

        Returns
        -------
        Tuple[torch.Tensor]
            (Output of the mutlti-head attention mechanism, attention_scores, past_key_values)

        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # add hyperbolic attention scores
        if hyperbolic_attention_scores is not None:
            attention_scores += (
                self.hyperbolic_attention_score_weight
                * hyperbolic_attention_scores[:, None, :, :]
            )  # add for each head

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MultiHeadAttentionOutput(nn.Module):
    # config: hidden_size, layer_norm_eps, hidden_dropout_prob
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EuclideanAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, is_decoder: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(
            config,
            position_embedding_type=position_embedding_type,
            is_decoder=is_decoder,
        )
        self.output = MultiHeadAttentionOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        hyperbolic_attention_scores: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            hyperbolic_attention_scores,
        )
        attention_output = self.output(attn_outputs[0], hidden_states)
        outputs = (attention_output,) + attn_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class HybridEuclideanHyperbolicAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, is_decoder: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(
            config,
            position_embedding_type=position_embedding_type,
            is_decoder=is_decoder,
        )

        self.proj_out = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # Hyperbolic part
        # # Project hidden_states
        # self.proj_in = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.output = MultiHeadAttentionOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        hyperbolic_attention_scores: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attn_hidden_states = attn_outputs[0]

        # hyperbolic_attention_scores are normalized
        if hyperbolic_attention_scores is not None:
            hyperbolic_attention_probs = self.dropout(hyperbolic_attention_scores)
            # cross-attention
            hs = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )
            # hs = self.proj_in(hs)
            eh_hidden_states = torch.matmul(hyperbolic_attention_probs, hs)
            attn_hidden_states = torch.cat(
                (attn_hidden_states, eh_hidden_states), dim=-1
            )  # (bs, seq_length, d*2)
            attn_hidden_states = self.proj_out(attn_hidden_states)

        attention_output = self.output(attn_hidden_states, hidden_states)
        outputs = (attention_output,) + attn_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
