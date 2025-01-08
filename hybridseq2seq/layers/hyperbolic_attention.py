from typing import Optional, Tuple

import torch
from torch import nn

from .hyperbolic_linear_layer import PoincareLinear

from ..utils import get_logger

logger = get_logger(__name__)


class HyperbolicMultiHeadAttention(nn.Module):
    """Compute hyperbolic multi-head attention in Poincaré Ball.
    Use the hyperbolic distance for attention weighting and weighted Mobius gyormidpoint for aggregation.
    """

    # config info: is_decoder, hidden_size, num_attention_heads
    def __init__(self, config, manifold, is_decoder: bool = False):
        super().__init__()
        if (
            config.hyperbolic_hidden_size % config.hyperbolic_num_attention_heads != 0
            and not hasattr(config, "embedding_size")
        ):
            raise ValueError(
                f"The hidden size ({config.hyperbolic_hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.hyperbolic_num_attention_heads})"
            )

        self.manifold = manifold
        self.num_attention_heads = config.hyperbolic_num_attention_heads
        self.attention_head_size = int(
            config.hyperbolic_hidden_size / config.hyperbolic_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Projection Layers for Q, K, V
        self.query = PoincareLinear(
            manifold=self.manifold,
            in_features=config.hyperbolic_hidden_size,
            out_features=self.all_head_size,
            out_split=self.num_attention_heads,
        )
        self.key = PoincareLinear(
            manifold=self.manifold,
            in_features=config.hyperbolic_hidden_size,
            out_features=self.all_head_size,
            out_split=self.num_attention_heads,
        )
        self.value = PoincareLinear(
            manifold=self.manifold,
            in_features=config.hyperbolic_hidden_size,
            out_features=self.all_head_size,
            out_split=self.num_attention_heads,
        )

        # Dropout
        self.dropout = nn.Dropout(config.hyperbolic_attention_probs_dropout_prob)

        # Causal self-attention
        self.is_decoder = is_decoder

    def transpose_for_scores(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x.permute(0, 2, 1, 3)  # (bs, num_heads, seq_len, head_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """Compute new hidden_states using attention mechanism.

        Parameters
        ----------
        hidden_states: torch.FloatTensor
            The previous hidden_states (input or output of previous layer).
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the hidden_states tensor (0 indicate padding tokens).
        head_mask: Optional[torch.FloatTensor]
            Mask to nullify selected attention heads.
        encoder_hidden_states: Optional[torch.FloatTensor]
            The hidden_states from the encoder model if encoder-decoder attention.
        encoder_attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).
        output_attentions: Optional[bool], default = False
            Return attention scores if True.

        Returns
        -------
        Tuple[torch.FloatTensor]
            (Output of the mutlti-head attention mechanism, attention_scores, past_key_values)

        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores
        #   Use the hyperoblic distance in the manifold between the query and key vectors
        attention_scores = -self.manifold.dist_matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (bs, num_heads, seq_len, seq_len)
        # alternative
        # query_layer = query_layer.unsqueeze(-2).repeat(1, 1, 1, query_layer.shape[-2], 1) # (bs, num_heads, seq_len, seq_len, head_size)
        # attention_scores = -self.manifold.dist(query_layer, key_layer.unsqueeze(-3)) # (bs, num_heads, seq_len, seq_len)
        # stable numeric
        # attention_scores = -self.manifold.batched_dist_stable(query_layer, key_layer) # (bs, num_heads, seq_len, seq_len)

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

        context_layer = self.manifold.weighted_midpoint_bmm(
            value_layer, weights=attention_probs
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class MultiHeadAttentionOutput(nn.Module):
    # config: hidden_size
    def __init__(self, config, manifold):
        super().__init__()
        self.manifold = manifold
        self.dense = PoincareLinear(
            self.manifold, config.hyperbolic_hidden_size, config.hyperbolic_hidden_size
        )
        # self.LayerNorm = nn.LayerNorm(config.hyperbolic_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hyperbolic_hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.FloatTensor, input_tensor: torch.FloatTensor
    ) -> torch.FloatTensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.manifold.mobius_fn_apply(
            lambda x: self.dropout(x), hidden_states
        )
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.manifold.mobius_add(input_tensor, hidden_states)
        return hidden_states


class HyperbolicAttention(nn.Module):
    def __init__(self, config, manifold, is_decoder: bool = False):
        super().__init__()
        self.manifold = manifold
        self.attn = HyperbolicMultiHeadAttention(
            config, manifold=self.manifold, is_decoder=is_decoder
        )
        self.output = MultiHeadAttentionOutput(config, manifold=self.manifold)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )

        attention_output = self.output(attn_outputs[0], hidden_states)
        outputs = (attention_output,) + attn_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


def hyperbolic_distance_scores(
    manifold,
    hidden_states: torch.FloatTensor,  # (bs, tgt_seq_legnth, dim)
    attention_mask: Optional[torch.FloatTensor] = None,  # (bs, tgt_seq_legnth)
    encoder_hidden_states: Optional[
        torch.FloatTensor
    ] = None,  # (bs, src_seq_legnth, dim)
    encoder_attention_mask: Optional[torch.FloatTensor] = None,  # (bs, src_seq_legnth)
    normalize: Optional[bool] = True,
) -> torch.FloatTensor:
    if encoder_hidden_states is not None:
        attention_mask = encoder_attention_mask
        key_vectors = encoder_hidden_states
    else:
        key_vectors = hidden_states
    # Attention scores
    #   Use the hyperoblic distance in the manifold between the query and key vectors
    attention_scores = -manifold.batched_dist_stable(
        hidden_states, key_vectors
    )  # (bs, tgt_seq_len, src_seq_len|tgt_seq_len)

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    if normalize:
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
    return attention_scores


class EuclideanAttentionWithHyperbolicScoresOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.FloatTensor, input_tensor: torch.FloatTensor
    ) -> torch.FloatTensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EuclideanAttentionWithHyperbolicScores(nn.Module):
    def __init__(self, config):
        super().__init__()
        # # Project hidden_states
        # self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Output layers
        self.output = EuclideanAttentionWithHyperbolicScoresOutput(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        hyperbolic_attention_scores: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        attention_probs = self.dropout(hyperbolic_attention_scores)
        # hs = self.proj(hidden_states)
        # attention_output = torch.matmul(attention_probs, hs)
        attention_output = torch.matmul(attention_probs, hidden_states)

        attention_output = self.output(attention_output, hidden_states)
        return attention_output
