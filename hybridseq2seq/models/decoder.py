from typing import Callable, List, Optional

import torch

from ..layers import (
    DecoderOutput,
    DecoderPredictionHead,
    HMDSEmbeddingFromCrossDistances,
    HybridTransformerDecoder,
    TransformerDecoder,
    TransformerEmbeddings,
)
from ..manifolds import PoincareBall
from ..utils import get_logger
from .base import BaseModule

logger = get_logger(__name__)


def shift_tokens_right(
    input_ids: torch.Tensor,
    decoder_start_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if attention_mask is not None:
        shifted_attn_mask = input_ids.new_zeros(attention_mask.shape)
        shifted_attn_mask[:, 1:] = attention_mask[:, :-1].clone()
        shifted_attn_mask[:, 0] = 1
        return shifted_input_ids, shifted_attn_mask.to(attention_mask.dtype)

    return shifted_input_ids, None


class HybridDecoderModel(BaseModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.manifold = PoincareBall()

        assert (
            self.config.semantic_parser_epsilon is not None
        ), f"config.model.semantic_parser_epsilon not set, check config file"
        assert (
            self.config.semantic_parser_relation_edge_weight is not None
        ), f"config.model.semantic_parser_relation_edge_weight not set, check config file"
        assert (
            self.config.add_sos_token is not None
        ), f"config.model.add_sos_token not set, check model construction"

        self.hyperbolic_embedding = HMDSEmbeddingFromCrossDistances(
            epsilon=self.config.semantic_parser_epsilon,
            relation_edge_weight=self.config.semantic_parser_relation_edge_weight,
            default_max_distance=self.config.default_max_distance,
            add_sos_token=self.config.add_sos_token,
            hyperbolic_embedding_size=self.config.hyperbolic_hidden_size,
            hmds_scale=self.config.hyperbolic_scale,
            manifold=self.manifold,
        )

        self.hybrid_decoder = HybridTransformerDecoder(config, self.manifold)

        self.output = DecoderOutput(config)

        self.is_decoder = True

        self.init_weights()
        # own init_weights
        self.embeddings = TransformerEmbeddings(config)
        self.prediction_head = DecoderPredictionHead(
            config, input_embed=self.embeddings.word_embeddings
        )

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        source_distance_matrix: torch.FloatTensor,
        decode: Callable,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ):
        """Decode the encoder embeddings.

        Parameters
        ----------
        input_ids: torch.FloatTensor : (bs, T)
            The decoder input ids of shape `(bs, tgt_seq_length)`, for teacher forcing.
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the target sequences (0 indicate padding tokens).
        labels: torch.FloatTensor : (bs, T)
            The target input ids of shape `(bs, tgt_seq_length)`, the start of token is not prepended.
        encoder_hidden_states: torch.FloatTensor : (bs, src_seq_length)
            The hidden_states from the encoder model.
        encoder_attention_mask: torch.FloatTensor : (bs, src_seq_length)
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).
        source_token_seqs: List[List[str]]
            The list of the source token sequences (textual)
        source_distance_matrix: torch.FloatTensor
            The symmetric distance matrices for the batch between the tokens in the source sequences

        Returns
        -------
        Tuple[torch.Tensor]

        """
        if labels is not None and input_ids is None:
            input_ids, attention_mask = shift_tokens_right(
                labels, self.config.sos_idx, attention_mask
            )

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=device
            )  # (bs, tgt_seq_length)

        euclidean_source_attention_mask = self.get_extended_attention_mask(
            encoder_attention_mask, input_shape=encoder_attention_mask.shape
        )  # (bs, None, None, src_seq_length)
        hyperbolic_source_attention_mask = self.get_extended_hyperbolic_attention_mask(
            encoder_attention_mask,
            input_shape=encoder_attention_mask.shape,
        )  # (bs, None, src_seq_length)

        cross_distance_matrices_i = source_distance_matrix
        hyperbolic_attention_mask_i = attention_mask[:, :1]

        euclidean_tgt_hs = self.embeddings(input_ids)  # (bs, tgt_seq_length, dim)

        out_euclidean_hidden_states = []

        T = input_shape[1]
        src_seq_length = source_distance_matrix.shape[1]

        for i in range(1, T + 1):
            previous_output_seq_i = input_ids[:, :i]
            euclidean_hidden_states_i = euclidean_tgt_hs[:, :i]
            euclidean_attention_mask_i = self.get_extended_attention_mask(
                attention_mask[:, :i],
                previous_output_seq_i.shape,
                causal=True,
            )  # (bs, None, tgt_seq_length, tgt_seq_length) causal mask for self attention

            previous_hyperbolic_attention_mask = torch.ones_like(attention_mask[:, :i])
            previous_hyperbolic_attention_mask[:, :-1] = hyperbolic_attention_mask_i

            previous_token_seqs_i = decode(previous_output_seq_i)

            (
                hyperbolic_cross_embeddings_i,
                hyperbolic_attention_mask_i,
                cross_distance_matrices_i,
            ) = self.hyperbolic_embedding.embed(
                step=i,
                previous_cross_distance_matrix=cross_distance_matrices_i,  # (bs, step-1, step-1)
                source_distance_matrix=source_distance_matrix,  # (bs, src_seq_length, src_seq_length)
                source_tokens_list=source_token_seqs,  # (bs, src_seq_length)
                source_attention_mask=encoder_attention_mask,
                previous_target_tokens_list=previous_token_seqs_i,  # (bs, step)
                previous_target_hyperbolic_attention_mask=previous_hyperbolic_attention_mask,  # (bs, step)
            )

            hyperbolic_source_hidden_states = hyperbolic_cross_embeddings_i[
                :, :src_seq_length
            ]

            hyperbolic_hidden_states_i = hyperbolic_cross_embeddings_i[
                :, src_seq_length:
            ]

            exp_hyperbolic_attention_mask_i = (
                self.get_extended_hyperbolic_attention_mask(
                    hyperbolic_attention_mask_i,
                    previous_output_seq_i.shape,
                    causal=True,
                )
            )  # (bs, tgt_seq_length, tgt_seq_length)

            # Hybrid decoder layers
            hybrid_decoder_outputs = self.hybrid_decoder(
                euclidean_hidden_states=euclidean_hidden_states_i,
                euclidean_attention_mask=euclidean_attention_mask_i,
                euclidean_encoder_hidden_states=encoder_hidden_states,
                euclidean_encoder_attention_mask=euclidean_source_attention_mask,
                hyperbolic_hidden_states=hyperbolic_hidden_states_i,
                hyperbolic_attention_mask=exp_hyperbolic_attention_mask_i,
                hyperbolic_encoder_hidden_states=hyperbolic_source_hidden_states,
                hyperbolic_encoder_attention_mask=hyperbolic_source_attention_mask,
            )
            euclidean_hidden_states_i = hybrid_decoder_outputs[0]

            out_euclidean_hidden_states.append(euclidean_hidden_states_i[:, -1, :])

        out_euclidean_hidden_states = torch.stack(out_euclidean_hidden_states, dim=1)

        out_euclidean_hidden_states = self.output(out_euclidean_hidden_states)
        out_predictions = self.prediction_head(out_euclidean_hidden_states)

        return out_predictions

    def forward_one_step(
        self,
        input_ids: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        cross_distance_matrix: torch.FloatTensor,
        source_distance_matrix: torch.FloatTensor,
        decode: Callable,
        attention_mask: Optional[torch.FloatTensor] = None,
        hyperbolic_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        running: Optional[torch.FloatTensor] = None,
    ):
        """Decode the encoder embeddings.

        Parameters
        ----------
        input_ids: torch.FloatTensor : (bs, T)
            The previous decoder outputs of shape `(bs, tgt_seq_length)`, for teacher forcing, eos token marks the end of the sequence.
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the target sequences (0 indicate padding tokens).
        encoder_hidden_states: torch.FloatTensor : (bs, src_seq_length)
            The hidden_states from the encoder model.
        encoder_attention_mask: torch.FloatTensor : (bs, src_seq_length)
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).
        source_token_seqs: List[List[str]]
            The list of the source token sequences (textual)
        cross_distance_matrix: torch.FloatTensor
            The symmetric cross-distance matrices for the batch between the tokens in the source+previous target sequences
        source_distance_matrix: torch.FloatTensor
            The symmetric distance matrices for the batch between the tokens in the source sequences

        Returns
        -------
        Tuple[torch.Tensor]

        """
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (
                (input_ids != self.config.pad_idx).to(device).float()
            )  # (bs, tgt_seq_length)

        euclidean_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape, causal=True
        )  # (bs, None, tgt_seq_length, tgt_seq_length) causal for self multi-head attention

        if hyperbolic_attention_mask is None:
            hyperbolic_attention_mask = torch.ones_like(attention_mask)

        euclidean_source_attention_mask = self.get_extended_attention_mask(
            encoder_attention_mask,
            encoder_attention_mask.shape,
        )  # (bs, None, None, src_seq_length) for cross multi-head attention

        hyperbolic_source_attention_mask = self.get_extended_hyperbolic_attention_mask(
            encoder_attention_mask,
            encoder_attention_mask.shape,
        )  # (bs, None, src_seq_length) for cross hyperbolic attentino scores (one head)

        euclidean_hidden_states = self.embeddings(
            input_ids
        )  # (bs, tgt_seq_length, dim)

        src_seq_length = source_distance_matrix.shape[1]

        token_seqs = decode(input_ids)

        (
            hyperbolic_cross_hidden_states_step,
            hyperbolic_attention_mask_step,
            cross_distance_matrix_step,
        ) = self.hyperbolic_embedding.embed(
            step=input_shape[1],
            previous_cross_distance_matrix=cross_distance_matrix,  # (bs, step-1, step-1)
            source_distance_matrix=source_distance_matrix,  # (bs, src_seq_length, src_seq_length)
            source_tokens_list=source_token_seqs,  # (bs, src_seq_length)
            source_attention_mask=encoder_attention_mask,
            previous_target_tokens_list=token_seqs,  # (bs, step)
            previous_target_hyperbolic_attention_mask=hyperbolic_attention_mask,  # (bs, step)
            running=running,
        )

        hyperbolic_source_hidden_states = hyperbolic_cross_hidden_states_step[
            :, :src_seq_length
        ]

        hyperbolic_hidden_states = hyperbolic_cross_hidden_states_step[
            :, src_seq_length:
        ]

        exp_hyperbolic_attention_mask = self.get_extended_hyperbolic_attention_mask(
            hyperbolic_attention_mask_step,
            input_ids.shape,
            causal=True,
        )  # (bs, tgt_seq_length, tgt_seq_length) causal mask for hyperbolic self attn scores (one head)

        # Hybrid decoder layers
        hybrid_decoder_outputs = self.hybrid_decoder(
            euclidean_hidden_states=euclidean_hidden_states,
            euclidean_attention_mask=euclidean_attention_mask,
            euclidean_encoder_hidden_states=encoder_hidden_states,
            euclidean_encoder_attention_mask=euclidean_source_attention_mask,
            hyperbolic_hidden_states=hyperbolic_hidden_states,
            hyperbolic_attention_mask=exp_hyperbolic_attention_mask,
            hyperbolic_encoder_hidden_states=hyperbolic_source_hidden_states,
            hyperbolic_encoder_attention_mask=hyperbolic_source_attention_mask,
        )
        euclidean_hidden_states = hybrid_decoder_outputs[0]

        out_euclidean_hidden_states = self.output(euclidean_hidden_states)  # dense+act
        out_predictions = self.prediction_head(
            out_euclidean_hidden_states
        )  # map to vocab

        return (
            out_predictions,
            hyperbolic_attention_mask_step,
            cross_distance_matrix_step,
        )


class EuclideanDecoderModel(BaseModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.decoder = TransformerDecoder(config)

        self.output = DecoderOutput(config)

        self.is_decoder = True

        self.init_weights()
        # own init_weights
        self.embeddings = TransformerEmbeddings(config)
        self.prediction_head = DecoderPredictionHead(
            config, input_embed=self.embeddings.word_embeddings
        )

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
    ):
        """Decode the encoder embeddings.

        Parameters
        ----------
        input_ids: torch.FloatTensor : (bs, T)
            The decoder input ids of shape `(bs, tgt_seq_length)`, for teacher forcing.
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the target sequences (0 indicate padding tokens).
        labels: torch.FloatTensor : (bs, T)
            The target input ids of shape `(bs, tgt_seq_length)`, the start of token is not prepended.
        encoder_hidden_states: torch.FloatTensor : (bs, src_seq_length)
            The hidden_states from the encoder model.
        encoder_attention_mask: torch.FloatTensor : (bs, src_seq_length)
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).

        Returns
        -------
        Tuple[torch.Tensor]

        """
        if labels is not None and input_ids is None:
            input_ids, attention_mask = shift_tokens_right(
                labels, self.config.sos_idx, attention_mask
            )

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=device
            )  # (bs, tgt_seq_length)

        hidden_states = self.embeddings(input_ids)  # (bs, tgt_seq_length, dim)

        attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
            causal=True,
        )  # (bs, None, tgt_seq_length, tgt_seq_length) for multi head self attention

        encoder_attention_mask = self.get_extended_attention_mask(
            encoder_attention_mask,
            encoder_attention_mask.shape,
        )  # (bs, None, None, src_seq_length) for multi head cross attention

        decoder_outputs = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = self.output(decoder_outputs[0])
        out_predictions = self.prediction_head(hidden_states)

        return out_predictions

    def forward_one_step(
        self,
        input_ids: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """Decode the encoder embeddings.

        Parameters
        ----------
        input_ids: torch.FloatTensor : (bs, T)
            The previous decoder outputs of shape `(bs, tgt_seq_length)`, for teacher forcing, eos token marks the end of the sequence.
        attention_mask: Optional[torch.FloatTensor]
            The attention mask corresponding to the target sequences (0 indicate padding tokens).
        encoder_hidden_states: torch.FloatTensor : (bs, src_seq_length)
            The hidden_states from the encoder model.
        encoder_attention_mask: torch.FloatTensor : (bs, src_seq_length)
            The attention mask corresponding to the encoder_hidden_states tensor (0 indicate padding tokens).

        Returns
        -------
        Tuple[torch.Tensor]

        """
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (
                (input_ids != self.config.pad_idx).to(device).float()
            )  # (bs, tgt_seq_length)

        attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
            causal=True,
        )  # (bs, None, tgt_seq_length, tgt_seq_length) for multi-head attention

        encoder_attention_mask = self.get_extended_attention_mask(
            encoder_attention_mask, encoder_attention_mask.shape
        )  # (bs, None, None, src_seq_length) for multi head attention

        hidden_states = self.embeddings(input_ids)  # (bs, tgt_seq_length, dim)

        # Hybrid decoder layers
        decoder_outputs = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = decoder_outputs[0]

        out_euclidean_hidden_states = self.output(hidden_states)  # dense+act
        out_predictions = self.prediction_head(
            out_euclidean_hidden_states
        )  # map to vocab

        return out_predictions


_DECODERS = {
    "hybrid-decoder": HybridDecoderModel,
    "euclidean-decoder": EuclideanDecoderModel,
}


def get_decoder(config):
    if config.decoder in _DECODERS:
        cls = _DECODERS[config.decoder]
        logger.info(f"Using decoder {config.decoder}")
        return cls(config)
    else:
        raise KeyError(
            f"Unknown {config.decoder} decoder class. Use one of {_DECODERS.keys()}"
        )
