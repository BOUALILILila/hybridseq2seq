from typing import Callable, List, Optional

import torch

from ..utils import get_logger
from .base import BaseModel, BaseModule
from .search import get_searcher

logger = get_logger(__name__)


class EncoderDecoderModel(BaseModel):
    def __init__(
        self,
        config,
        encoder_model: BaseModule,
        decoder_model: BaseModule,
    ) -> None:
        super().__init__()
        self.config = config

        self.encoder = encoder_model
        self.decoder = decoder_model

    def forward(
        self,
        source_token_ids: torch.FloatTensor,
        source_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        source_distance_matrix: torch.FloatTensor,
        decode_fn: Callable,
        target_token_ids: Optional[torch.FloatTensor] = None,
        target_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encoder(
            input_ids=source_token_ids,
            attention_mask=source_attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]

        decoder_output = self.decoder(
            input_ids=target_token_ids,
            attention_mask=target_attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=source_attention_mask,
            source_token_seqs=source_token_seqs,
            source_distance_matrix=source_distance_matrix,
            decode=decode_fn,
        )

        return decoder_output

    def generate(
        self,
        source_token_ids: torch.FloatTensor,
        source_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        source_distance_matrix: torch.FloatTensor,
        decode_fn: Callable,
        **kwargs,
    ):
        encoder_outputs = self.encoder(
            input_ids=source_token_ids,
            attention_mask=source_attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]

        batch_size = source_token_ids.shape[0]

        running = torch.ones(
            [batch_size], dtype=torch.bool, device=source_token_ids.device
        )
        out_len = torch.zeros_like(running, dtype=torch.long)

        previous_tgt_token_ids = torch.full(
            [batch_size, 1],
            self.config.sos_idx,
            dtype=torch.long,
            device=source_token_ids.device,
        )
        previous_cross_distance_matrix = source_distance_matrix
        previous_hyperbolic_attention_mask = None

        search = get_searcher(self.config.generation.searcher)  # GreedySearcher
        all_outputs = []
        all_pred_seqs = []
        for i in range(self.config.generation.stop_criteria.max_len):
            decoder_outputs = self.decoder.forward_one_step(
                input_ids=previous_tgt_token_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=source_attention_mask,
                source_token_seqs=source_token_seqs,
                cross_distance_matrix=previous_cross_distance_matrix,
                source_distance_matrix=source_distance_matrix,
                decode=decode_fn,
                hyperbolic_attention_mask=previous_hyperbolic_attention_mask,
                running=running,
            )

            output = decoder_outputs[0]
            all_outputs.append(
                output[:, -1:, :]
            )  # the logits of the last tokens (bs, 1, vocab_size)

            out_token = search(output[:, -1, :])  # the token id (bs, 1)
            all_pred_seqs.append(out_token)

            out_len[running] = i + 1
            running &= out_token.squeeze(1) != self.config.eos_idx

            previous_tgt_token_ids = torch.cat(
                (previous_tgt_token_ids, out_token), dim=1
            )
            previous_cross_distance_matrix = decoder_outputs[2]
            previous_hyperbolic_attention_mask = torch.ones_like(
                previous_tgt_token_ids
            ).float()
            previous_hyperbolic_attention_mask[:, :-1] = decoder_outputs[1]

        return {
            "logits": torch.cat(all_outputs, dim=1),
            "pred_lens": out_len,
            "pred_seqs": torch.cat(all_pred_seqs, dim=1),
        }


class EuclideanEncoderDecoderModel(BaseModel):
    def __init__(
        self,
        config,
        encoder_model: BaseModule,
        decoder_model: BaseModule,
    ) -> None:
        super().__init__()
        self.config = config

        self.encoder = encoder_model
        self.decoder = decoder_model

    def forward(
        self,
        source_token_ids: torch.FloatTensor,
        source_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        source_distance_matrix: torch.FloatTensor,
        decode_fn: Callable,
        target_token_ids: Optional[torch.FloatTensor] = None,
        target_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encoder(
            input_ids=source_token_ids,
            attention_mask=source_attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]

        decoder_output = self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=source_attention_mask,
            input_ids=target_token_ids,
            attention_mask=target_attention_mask,
            labels=labels,
        )

        return decoder_output

    def generate(
        self,
        source_token_ids: torch.FloatTensor,
        source_attention_mask: torch.FloatTensor,
        source_token_seqs: List[List[str]],
        source_distance_matrix: torch.FloatTensor,
        decode_fn: Callable,
        **kwargs,
    ):
        encoder_outputs = self.encoder(
            input_ids=source_token_ids,
            attention_mask=source_attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]

        batch_size = source_token_ids.shape[0]

        running = torch.ones(
            [batch_size], dtype=torch.bool, device=source_token_ids.device
        )
        out_len = torch.zeros_like(running, dtype=torch.long)

        previous_tgt_token_ids = torch.full(
            [batch_size, 1],
            self.config.sos_idx,
            dtype=torch.long,
            device=source_token_ids.device,
        )

        search = get_searcher(self.config.generation.searcher)  # GreedySearcher
        all_outputs = []
        all_pred_seqs = []
        for i in range(self.config.generation.stop_criteria.max_len):
            decoder_outputs = self.decoder.forward_one_step(
                input_ids=previous_tgt_token_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=source_attention_mask,
            )
            all_outputs.append(
                decoder_outputs[:, -1:, :]
            )  # the logits of the last tokens (bs, 1, vocab_size)

            out_token = search(decoder_outputs[:, -1, :])  # the token id (bs, 1)
            all_pred_seqs.append(out_token)

            out_len[running] = i + 1
            running &= out_token.squeeze(1) != self.config.eos_idx

            previous_tgt_token_ids = torch.cat(
                (previous_tgt_token_ids, out_token), dim=1
            )

        return {
            "logits": torch.cat(all_outputs, dim=1),
            "pred_lens": out_len,
            "pred_seqs": torch.cat(all_pred_seqs, dim=1),
        }


_MODELS = {
    "hybrid": EncoderDecoderModel,
    "euclidean": EuclideanEncoderDecoderModel,
}


def get_model(config, encoder_model, decoder_model):
    if config.name in _MODELS:
        cls = _MODELS[config.name]
        logger.info(f"Using model {config.name}")
        return cls(
            config=config, encoder_model=encoder_model, decoder_model=decoder_model
        )
    else:
        raise KeyError(
            f"Unknown {config.name} model class. Use one of {_MODELS.keys()}"
        )
