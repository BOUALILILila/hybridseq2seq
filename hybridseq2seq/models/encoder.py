from typing import Optional

import torch

from .base import BaseModule
from ..layers import TransformerEmbeddings, TransformerEncoder

from ..utils import get_logger

logger = get_logger(__name__)


class EuclideanTranformerEncoderModel(BaseModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TransformerEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.is_decoder = False

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        embeddings = self.embeddings(input_ids)  # (bs, seq_length, dim)

        return self.encoder(
            hidden_states=embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


_ENCODERS = {
    "euclidean-encoder": EuclideanTranformerEncoderModel,
}


def get_encoder(config):
    if config.encoder in _ENCODERS:
        cls = _ENCODERS[config.encoder]
        logger.info(f"Using encoder {config.encoder}")
        return cls(config)
    else:
        raise KeyError(
            f"Unknown {config.encoder} encoder class. Use one of {_ENCODERS.keys()}"
        )
