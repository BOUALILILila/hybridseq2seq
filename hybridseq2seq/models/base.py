import os
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from ..utils import get_logger

logger = get_logger(__name__)


CONFIG_FILE_NAME = "config.yaml"
MODEL_WEIGHTS_NAME = "model.pt"


class BaseModule(nn.Module):
    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, path, config, load_function: Callable = torch.load):
        """Loads a model saved on local storage."""
        model = cls(config)

        model_weight_path = os.path.join(path, MODEL_WEIGHTS_NAME)
        if os.path.exists(model_weight_path):
            logger.info(f"Loading model weights from {model_weight_path}")
            model_dict = load_function(model_weight_path, map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
            if len(load_result.missing_keys) != 0:
                logger.error(
                    f"There were missing keys in the checpoint model loaded: {load_result.missing_keys}."
                )
            if len(load_result.unexpected_keys) != 0:
                logger.warn(
                    f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
                )
        return model

    def save_pretrained(self, output_dir: str, save_function: Callable = torch.save):
        """Saves the model on local storage."""
        model_dict = self.state_dict()
        save_function(model_dict, os.path.join(output_dir, MODEL_WEIGHTS_NAME))

    def load(self, path: str, load_function: Callable = torch.load):
        """load the model weights from local storage into self"""
        model_weight_path = os.path.join(path, MODEL_WEIGHTS_NAME)
        load_result = None
        if os.path.exists(model_weight_path):
            logger.info(f"Loading model weights from {model_weight_path}")
            model_dict = load_function(model_weight_path, map_location="cpu")
            load_result = self.load_state_dict(model_dict, strict=False)
            if len(load_result.missing_keys) != 0:
                logger.error(
                    f"There were missing keys in the checpoint model loaded: {load_result.missing_keys}."
                )
            if len(load_result.unexpected_keys) != 0:
                logger.warn(
                    f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
                )
        return load_result

    @staticmethod
    def create_extended_attention_mask_for_decoder(
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
    ) -> torch.Tensor:
        batch_size, seq_length = input_shape
        device = attention_mask.device
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = (
            seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
            <= seq_ids[None, :, None]
        )
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, seq_length, prefix_seq_len),
                        device=device,
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        return extended_attention_mask

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int] = None,
        causal: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Parameters
        ----------
            attention_mask: torch.Tensor
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape: Tuple[int]
                The shape of the input to the model.
            Causal: Optional[bool]
                Mask future position for decoding

        Returns
        -------
            torch.Tensor
                The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if causal:
                extended_attention_mask = (
                    BaseModel.create_extended_attention_mask_for_decoder(
                        attention_mask, input_shape
                    )
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=attention_mask.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            attention_mask.dtype
        ).min
        return extended_attention_mask

    def get_extended_hyperbolic_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        causal: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored. Hyperbolic attention uses only one head. No boradcasting for multi-head attention.

        Parameters
        ----------
            attention_mask: torch.Tensor
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape: Tuple[int]
                The shape of the input to the model.
            Causal: Optional[bool]
                Mask future position for decoding

        Returns
        -------
            torch.Tensor
                The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, seq_length, seq_length]
            if causal:
                extended_attention_mask = (
                    BaseModel.create_extended_attention_mask_for_decoder(
                        attention_mask, input_shape
                    ).squeeze(
                        1
                    )  # squeeze the mutli-head broadcast
                )
            else:
                extended_attention_mask = attention_mask[:, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=attention_mask.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            attention_mask.dtype
        ).min
        return extended_attention_mask


class BaseModel(BaseModule):
    pass
