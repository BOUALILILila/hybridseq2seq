import torch

from ..utils import get_logger

logger = get_logger(__name__)


class CELoss:
    def __init__(self, config, ignore_idx: int = 0) -> None:
        self.ignore_idx = ignore_idx
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_idx,  # <pad>
            reduction="mean",
        )

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fct(
            pred.view(-1, pred.shape[-1]),  # (bs*seq_len, vocab_size)
            target.view(-1),  # (bs*seq_len)
        )


_LOSSES = {
    "cross-entropy": CELoss,
}


def get_criterion(config, ignore_idx: int = 0):
    if config.name in _LOSSES:
        cls = _LOSSES[config.name]
        logger.info(
            f"Using loss {config.name} with config {config}, ignore index {ignore_idx}"
        )
        return cls(config, ignore_idx=ignore_idx)
    else:
        raise KeyError(f"Unknown {config.name} loss. Use one of {_LOSSES.keys()}")
