from typing import Any
import torch

from ..utils import get_logger

logger = get_logger(__name__)


class Searcher:
    def __init__(self, config) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class GreedySearcher(Searcher):
    def __init__(self, config) -> None:
        super().__init__(config)

    def __call__(self, preds: torch.Tensor) -> Any:
        return torch.argmax(preds, dim=-1, keepdim=True)


_SEARCHERS = {
    "greedy-search": GreedySearcher,
}


def get_searcher(config):
    if config.name in _SEARCHERS:
        logger.info(f"Using {config.name} for generation.")
        return _SEARCHERS[config.name](config)
    else:
        raise KeyError(
            f"Unknown {config.name} searcher class. Use one of {_SEARCHERS.keys()}"
        )
