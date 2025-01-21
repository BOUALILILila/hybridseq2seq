from abc import ABC, abstractmethod
from typing import Any
import torch

from ..utils import get_logger

logger = get_logger(__name__)


class Searcher(ABC):
    """
    Abstract base class for search strategies.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the searcher with a configuration.

        Parameters
        ----------
        config : dict
            Configuration for the searcher.
        """
        self.config = config

    @abstractmethod
    def __call__(self, preds: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Executes the search logic.

        Parameters
        ----------
        preds : torch.Tensor
            A tensor of predictions.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The result of the search.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the '__call__' method.")


class GreedySearcher(Searcher):
    """
    Implements a greedy search strategy.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def __call__(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Performs greedy search by selecting the index of the maximum value
        along the last dimension.

        Parameters
        ----------
        preds : torch.Tensor
            A tensor of predictions.

        Returns
        -------
        torch.Tensor
            Indices of the maximum values along the last dimension.
        """
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
