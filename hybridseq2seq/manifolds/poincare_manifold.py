from typing import Union

import numpy as np
import torch
import geoopt

from . import math

from ..utils import get_logger

logger = get_logger(__name__)


class PoincareBall(geoopt.PoincareBall):
    def projx_from_hyperboloid(self, x: Union[torch.Tensor, np.ndarray]):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        return math.gans_to_manifold(x, k=self.k)

    def weighted_midpoint_bmm(
        self,
        xs: torch.Tensor,
        weights: torch.Tensor,
        lincomb: bool = False,
        project=True,
    ):
        mid = math.weighted_midpoint_bmm(
            xs=xs,
            weights=weights,
            k=self.k,
            lincomb=lincomb,
        )
        if project:
            out = self.projx(mid, dim=-1)
            return out
        else:
            return mid

    def dist_matmul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return math.dist_matmul(x, y, k=self.k)

    def batched_dist_stable(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = x.unsqueeze(-2)
        size = [1 for _ in range(x.dim())]
        size[-2] = x.shape[-2]
        x = x.repeat(size)  # (bs, .., seq_len, seq_len, dim)
        return math.dist(x, y.unsqueeze(-3), k=self.k)  # (bs, .., seq_len, seq_len)
