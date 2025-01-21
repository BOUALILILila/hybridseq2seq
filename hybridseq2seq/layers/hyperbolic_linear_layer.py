import math

from scipy.special import beta

import torch
import geoopt
from geoopt.manifolds.stereographic.math import arsinh


# Ganea et., (2018) https://arxiv.org/pdf/1805.09112.pdf
class MobiusLinear(torch.nn.Module):
    """
    Mobius linear layer.
    """

    def __init__(
        self, manifold, in_features: int, out_features: int, use_bias: bool = True
    ):
        super().__init__()
        self.manifold = manifold

        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features

        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor):
        mv = self.manifold.mobius_matvec(self.weight, x)
        res = self.manifold.projx(mv)

        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias)
            res = self.manifold.projx(res)

        return res


# Shimizu et al., (2021) https://arxiv.org/pdf/2006.08210.pdf
class PoincareLinear(torch.nn.Module):
    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        out_split: int = 1,
        gain=1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_features
        self.out_dim = out_features
        self.out_split = out_split

        weight = torch.empty(self.in_dim, self.out_dim).normal_(
            mean=0, std=(2 * self.in_dim * self.out_dim / out_split) ** -0.5 * gain
        )
        self.weight_g = torch.nn.Parameter(weight.norm(dim=0))
        self.weight_v = torch.nn.Parameter(weight)

        self.bias = torch.nn.Parameter(
            torch.empty(self.out_dim), requires_grad=use_bias
        )

        self.reset_parameters()

        self.beta_ni = beta(self.out_dim / out_split / 2, 1 / 2)
        self.beta_n = beta(self.out_dim / 2, 1 / 2)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = poincare_linear(
            x,
            self.weight_g,
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15),
            self.bias,
            self.manifold.c,
            out_split=1,
        )

        x = self.manifold.projx(x)

        if self.out_split > 1:
            size = x.size()
            x = (
                self.manifold.logmap0(x)
                .contiguous()
                .view(*size[:-1], self.out_split, size[-1] // self.out_split)
            )
            x = self.manifold.expmap0(x * self.beta_ni / self.beta_n)

        return x


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2.0 * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return (
        2
        * z_norm
        / rc
        * arsinh(
            (2.0 * torch.matmul(rcx, z_unit) * drcr.cosh() - (1.0 + cx2) * drcr.sinh())
            / torch.clamp_min(1.0 - cx2, 1e-15)
        )
    )


def poincare_linear(x, weight_g, weight_v, bias, c, out_split: int = 1):
    rc = c.sqrt()
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    x = (rc * x).sinh() / rc

    if out_split > 1:
        size = x.size()
        x = x.view(*size[:-1], out_split, size[-1] // out_split)

    return x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt())
