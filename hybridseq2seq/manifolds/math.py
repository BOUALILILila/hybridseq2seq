import torch
from geoopt.manifolds.stereographic.math import _project, _lambda_x, _mobius_scalar_mul

from ..utils import get_logger

logger = get_logger(__name__)


def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
            - 1 / 11 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x**3
    elif order == 2:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5
    elif order == 3:
        return (
            x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5 - 1 / 7 * k**3 * x**7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)


def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)

    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        recip = k_sqrt.reciprocal()
        tanh_ = artanh(scaled_x)
        return recip * tanh_
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


# Copied from: https://github.com/mil-tokyo/hyperbolic_nn_plusplus/blob/28737e22822562ac18d5ea03f8c3e3929e945a83/geoopt_plusplus/manifolds/stereographic/math.py#L2085
def weighted_midpoint_bmm(
    xs: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    lincomb: bool = False,
):
    r"""
    Compute weighted Möbius gyromidpoint [1] in batch mode.
    The weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights
    :math:`\alpha_1,...,\alpha_n` is computed as follows:
    The weighted Möbius gyromidpoint is computed as follows
    .. math::
        m_c(v_1,\ldots,v_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_c
        \left(
        \frac{
        \sum_{i=1}^n\alpha_i (\lambda_{v_i}^c v_i)
        }{
        \sum_{i=1}^n|\alpha_i|(\lambda_{v_i}^c-1)
        }
        \right)
    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters).

    Parameters
    ----------
    xs : torch.Tensor : (*, n , d)
        points on Poincaré Ball
    weights : torch.Tensor : (*, m , n)
        weights for aggregating
    k : torch.Tensor
        constant sectional curvature
    lincomb : bool
        linear combination implementation

    Returns
    -------
    torch.Tensor : (*, m , d)
        Einstein midpoint in Poincaré coordinates

    References
    ----------
    .. [1] Shimizu, R., Mukuta, Y., and Harada, T. (2020). Hyperbolic neural
                networks++. arXiv preprint arXiv:2006.08210.

    """
    return _weighted_midpoint_bmm(
        xs=xs,
        weights=weights,
        k=k,
        lincomb=lincomb,
    )


# @torch.jit.script
def _weighted_midpoint_bmm(
    xs: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    lincomb: bool = False,
):
    gamma = _lambda_x(xs, k=k, dim=-1, keepdim=True)
    denominator = torch.matmul(weights.abs(), gamma - 1)
    nominator = torch.matmul(weights, gamma * xs)
    two_mean = nominator / denominator.clamp_min(1e-10)  ## instead of clamp_abs
    a_mean = two_mean / (
        1.0 + (1.0 + k * two_mean.pow(2).sum(dim=-1, keepdim=True)).sqrt()
    )

    if lincomb:
        alpha = weights.abs().sum(dim=-1, keepdim=True)
        a_mean = _mobius_scalar_mul(alpha, a_mean, k=k, dim=-1)
    return _project(a_mean, k, dim=-1)


def gans_to_manifold(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""Projection from Gans Model to Poincaré Model.

    Parameters
    ----------
    x : torch.Tensor : (bs, n, d)
        Batch of sequence embeddings
    k : torch.Tensor
        Curvature of the manifold * (-1)

    Returns
    -------
    torch.Tensor : (bs, n, d)
        The embeddings in the Poincaré Ball
    """
    return _gans_to_manifold(x, -k)


# @torch.jit.script
def _gans_to_manifold(x: torch.Tensor, k: torch.Tensor):
    r = k.sqrt().to(x.dtype).to(x.device)
    y = r + torch.sqrt(1.0 + x.norm(dim=-1, p=2) ** 2)
    y = y.unsqueeze(-1)
    return r * x / y


def dist_matmul(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor):
    r"""
    Compute the geodesic distance between :math:`x` and :math:`y` on the manifold.
    .. math::
        d_\kappa(x, y) = 2\tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)

    Parameters
    ----------
    x :  torch.Tensor : (*, n, d)
        point on manifold
    y :  torch.Tensor : (*, d, m)
        point on manifold
    k :  torch.Tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    torch.Tensor : (*, n, m)
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist_matmul(x, y, k)


# @torch.jit.script
def _dist_matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-2, keepdim=True)
    xy = torch.matmul(x, y)
    num = x2 - 2 * xy + y2
    denom = (1 + 2 * k * xy + k.pow(2) * x2 * y2).clamp_min(1e-15)
    return 2.0 * artan_k((num / denom).sqrt(), k)


def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k**2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(1e-15)


def dist(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )
