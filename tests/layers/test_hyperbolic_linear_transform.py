import torch
from hybridseq2seq.manifolds import PoincareBall

from hybridseq2seq.layers.hyperbolic_linear_layer import PoincareLinear, MobiusLinear


def test_mobius_linear_layer():
    manifold = PoincareBall()
    layer = MobiusLinear(manifold, 64, 8)

    x = manifold.expmap0(torch.rand(2, 4, 64))
    y = layer(x)

    assert manifold._check_point_on_manifold(y)
    assert y.shape == (2, 4, 8)


def test_poincare_linear_layer():
    manifold = PoincareBall()
    layer = PoincareLinear(manifold, 64, 8)

    x = manifold.expmap0(torch.rand(2, 4, 64))
    y = layer(x)

    assert manifold._check_point_on_manifold(y)
    assert y.shape == (2, 4, 8)


def test_poincare_linear_layer_split():
    manifold = PoincareBall()
    layer = PoincareLinear(manifold, 64, 8, out_split=2)

    x = manifold.expmap0(torch.rand(2, 4, 64))
    y = layer(x)

    assert manifold._check_point_on_manifold(y)
    assert y.shape == (2, 4, 2, 4)
