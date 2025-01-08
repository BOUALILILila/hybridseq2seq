import pytest
import torch

from hybridseq2seq.hmds.embedding import PCA, _hmds, get_eig_vals_and_vecs, hmds
from hybridseq2seq.manifolds import PoincareBall


@pytest.fixture
def distance_matrix():
    return torch.tensor(
        [
            [0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 3.25, 2.25, 0],
            [2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.25, 0.25, 0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.25, 4.25, 0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0],
            [3.25, 4.25, 0.25, 0.25, 0.25, 0.25, 0.0, 1, 0],
            [2.25, 0.25, 4.25, 0.25, 0.25, 0.25, 1.0, 0.0, 0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def distance_matrix_no_mask():
    return torch.tensor(
        [
            [0, 2, 3.0, 3.25, 2.25],
            [2, 0, 4.0, 4.25, 0.25],
            [3.0, 4.0, 0.0, 0.25, 4.25],
            [3.25, 4.25, 0.25, 0.0, 1],
            [2.25, 0.25, 4.25, 1.0, 0.0],
        ]
    )


def test_eigendecomposition(distance_matrix):
    eigen_vals, eigen_vecs = get_eig_vals_and_vecs(distance_matrix)
    assert eigen_vals.shape == distance_matrix.shape[:-1]
    assert eigen_vecs.shape == distance_matrix.shape
    val_vec_mul = torch.einsum("i,ji->ji", eigen_vals, eigen_vecs)
    mat_vec_mul = torch.einsum("ij,jk->ik", distance_matrix, eigen_vecs)
    assert torch.allclose(val_vec_mul, mat_vec_mul, rtol=1e-5, atol=1e-5)


def test_pca(distance_matrix):
    for k in range(2, 6):
        Xrec = PCA(distance_matrix, k)
        assert Xrec.shape == distance_matrix.shape[:-1] + (k,)


def test_raises_assert_error_when_k_leq_0(distance_matrix):
    for k in (-1, 0):
        with pytest.raises(Exception) as exc_info:
            _ = PCA(distance_matrix, k)
        assert (
            str(exc_info.value)
            == f"Rank k must be greater than 0, but k = {k} was given."
        )


def test_hmds(distance_matrix):
    src_mask = torch.tensor([1, 1, 1, 0, 0, 0.0])
    tgt_mask = torch.tensor([1.0, 1.0, 0])
    mask = torch.cat([src_mask, tgt_mask])
    for k in (3, 8, 10):
        Xrec = hmds(distance_matrix, mask, k)
        assert Xrec.shape == distance_matrix.shape[:-1] + (k,)
        ball = PoincareBall()
        Xrec = ball.projx_from_hyperboloid(Xrec)
        assert ball._check_point_on_manifold(Xrec)
        assert not torch.all(torch.isnan(Xrec))


def test_hmds_masking(distance_matrix_no_mask, distance_matrix):
    src_mask = torch.tensor([1, 1, 1, 0, 0, 0.0])
    tgt_mask = torch.tensor([1.0, 1.0, 0])
    mask = torch.cat([src_mask, tgt_mask])
    k = 4
    Xrec = hmds(distance_matrix, mask, k)
    Xrec_masked = _hmds(distance_matrix_no_mask, k)
    assert torch.allclose(
        Xrec_masked,
        Xrec.masked_select(mask.view(-1, 1).bool()).view(-1, k),
        rtol=1e-8,
        atol=1e-8,
    )
