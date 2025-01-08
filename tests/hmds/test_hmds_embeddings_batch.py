import pytest
import torch

from hybridseq2seq.hmds.embedding_batched import PCA, get_eig_vals_and_vecs, hmds
from hybridseq2seq.manifolds import PoincareBall


@pytest.fixture
def batch_distance_matrices():
    A = torch.tensor(
        [
            [0.0, 6, 8, 9, 10, 11, 12, 12],
            [6.0, 0, 6, 7, 8, 9, 10, 10],
            [8.0, 6, 0, 7, 8, 9, 10, 10],
            [9.0, 7, 7, 0, 7, 8, 9, 9],
            [10, 8, 8, 7, 0, 5, 6, 6],
            [11, 9, 9, 8, 5, 0, 5, 5],
            [12, 10, 10, 9, 6, 5, 0, 4],
            [12, 10, 10, 9, 6, 5, 4, 0],
        ]
    )

    B = torch.tensor([[0, 2, 3.0], [2, 0, 4], [3, 4, 0]])

    n = A.shape[-1] - B.shape[-1]
    B = torch.nn.functional.pad(B, pad=(0, n, 0, n), mode="constant", value=0.0)
    return torch.stack((A, B))


@pytest.fixture
def batch_non_symm_matrices():
    B = torch.tensor([[0, 2, 3.0], [2, 0, 4], [11, 4, 0]])

    return B.unsqueeze(0)


def test_eigendecomposition(batch_distance_matrices):
    eigen_vals, eigen_vecs = get_eig_vals_and_vecs(batch_distance_matrices)
    assert eigen_vals.shape == batch_distance_matrices.shape[:-1]
    assert eigen_vecs.shape == batch_distance_matrices.shape
    val_vec_mul = torch.einsum("bi,bji->bji", eigen_vals, eigen_vecs)
    mat_vec_mul = torch.einsum("bij,bjk->bik", batch_distance_matrices, eigen_vecs)
    assert torch.allclose(val_vec_mul, mat_vec_mul, rtol=1e-5, atol=1e-5)


def test_raises_assert_error_when_matrix_not_symm(batch_non_symm_matrices):
    with pytest.raises(Exception) as exc_info:
        _ = get_eig_vals_and_vecs(batch_non_symm_matrices)
    assert (
        str(exc_info.value)
        == f"A = {batch_non_symm_matrices} is not a batch of symmetric matrices"
    )


def test_pca(batch_distance_matrices):
    for k in range(2, 6):
        Xrec = PCA(batch_distance_matrices, k)
        assert Xrec.shape == batch_distance_matrices.shape[:-1] + (k,)


def test_raises_assert_error_when_k_leq_0(batch_distance_matrices):
    for k in (-1, 0):
        with pytest.raises(Exception) as exc_info:
            _ = PCA(batch_distance_matrices, k)
        assert (
            str(exc_info.value)
            == f"Rank k must be greater than 0, but k = {k} was given."
        )


def test_hmds(batch_distance_matrices):
    for k in (3, 8, 10):
        Xrec = hmds(batch_distance_matrices, k)
        assert Xrec.shape == batch_distance_matrices.shape[:-1] + (k,)
        ball = PoincareBall()
        Xrec = ball.projx_from_hyperboloid(Xrec)
        assert ball._check_point_on_manifold(Xrec)
        assert not torch.all(torch.isnan(Xrec))
