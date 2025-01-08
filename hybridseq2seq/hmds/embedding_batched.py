from typing import Tuple

import torch
from torch import linalg as LA


def get_eig_vals_and_vecs(
    A: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine the Eigenvalues and corresponding Eigenvectors (eigendecomposition) of a batch A.
        A comprises symmetric matrices:
            eigenvectors(A.transpose(1,2) @ A) == eigenvectors(A)

    Parameters
    ----------
    A : torch.Tensor: (bs, n, n)
        A batch of symmetric matrices

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The approximations for the (eigenvalues, and eigenvectors) of each matrix in A

    Raises
    ------
    AssertionError
        if the input tensor A does not contain symmetric matrices
    """
    assert len(A.shape) == 3 and torch.allclose(
        A, A.transpose(2, 1), rtol=1e-5, atol=1e-8
    ), f"A = {A} is not a batch of symmetric matrices"

    # Compute the eigenvalues and right eigenvectors of a square array (..,n,n)
    lambdas, U = LA.eig(torch.matmul(A.transpose(2, 1), A))
    lambdas = lambdas.float()
    U = U.float()  # (bs, n, n)

    # Sort eigenvectors from largest to smallest corresponding eigenvalue
    index = torch.argsort(torch.abs(lambdas), dim=-1, descending=True)  # (bs, n)
    index = index.unsqueeze(1)  # (bs, None, n)
    U = torch.take_along_dim(U, index, dim=-1)

    # A = U @ LAMBDA @ U.T where LAMBDA is the diagonal matrix with eigenvalues
    # Find LAMBDA = U.T @ A @ U
    LAMBDA = torch.matmul(torch.matmul(U.transpose(2, 1), A), U)  # U.T @ A @ U
    lambdas_signed = torch.diagonal(LAMBDA, dim1=1, dim2=2)  # (bs, n)

    return lambdas_signed, U


def PCA(Z: torch.Tensor, k: int) -> Tuple[torch.Tensor, int]:
    """Run Principal Component Analysis on A to find the k most significant
    non-negative eigenvalues to recover X.

    Parameters
    ----------
    Z : torch.Tensor : (bs, n, n)
        A batch of symmetric matrices.
    k : int
        The number of non-negative eigenvalues to retain

    Returns
    -------
    torch.Tensor
        The recovered X matrix

    Raises
    ------
    AssertionError
        if the number of eigenvalues k is less than 0
    """
    assert k > 0, f"Rank k must be greater than 0, but k = {k} was given."

    lambdasM, usM = get_eig_vals_and_vecs(Z)

    # Among the n eigenvalues ordered by significance take the non-negative ones up to k
    bs = Z.shape[0]

    # Positive (and pad=0) eignevalues
    mask_neg = lambdasM >= 0

    Xrec_all = []
    # one matrix at a time
    for i in range(bs):
        # Filter out negative eigenvalues and corresponding eigenvectors
        lambdasM_pos = lambdasM[i][mask_neg[i]]
        usM_pos = usM[i][:, mask_neg[i]]  # (n, n')
        idx = min(usM_pos.shape[-1], k)

        # A = U @ LAMBDA @ U.T => A = X @ X.T => X = U @ sqrt(LAMBDA)
        # Xrec is an approximation of X
        # Xrec does not consider the larget eigenvalue (the only negative eigenvalue)
        # and its corresponding eigenvector
        # Xrec takes the remaining eigenvalues-eigenvectors up to k
        Xrec = torch.matmul(
            usM_pos[:, 0:idx], torch.diag(torch.sqrt(lambdasM_pos[0:idx]))
        )
        if usM_pos.shape[-1] < k:
            # Pad to fixed dimension k
            Xrec = torch.nn.functional.pad(
                Xrec, pad=(0, k - usM_pos.shape[-1]), mode="constant", value=0.0
            )
        Xrec_all.append(Xrec)
    return torch.stack(Xrec_all)


def hmds(D: torch.Tensor, k: int, scale: float = 1.0) -> torch.Tensor:
    Y = torch.cosh(D * scale)
    # this is a Gans model set of points
    Xrec = PCA(-Y, k)  # (bs, n, k)
    return Xrec
