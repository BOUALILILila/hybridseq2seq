from typing import Tuple

import torch
from torch import linalg as LA


def get_eig_vals_and_vecs(
    A: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine the Eigenvalues and corresponding Eigenvectors (eigendecomposition) of a symmetric matrix A.
        A is a symmetric matrice:
            eigenvectors(A.transpose(1,2) @ A) == eigenvectors(A)

    Parameters
    ----------
    A : torch.Tensor: (n, n)
        A symmetric matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The approximations for the (eigenvalues, and eigenvectors) of A

    Raises
    ------
    AssertionError
        if the input tensor A is not symmetric
    """
    assert len(A.shape) == 2 and torch.allclose(
        A, A.T, rtol=1e-5, atol=1e-8
    ), f"A = {A} is not symmetric"

    # Compute the eigenvalues and right eigenvectors of a square array (n,n)
    lambdas, U = LA.eig(torch.matmul(A.transpose(1, 0), A))
    lambdas = lambdas.float()
    U = U.float()  # (n, n)

    # Sort eigenvectors from largest to smallest corresponding eigenvalue
    index = torch.argsort(torch.abs(lambdas), dim=-1, descending=True)  # (n)
    index = index.unsqueeze(0)  # (1, n)
    U = torch.take_along_dim(U, index, dim=-1)

    # A = U @ LAMBDA @ U.T where LAMBDA is the diagonal matrix with eigenvalues
    # Find LAMBDA = U.T @ A @ U
    LAMBDA = torch.matmul(torch.matmul(U.T, A), U)  # U.T @ A @ U
    lambdas_signed = torch.diagonal(LAMBDA, dim1=0, dim2=1)  # (n)

    return lambdas_signed, U


def PCA(Z: torch.Tensor, k: int) -> Tuple[torch.Tensor, int]:
    """Run Principal Component Analysis on A to find the k most significant
    non-negative eigenvalues to recover X.

    Parameters
    ----------
    Z : torch.Tensor : (n, n)
        A symmetric distance matrice.
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
    # Positive (and pad=0) eignevalues
    mask_neg = lambdasM >= 0

    # Filter out negative eigenvalues and corresponding eigenvectors
    lambdasM_pos = lambdasM[mask_neg]
    usM_pos = usM[:, mask_neg]  # (n, n')
    idx = min(usM_pos.shape[-1], k)

    # A = U @ LAMBDA @ U.T => A = X @ X.T => X = U @ sqrt(LAMBDA)
    # Xrec is an approximation of X
    # Xrec does not consider the larget eigenvalue (the only negative eigenvalue)
    # and its corresponding eigenvector
    # Xrec takes the remaining eigenvalues-eigenvectors up to k
    Xrec = torch.matmul(usM_pos[:, 0:idx], torch.diag(torch.sqrt(lambdasM_pos[0:idx])))
    if usM_pos.shape[-1] < k:
        # Pad to fixed dimension k
        Xrec = torch.nn.functional.pad(
            Xrec, pad=(0, k - usM_pos.shape[-1]), mode="constant", value=0.0
        )
    return Xrec


def _hmds(D: torch.Tensor, k: int, scale: float = 1.0) -> torch.Tensor:
    Y = torch.cosh(D * scale)
    # this is a Gans model set of points
    Xrec = PCA(-Y, k)  # (n, k)
    return Xrec


def hmds(
    distance_matrix: torch.Tensor, mask: torch.Tensor, k: int, scale: float = 1.0
) -> torch.Tensor:
    # mask select
    mat_mask = mask.repeat([distance_matrix.shape[-1], 1])
    mat_mask = mat_mask * mat_mask.T
    D_masked = torch.masked_select(distance_matrix, mat_mask.bool())
    masked_len = int(torch.sum(mask).item())
    D_masked = D_masked.view(masked_len, masked_len)
    # run hmds
    masked_Xrec = _hmds(D_masked, k, scale)
    # mask scatter
    mask_scatter = mask.view(-1, 1).repeat(1, k).bool()
    Xrec = (
        torch.zeros(distance_matrix.shape[0], k)
        .to(masked_Xrec.dtype)
        .to(masked_Xrec.device)
    )
    Xrec = Xrec.masked_scatter_(mask_scatter, masked_Xrec)
    return Xrec
