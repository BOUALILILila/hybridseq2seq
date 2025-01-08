from typing import Tuple

import numpy as np
from numpy import linalg as LA

# def gans_to_poincare(X: np.ndarray) -> np.ndarray:
#     """ Projection from Gans Model to Poincaré Model.

#     Parameters
#     ----------
#     X : np.ndarray
#         Batch of sequence embeddings (bs, n, k).

#     Returns
#     -------
#     np.ndarray
#         The embeddings in the Poincaré Model
#     """
#     Y = 1.0 + np.sqrt(1.0 + LA.norm(X, axis=2)**2)
#     Y = np.expand_dims(Y, axis=-1)
#     return X/Y


def get_eig_vals_and_vecs(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine the Eigenvalues and corresponding Eigenvectors (eigendecomposition) of a batch A.
        A comprises symmetric matrices:
            eigenvectors(A.transpose(1,2) @ A) == eigenvectors(A)

    Parameters
    ----------
    A : np.ndarray: (bs, n, n)
        A batch of symmetric matrices

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The approximations for the (eigenvalues, and eigenvectors) of each matrix in A

    Raises
    ------
    AssertionError
        if the input tensor A does not contain symmetric matrices
    """
    assert len(A.shape) == 3 and np.allclose(
        A, A.transpose(0, 2, 1), rtol=1e-5, atol=1e-8
    ), f"A = {A} is not a batch of symmetric matrices"

    # Compute the eigenvalues and right eigenvectors of a square array (..,n,n)
    lambdas, U = LA.eig(np.matmul(A.transpose(0, 2, 1), A))
    lambdas = lambdas.astype(np.float32)
    U = U.astype(np.float32)  # (bs, n, n)

    # Sort eigenvectors from largest to smallest corresponding eigenvalue
    index = np.abs(lambdas).argsort(axis=-1)
    index = np.flip(index, -1)  # reverse order
    index = np.expand_dims(index, axis=1)  # (bs, None, n)
    U = np.take_along_axis(U, index, axis=-1)

    # A = U @ LAMBDA @ U.T where LAMBDA is the diagonal matrix with eigenvalues
    # Find LAMBDA = U.T @ A @ U
    LAMBDA = np.matmul(np.matmul(U.transpose(0, 2, 1), A), U)  # U.T @ A @ U
    lambdas_signed = np.diagonal(LAMBDA, axis1=1, axis2=2)  # (bs, n)

    return lambdas_signed, U


def PCA(Z: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    """Run Principal Component Analysis on A to find the k most significant
    non-negative eigenvalues to recover X.

    Parameters
    ----------
    Z : np.ndarray : (bs, n, n)
        A batch of symmetric matrices.
    k : int
        The number of non-negative eigenvalues to retain

    Returns
    -------
    np.ndarray
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
        Xrec = np.matmul(usM_pos[:, 0:idx], np.diag(np.sqrt(lambdasM_pos[0:idx])))
        if usM_pos.shape[-1] < k:
            # Pad to fixed dimension k
            Xrec = np.pad(
                array=Xrec,
                pad_width=((0, 0), (0, k - usM_pos.shape[-1])),
                mode="constant",
                constant_values=(0,),
            )
        Xrec_all.append(Xrec)
    return np.stack(Xrec_all)


def hmds(D: np.ndarray, k: int, scale: float = 1.0) -> np.ndarray:
    Y = np.cosh(D * scale)
    # this is a Gans model set of points
    Xrec = PCA(-Y, k)  # (bs, n, k)
    # project from Gans Model to Poincaré Ball model
    # X = gans_to_poincare(Xrec)
    return Xrec
