"""
PCA risk allocation
====================

Computes portfolio weights by distributing risk across the principal components
(eigenportfolios) of the covariance matrix, following the spectral allocation
method in *Advances in Financial Machine Learning* (Lopez de Prado, Section
2.4.2).

Given a target volatility and a desired split of risk across components, the
weights are obtained by scaling each eigenportfolio by the square root of its
allocated variance and projecting back into the asset space. By default all risk
is placed on the lowest-variance component, a common starting point for
mean-reversion baskets.

Dependencies:
    - numpy
"""

import numpy as np
from numpy.typing import NDArray


def pcaweights(
    cov: NDArray[np.float64], riskDistr: NDArray[np.float64] | None = None, riskTarget: float = 1.0
) -> NDArray[np.float64]:
    """
    Computes portfolio weights based on a spectral decomposition (PCA) of the covariance matrix.

    This function calculates the weights required to achieve a specific risk distribution
    across the Principal Components (Eigenportfolios) of the covariance matrix.

    Formula:
        omega = P * beta
        beta = riskTarget * sqrt(riskDistr / eigenvalues)

    Args:
        cov (NDArray[np.float64]): The square covariance matrix (N x N) of asset returns.
        riskDistr (NDArray[np.float64] | None): A 1D array (N,) defining the desired contribution
            of each Principal Component to the total risk. Must sum to 1.
            If None, defaults to allocating 100% of risk to the last component
            (smallest eigenvalue), often used for mean-reversion strategies.
        riskTarget (float): The scalar volatility target for the portfolio. Defaults to 1.0.

    Returns:
        NDArray[np.float64]: The calculated asset weights vector (omega) of shape (N,).

    Raises:
        ValueError: If `cov` is not square, or if `riskDistr` dimensions do not match,
                    or if `riskDistr` does not sum to 1.
    """

    # 1. Validation: Ensure Covariance Matrix is square
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance Matrix must be square")

    n = cov.shape[0]

    # 2. Default Risk Distribution
    # If not provided, allocate all risk to the component with the smallest eigenvalue (last one after sort)
    if riskDistr is None:
        riskDistr = np.zeros(n)
        riskDistr[-1] = 1.0

    # 3. Validation: Ensure Risk Distribution vector matches dimensions
    if riskDistr.ndim != 1 or riskDistr.shape[0] != n:
        raise ValueError("Risk distribution must be a vector with same dimension as covariance matrix")

    # 4. Validation: Check if risk distribution sums to 1 (within floating point tolerance)
    if not np.isclose(riskDistr.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The risk distribution sum must be 1.")

    # 5. Spectral Decomposition
    # d: Eigenvalues (variances of PCs), P: Eigenvectors (loadings of PCs)
    # Note: eigh returns eigenvalues in ascending order
    d, P = np.linalg.eigh(cov)

    # 6. Sort by Eigenvalue (Descending Order)
    # We reorder from largest variance (PC1) to smallest (Noise)
    idx = d.argsort()[::-1]
    d, P = d[idx], P[:, idx]

    # 7. Computation of Beta (Allocated Risk per Component)
    # beta scales the exposure to each PC based on the target risk and the PC's volatility (sqrt(d))
    beta = riskTarget * np.sqrt(riskDistr / d)

    # 8. Transform back to Asset Basis (Omega)
    # Project the PC weights (beta) back into the original asset space using Eigenvectors (P)
    omega = P @ beta

    return omega
