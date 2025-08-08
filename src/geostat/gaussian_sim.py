"""Fast Gaussian field generation."""
import sys
import numpy as np
from scipy.linalg import toeplitz


def fast_gaussian(dimension, sdev, corr, num_samples=1):
    """

    Generates random vector from distribution satisfying Gaussian variogram in dimension up to 3-d.

    Parameters
    ----------
    dimension : int
        Dimension of the grid.

    sdev : float
        Standard deviation.

    corr : float or array-like
        Correlation length, in units of block length.
        If a single float is provided, it represents the correlation length in all directions.
        If an array-like object with length 3 is provided, it represents the correlation length in the x-, y-, and z-directions.

    num_samples : int, optional
        Number of samples to generate. Default is 1.
        If greater than 1, the function will return an array of shape (dimension, num_samples).

    Returns
    -------
    x : array-like
        The generated random vectors of shape (dimension, num_samples).

    Notes
    -----
    The parametrization of the grid is assumed to have size dimension, if dimension is a vector,
    or [dimension,1] if dimension is scalar. The coefficients of the grid is assumed to be reordered
    columnwise into the parameter vector. The grid is assumed to have a local basis.

    Example of use:

    Want to generate a field on a 3-d grid with dimension m x n x p, with correlation length a along first coordinate
    axis, b along second coordinate axis, c alone third coordinate axis, and standard deviation sigma:

    x=fast_gaussian(np.array([m, n, p]),np.array([sigma]),np.array([a b c]))

    If the dimension is n x 1 one can write

    x=fast_gaussian(np.array([n]),np.array([sigma]),np.array([a]))

    If the correlation length is the same in all directions:

    x=fast_gaussian(np.array([m n p]),np.array([sigma]),np.array([a]))

    The properties on the Kronecker product behind this algorithm can be found in
    Horn & Johnson: Topics in Matrix Analysis, Cambridge UP, 1991.

    Note that we add a small number on the diagonal of the covariance matrix to avoid numerical problems with Cholesky
    decomposition (a nugget effect).

    Also note that reshape with order='F' is used to keep the code identical to the Matlab code.

    The method was invented and implemented in Matlab by Geir Nævdal in 2011.
    Memory-efficient implementation for batch generation of samples was added in 2025.
    """

    if len(dimension) == 0:
        sys.exit("fast_gaussian: Wrong input, dimension should have length at least 1")
    m = dimension[0]
    n = 1
    p = None
    if len(dimension) > 1:
        n = dimension[1]
    dim = m * n
    if len(dimension) > 2:
        p = dimension[2]
        dim = dim * p
    if len(dimension) > 3:
        sys.exit("fast_gaussian: Wrong input, dimension should have length at most 3")

    if len(sdev) > 1:
        std = 1
    else:
        std = sdev

    if len(corr) == 0:
        sys.exit("fast_gaussian: Wrong input, corr should have length at least 1")
    if len(corr) == 1:
        corr = np.append(corr, corr[0])
    if len(corr) == 2 and p is not None:
        corr = np.append(corr, corr[1])
    corr = np.maximum(corr, 1)

    dist1 = np.arange(m) / corr[0]
    t1 = toeplitz(dist1)
    t1 = std * np.exp(-t1 ** 2) + 1e-10 * np.eye(m)
    cholt1 = np.linalg.cholesky(t1)

    if corr[0] == corr[1] and n == m:
        cholt2 = cholt1
    else:
        dist2 = np.arange(n) / corr[1]
        t2 = toeplitz(dist2)
        t2 = std * np.exp(-t2 ** 2) + 1e-10 * np.eye(n)
        cholt2 = np.linalg.cholesky(t2)

    cholt3 = None
    if p is not None:
        dist3 = np.arange(p) / corr[2]
        t3 = toeplitz(dist3)
        t3 = np.exp(-t3 ** 2) + 1e-10 * np.eye(p)
        cholt3 = np.linalg.cholesky(t3)

    x = np.random.randn(dim, num_samples)

    # Memory-efficient multiplication without explicit large Kronecker product
    if p is None:
        x = x.reshape(m, n, num_samples, order='F')
        x = np.tensordot(cholt1, x, axes=([1], [0]))
        x = np.tensordot(cholt2, x, axes=([1], [1]))
    else:
        x = x.reshape(m, n, p, num_samples, order='F')
        if n <= p:
            x = np.tensordot(cholt1, x, axes=([1], [0]))
            x = np.tensordot(cholt2, x, axes=([1], [1]))
            x = np.tensordot(cholt3, x, axes=([1], [2]))
        else:
            x = np.tensordot(cholt1, x, axes=([1], [0]))
            x = np.tensordot(cholt2, x, axes=([1], [1]))
            x = np.tensordot(cholt3, x, axes=([1], [2]))

    # Reshape back to (dim, num_samples) (order='C' is used here to match the original function's output)
    x = x.reshape((dim, num_samples), order='C')

    if len(sdev) > 1:
        if len(sdev) == dim:
            x = sdev[:, None] * x
        else:
            sys.exit('fast_gaussian: Inconsistent dimension of sdev')

    return x

