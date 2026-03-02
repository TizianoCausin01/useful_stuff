from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np


"""
center_gram
Center a Gram matrix.
"""
def center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

"""
hsic
Hilbert-Schmidt Independence Criterion (biased).
"""
def hsic(K, L):
    n = K.shape[0]
    Kc, Lc = center_gram(K), center_gram(L)
    trace_test = np.trace(Kc @ Lc) / (n - 1) ** 2
    dot_test = np.dot(Kc.flatten(), Lc.flatten())
    return np.trace(Kc @ Lc) / (n - 1) ** 2



"""
cka
Compute CKA between two representations X, Y.

Parameters
----------
X : ndarray of shape (n_samples, d1)
Y : ndarray of shape (n_samples, d2)
kernel : str or callable
    - "linear": use linear kernel
    - any kernel name supported by sklearn (e.g. "rbf", "poly")
kwargs : extra arguments for sklearn pairwise_kernels

Returns
-------
cka_value : float
    Centered Kernel Alignment value in [0, 1].
"""
def cka(X, Y, kernel="linear", **kwargs):
    # Build Gram matrices
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    else:
        K = pairwise_kernels(X, metric=kernel, **kwargs)
        L = pairwise_kernels(Y, metric=kernel, **kwargs)
    
    # Compute normalized HSIC
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)
    
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-12)


"""
cka_minibatch
Minibatch version of CKA in PyTorch.

Parameters
----------
X : tensor, shape (batch_size, d1)
Y : tensor, shape (batch_size, d2)
kernel : str
    "linear" or "rbf"
kwargs : parameters for RBF kernel (e.g., gamma)
"""
def cka_minibatch(X, Y, kernel="linear", **kwargs):
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    else:
        K = pairwise_kernels(X, metric=kernel, **kwargs)
        L = pairwise_kernels(Y, metric=kernel, **kwargs)
    
    hsic_xy = hsic_unbiased(K, L)
    hsic_xx = hsic_unbiased(K, K)
    hsic_yy = hsic_unbiased(L, L)
    #return hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-12)
    return hsic_xy, hsic_xx, hsic_yy 



"""
hsic_unbiased
Unbiased HSIC estimator (Song et al. 2007) in NumPy.
"""
def hsic_unbiased(K, L):
    n = K.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 samples for unbiased HSIC")

    # make a copy and zero diagonals
    K = K.copy()
    L = L.copy()
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    ones = np.ones((n, 1))

    term1 = np.trace(K @ L).item()  
    term2 = ((ones.T @ K @ ones) * (ones.T @ L @ ones) / ((n - 1) * (n - 2))).item()
    term3 = (2 * (ones.T @ (K @ L) @ ones) / (n - 2)).item()
    return float((term1 + term2 - term3) / (n * (n - 3)))

def cka_batch_collection(xy, xx, yy):
    return xy / (np.sqrt(xx) *np.sqrt(yy)+ 1e-12)
