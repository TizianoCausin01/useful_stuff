import os, yaml, sys
import numpy as np
from scipy import stats

sys.path.append("../..")


"""
permutation_test_corr
Performs a permutation test for correlation between two variables.
Supports Pearson or Spearman correlation and one- or two-sided hypothesis testing.

INPUT:
    - x: np.ndarray(float) -> (n,) first variable
    - y: np.ndarray(float) -> (n,) second variable
    - n_perm: int -> number of permutations used to build null distribution
    - corr_type: str -> correlation type ("pearson" or "spearman")
    - test_type: str -> type of statistical test:
        * "two"   -> two-sided test (|null| >= |obs|)
        * "upper" -> upper-tail test (null >= obs)
        * "lower" -> lower-tail test (null <= obs)

OUTPUT:
    - obs: float -> observed correlation between x and y
    - p: float -> permutation-based p-value
    - null_dist: np.ndarray(float) -> (n_perm,) distribution of correlation values under the null hypothesis
"""
def permutation_test_corr(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    corr_type: str = "pearson",  # "pearson" or "spearman"
    test_type: str = "two",   # "two", "upper", "lower"
) -> tuple[float, float, np.ndarray]:

    x, y = np.asarray(x), np.asarray(y)

    if corr_type == "pearson":
        corr_fn = lambda a, b: np.corrcoef(a, b)[0, 1]
    elif corr_type == "spearman":
        corr_fn = lambda a, b: stats.spearmanr(a, b).statistic
    else:
        raise ValueError(f"corr_type must be 'pearson' or 'spearman', got '{corr_type}'")

    obs = corr_fn(x, y)

    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        null_dist[i] = corr_fn(x, np.random.permutation(y))

    if test_type == "two":
        p = np.mean(np.abs(null_dist) >= np.abs(obs))
    elif test_type == "upper":
        p = np.mean(null_dist >= obs)
    elif test_type == "lower":
        p = np.mean(null_dist <= obs)
    else:
        raise ValueError(f"test_type must be 'two', 'upper', or 'lower', got '{test_type}'")

    return obs, p, null_dist
