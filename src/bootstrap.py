"""
Non-parametric percentile bootstrap for 95% CIs.
Works on any list of scalar metric values (Track 1 or Track 2).
"""
import numpy as np


def bootstrap_ci(
    values: list[float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Return {mean, ci_low, ci_high} using the percentile bootstrap.

    Parameters
    ----------
    values : per-question metric scores (length = dataset size)
    n      : number of bootstrap resamples (1 000 per spec)
    alpha  : two-sided error rate (0.05 -> 95% CI)
    seed   : RNG seed for reproducibility

    Returns NaN for all fields if values is empty.
    """
    if not values:
        nan = float("nan")
        return {"mean": nan, "ci_low": nan, "ci_high": nan}

    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())

    rng = np.random.default_rng(seed)
    # Sample n bootstrap datasets and compute their means
    indices = rng.integers(0, len(arr), size=(n, len(arr)))
    bootstrap_means = arr[indices].mean(axis=1)

    lo = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {"mean": mean, "ci_low": lo, "ci_high": hi}
