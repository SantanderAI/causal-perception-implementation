# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Distance functions for comparing 1D empirical distributions.

Implements:
  - Wasserstein-2 (W2): exact 1D formula via sorted quantiles
  - KL divergence: kernel density estimation (KDE) based
  - Total Variation (TV): histogram-based

All functions take two 1D arrays of samples and return a scalar distance.
"""

import numpy as np
from scipy import stats


def wasserstein2(samples1, samples2):
    """
    1D Wasserstein-2 distance between two empirical distributions.

    For 1D distributions, W2 = sqrt(1/n * sum((F1_inv(i) - F2_inv(i))^2))
    where F_inv are the sorted samples (quantile functions).

    Parameters
    ----------
    samples1, samples2 : array-like, shape (n,)
        Samples from the two distributions (must have same length).

    Returns
    -------
    float
        W2 distance.
    """
    s1 = np.sort(np.asarray(samples1, dtype=float))
    s2 = np.sort(np.asarray(samples2, dtype=float))
    if len(s1) != len(s2):
        raise ValueError(f"Sample sizes must match: {len(s1)} != {len(s2)}")
    return float(np.sqrt(np.mean((s1 - s2) ** 2)))


def kl_divergence(samples1, samples2, bw_method="scott"):
    """
    KL divergence D_KL(P || Q) estimated via kernel density estimation.

    Parameters
    ----------
    samples1 : array-like, shape (n,)
        Samples from P.
    samples2 : array-like, shape (n,)
        Samples from Q.
    bw_method : str
        Bandwidth method for KDE (default: 'scott').

    Returns
    -------
    float
        Estimated KL divergence.
    """
    s1 = np.asarray(samples1)
    s2 = np.asarray(samples2)

    # Fit KDEs
    kde_p = stats.gaussian_kde(s1, bw_method=bw_method)
    kde_q = stats.gaussian_kde(s2, bw_method=bw_method)

    # Evaluate on P's samples
    p_vals = kde_p(s1)
    q_vals = kde_q(s1)

    # Avoid log(0) by clipping
    q_vals = np.clip(q_vals, 1e-10, None)
    p_vals = np.clip(p_vals, 1e-10, None)

    return np.mean(np.log(p_vals / q_vals))


def total_variation(samples1, samples2, n_bins=50):
    """
    Total Variation distance estimated via histograms.

    TV(P, Q) = 0.5 * sum |p_i - q_i|

    Parameters
    ----------
    samples1, samples2 : array-like, shape (n,)
        Samples from the two distributions.
    n_bins : int
        Number of histogram bins (default: 50).

    Returns
    -------
    float
        Estimated TV distance in [0, 1].
    """
    s1 = np.asarray(samples1)
    s2 = np.asarray(samples2)

    # Common bin edges
    lo = min(s1.min(), s2.min())
    hi = max(s1.max(), s2.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    # Normalized histograms
    h1, _ = np.histogram(s1, bins=bins, density=False)
    h2, _ = np.histogram(s2, bins=bins, density=False)
    p1 = h1 / max(h1.sum(), 1)
    p2 = h2 / max(h2.sum(), 1)

    return float(0.5 * np.sum(np.abs(p1 - p2)))


def compute_all_distances(samples1, samples2):
    """
    Compute all three distances between two 1D sample arrays.

    Returns
    -------
    dict
        {"W2": float, "KL": float, "TV": float}
    """
    return {
        "W2": wasserstein2(samples1, samples2),
        "KL": kl_divergence(samples1, samples2),
        "TV": total_variation(samples1, samples2),
    }


def print_distances(label, d):
    """Print distances in a compact format."""
    print(f"  {label}: W2 = {d['W2']:.4f}, KL = {d['KL']:.4f}, TV = {d['TV']:.4f}")


if __name__ == "__main__":
    # Quick test with known distributions
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, size=300)
    b = rng.normal(0.6, 0.1, size=300)

    d = compute_all_distances(a, b)
    print("Test: N(0.5, 0.1) vs N(0.6, 0.1), n=300")
    print(f"  W2 = {d['W2']:.4f}")
    print(f"  KL = {d['KL']:.4f}")
    print(f"  TV = {d['TV']:.4f}")
