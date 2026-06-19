# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Shared engine for running causal perception experiments.

Provides reusable functions for fitting competing SCMs, computing
perception distances, and running bootstrap confidence intervals.
"""

import numpy as np

from src.distances import compute_all_distances
from src.linear_anm import LinearANM


def fit_scms(train, edges1, edges2, outcome="Y", y_model=None):
    """
    Fit two competing SCMs on the same training data.

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    edges1, edges2 : dict
        DAG structures as {node: [parent_nodes]}.
    outcome : str
        Name of the outcome node (default: "Y").
    y_model : sklearn classifier, optional
        Custom classifier for the outcome node.

    Returns
    -------
    tuple of (LinearANM, LinearANM)
    """
    scm1 = LinearANM(edges=edges1, outcome=outcome, y_model=y_model).fit(train)
    scm2 = LinearANM(edges=edges2, outcome=outcome, y_model=y_model).fit(train)
    return scm1, scm2


def run_perception(scm1, scm2, test, variable, values, rung="interventional"):
    """
    Core perception computation between two fitted SCMs.

    Parameters
    ----------
    scm1, scm2 : LinearANM
        Fitted competing SCMs.
    test : pd.DataFrame
        Test data.
    variable : str
        Variable to intervene on (e.g. "A", "C").
    values : list
        Intervention values (e.g. [0, 1] or [26, 43]).
    rung : str
        "interventional" (2nd rung) or "counterfactual" (3rd rung).

    Returns
    -------
    dict with keys:
        "between"     : {value: {"W2": ..., "KL": ..., "TV": ...}}
        "aggregated"  : {"W2": ..., "KL": ..., "TV": ...}
        "within_scm1" : {"W2": ..., "KL": ..., "TV": ...}
        "within_scm2" : {"W2": ..., "KL": ..., "TV": ...}
        "probs"       : {("scm1"|"scm2", value): np.ndarray}
    """
    method = "counterfactual" if rung == "counterfactual" else "intervene"

    # Get distributions for each SCM x each intervention value
    probs = {}
    for label, scm in [("scm1", scm1), ("scm2", scm2)]:
        for v in values:
            _, p = getattr(scm, method)(test, {variable: v})
            probs[(label, v)] = p

    # Between-SCM distances (M1 vs M2, same intervention)
    between = {}
    for v in values:
        between[v] = compute_all_distances(probs[("scm1", v)], probs[("scm2", v)])

    # Aggregated (mean across interventions)
    metrics = ["W2", "KL", "TV"]
    aggregated = {k: float(np.mean([between[v][k] for v in values])) for k in metrics}

    # Within-SCM distances (value[0] vs value[1], same SCM)
    within_scm1 = compute_all_distances(probs[("scm1", values[0])], probs[("scm1", values[1])])
    within_scm2 = compute_all_distances(probs[("scm2", values[0])], probs[("scm2", values[1])])

    return {
        "between": between,
        "aggregated": aggregated,
        "within_scm1": within_scm1,
        "within_scm2": within_scm2,
        "probs": probs,
    }


def perception_flag(distances, epsilon=0.1):
    """
    Compute perception flag for each metric.

    Parameters
    ----------
    distances : dict
        {"W2": float, "KL": float, "TV": float}
    epsilon : float
        Perception threshold.

    Returns
    -------
    dict : {"W2": 0|1, "KL": 0|1, "TV": 0|1}
    """
    return {k: int(v > epsilon) for k, v in distances.items()}


def bootstrap_ci(samples1, samples2, B=1000, alpha=0.05, rng=None):
    """
    Bootstrap 95% CI for W2, KL, TV between two paired sample arrays.

    Parameters
    ----------
    samples1, samples2 : np.ndarray
        Paired 1D arrays.
    B : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (default: 0.05 for 95% CI).
    rng : np.random.Generator, optional

    Returns
    -------
    dict : {metric: (point, lo, hi)}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(samples1)
    point = compute_all_distances(samples1, samples2)

    boot = {k: [] for k in point}
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        d = compute_all_distances(samples1[idx], samples2[idx])
        for k in boot:
            boot[k].append(d[k])

    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    result = {}
    for k in point:
        lo, hi = np.percentile(boot[k], [lo_pct, hi_pct])
        result[k] = (point[k], lo, hi)

    return result


def run_perception_bootstrap(
    scm1, scm2, test, variable, values, rung="interventional", B=1000, rng=None
):
    """
    Run perception with bootstrap CIs for all comparisons.

    Returns
    -------
    dict with keys:
        "between"     : {value: {metric: (point, lo, hi)}}
        "within_scm1" : {metric: (point, lo, hi)}
        "within_scm2" : {metric: (point, lo, hi)}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    method = "counterfactual" if rung == "counterfactual" else "intervene"

    probs = {}
    for label, scm in [("scm1", scm1), ("scm2", scm2)]:
        for v in values:
            _, p = getattr(scm, method)(test, {variable: v})
            probs[(label, v)] = p

    between = {}
    for v in values:
        between[v] = bootstrap_ci(probs[("scm1", v)], probs[("scm2", v)], B=B, rng=rng)

    within_scm1 = bootstrap_ci(probs[("scm1", values[0])], probs[("scm1", values[1])], B=B, rng=rng)
    within_scm2 = bootstrap_ci(probs[("scm2", values[0])], probs[("scm2", values[1])], B=B, rng=rng)

    return {
        "between": between,
        "within_scm1": within_scm1,
        "within_scm2": within_scm2,
    }


def format_distances(results, label="", epsilon=0.1):
    """Print perception results in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"PERCEPTION {label}")
    print(f"{'=' * 60}")

    for v, d in results["between"].items():
        print(f"\n  M1 vs M2 | do({v}):")
        print(f"    W2 = {d['W2']:.4f}, KL = {d['KL']:.4f}, TV = {d['TV']:.4f}")

    agg = results["aggregated"]
    print("\n  Aggregated (mean):")
    print(f"    W2 = {agg['W2']:.4f}, KL = {agg['KL']:.4f}, TV = {agg['TV']:.4f}")

    phi = perception_flag(agg, epsilon)
    print(f"\n  Perception flag (eps={epsilon}):")
    for k in ["W2", "KL", "TV"]:
        sym = ">" if phi[k] else "<="
        print(f"    phi({k}) = {phi[k]}  (d={agg[k]:.4f} {sym} {epsilon})")

    w1 = results["within_scm1"]
    w2 = results["within_scm2"]
    print(f"\n  Within M1: W2={w1['W2']:.4f}, KL={w1['KL']:.4f}, TV={w1['TV']:.4f}")
    print(f"  Within M2: W2={w2['W2']:.4f}, KL={w2['KL']:.4f}, TV={w2['TV']:.4f}")
