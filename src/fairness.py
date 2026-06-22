# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Fairness metrics for credit decision experiments.

Provides reusable functions for demographic-parity gaps, equal-opportunity
gaps, and decision disagreement between competing receivers.
"""

import numpy as np


def fairness_metrics(y_true, y_hat, a, tau=0.5):
    """
    Compute fairness metrics for a single receiver.

    Parameters
    ----------
    y_true : array-like
        Ground-truth outcome labels (1 = positive class).
    y_hat : array-like
        Predicted risk scores.
    a : array-like
        Protected attribute (0 = female, 1 = male).
    tau : float
        Decision threshold; accept when y_hat >= tau.

    Returns
    -------
    dict
        Acceptance rates, DP gap, TPR/FPR by group, and EO gaps.
    """
    d = (y_hat >= tau).astype(int)
    female = a == 0
    male = a == 1

    accept_f = np.mean(d[female])
    accept_m = np.mean(d[male])
    accept_all = np.mean(d)
    dp_gap = accept_f - accept_m  # signed: positive = females accepted more

    # TPR = P(D=1 | Y=1, A=a)
    tpr_f = np.mean(d[female & (y_true == 1)])
    tpr_m = np.mean(d[male & (y_true == 1)])
    # FPR = P(D=1 | Y=0, A=a)
    fpr_f = np.mean(d[female & (y_true == 0)])
    fpr_m = np.mean(d[male & (y_true == 0)])

    return {
        "accept_all": accept_all,
        "accept_female": accept_f,
        "accept_male": accept_m,
        "dp_gap": dp_gap,
        "tpr_female": tpr_f,
        "tpr_male": tpr_m,
        "tpr_gap": tpr_f - tpr_m,
        "fpr_female": fpr_f,
        "fpr_male": fpr_m,
        "fpr_gap": fpr_f - fpr_m,
    }


def decision_disagreement(y_hat_a, y_hat_b, a, tau):
    """
    Compute decision disagreement between two receivers.

    Parameters
    ----------
    y_hat_a, y_hat_b : array-like
        Predicted risk scores from receiver A and B.
    a : array-like
        Protected attribute (0 = female, 1 = male).
    tau : float
        Decision threshold; accept when y_hat >= tau.

    Returns
    -------
    dict
        Overall and group-stratified disagreement rates, counts, and
        directional flip counts.
    """
    d_a = (y_hat_a >= tau).astype(int)
    d_b = (y_hat_b >= tau).astype(int)
    disagree = d_a != d_b

    r1_grant_r2_deny = (d_a == 1) & (d_b == 0)
    r1_deny_r2_grant = (d_a == 0) & (d_b == 1)

    return {
        "disagree_rate": np.mean(disagree),
        "disagree_female": np.mean(disagree[a == 0]),
        "disagree_male": np.mean(disagree[a == 1]),
        "n_disagree": int(disagree.sum()),
        "n_disagree_female": int(disagree[a == 0].sum()),
        "n_disagree_male": int(disagree[a == 1].sum()),
        "r1_grant_r2_deny": int(r1_grant_r2_deny.sum()),
        "r1_deny_r2_grant": int(r1_deny_r2_grant.sum()),
    }
