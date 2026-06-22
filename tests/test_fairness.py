# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.fairness."""

import numpy as np
import pytest

from src.fairness import decision_disagreement, fairness_metrics


def test_fairness_metrics_perfect_separation_by_group():
    """Known scores produce exact DP, TPR, and FPR gaps."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    a = np.array([0, 0, 0, 1, 1, 1])
    # Female: accept 2/3; Male: accept 1/3 -> dp_gap = 2/3 - 1/3 = 1/3
    y_hat = np.array([0.9, 0.8, 0.2, 0.8, 0.2, 0.1])

    m = fairness_metrics(y_true, y_hat, a, tau=0.5)

    assert m["accept_female"] == pytest.approx(2 / 3)
    assert m["accept_male"] == pytest.approx(1 / 3)
    assert m["dp_gap"] == pytest.approx(1 / 3)
    assert m["tpr_female"] == pytest.approx(1.0)
    assert m["tpr_male"] == pytest.approx(0.0)
    assert m["tpr_gap"] == pytest.approx(1.0)
    assert m["fpr_female"] == pytest.approx(0.0)
    assert m["fpr_male"] == pytest.approx(0.5)
    assert m["fpr_gap"] == pytest.approx(-0.5)


def test_fairness_metrics_tau_threshold():
    """Raising tau reduces acceptance rates."""
    y_true = np.array([1, 0, 1, 0])
    a = np.array([0, 0, 1, 1])
    y_hat = np.array([0.6, 0.4, 0.6, 0.4])

    low_tau = fairness_metrics(y_true, y_hat, a, tau=0.3)
    high_tau = fairness_metrics(y_true, y_hat, a, tau=0.7)

    assert low_tau["accept_all"] == pytest.approx(1.0)
    assert high_tau["accept_all"] == pytest.approx(0.0)


def test_fairness_metrics_empty_group_safe():
    """np.mean on empty slice returns nan when a group has no Y=1 rows."""
    y_true = np.array([0, 0, 1, 1])
    a = np.array([0, 0, 1, 1])
    y_hat = np.array([0.9, 0.8, 0.7, 0.6])

    m = fairness_metrics(y_true, y_hat, a, tau=0.5)

    assert np.isnan(m["tpr_female"])
    assert m["tpr_male"] == pytest.approx(1.0)


def test_decision_disagreement_symmetric():
    """Swapping receivers preserves overall disagreement rate."""
    y_hat_a = np.array([0.9, 0.2, 0.8, 0.1])
    y_hat_b = np.array([0.2, 0.9, 0.1, 0.8])
    a = np.array([0, 0, 1, 1])
    tau = 0.5

    dd_ab = decision_disagreement(y_hat_a, y_hat_b, a, tau)
    dd_ba = decision_disagreement(y_hat_b, y_hat_a, a, tau)

    assert dd_ab["disagree_rate"] == pytest.approx(dd_ba["disagree_rate"])
    assert dd_ab["n_disagree"] == dd_ba["n_disagree"]


def test_decision_disagreement_directional_counts():
    """Constructed scores yield exact flip counts."""
    y_hat_a = np.array([0.9, 0.1, 0.8, 0.2])
    y_hat_b = np.array([0.1, 0.9, 0.2, 0.8])
    a = np.array([0, 0, 1, 1])
    tau = 0.5

    dd = decision_disagreement(y_hat_a, y_hat_b, a, tau)

    assert dd["disagree_rate"] == pytest.approx(1.0)
    assert dd["n_disagree"] == 4
    assert dd["r1_grant_r2_deny"] == 2
    assert dd["r1_deny_r2_grant"] == 2
    assert dd["n_disagree_female"] == 2
    assert dd["n_disagree_male"] == 2
