# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.perception."""

import numpy as np

from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_AY, LinearANM
from src.perception import (
    bootstrap_ci,
    fit_scms,
    format_distances,
    perception_flag,
    run_perception,
    run_perception_bootstrap,
)


def test_fit_scms_returns_two_models(train_df):
    scm1, scm2 = fit_scms(train_df, CHIAPPA_FULL, CHIAPPA_NO_AY)
    assert isinstance(scm1, LinearANM)
    assert isinstance(scm2, LinearANM)


def test_run_perception_structure(train_df, test_df):
    scm1, scm2 = fit_scms(train_df, CHIAPPA_FULL, CHIAPPA_NO_AY)
    res = run_perception(scm1, scm2, test_df, "A", [0, 1])
    assert set(res) == {"between", "aggregated", "within_scm1", "within_scm2", "probs"}
    assert set(res["aggregated"]) == {"W2", "KL", "TV"}
    assert set(res["between"]) == {0, 1}
    assert ("scm1", 0) in res["probs"]


def test_run_perception_counterfactual_rung(train_df, test_df):
    scm1, scm2 = fit_scms(train_df, CHIAPPA_FULL, CHIAPPA_NO_AY)
    res = run_perception(scm1, scm2, test_df, "A", [0, 1], rung="counterfactual")
    assert set(res["aggregated"]) == {"W2", "KL", "TV"}


def test_perception_flag_threshold():
    flags = perception_flag({"W2": 0.2, "KL": 0.05, "TV": 0.5}, epsilon=0.1)
    assert flags == {"W2": 1, "KL": 0, "TV": 1}


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=150)
    b = rng.normal(0.6, 0.1, size=150)
    ci = bootstrap_ci(a, b, B=200, rng=np.random.default_rng(1))
    for metric in ("W2", "KL", "TV"):
        point, lo, hi = ci[metric]
        assert lo <= hi
        assert (
            lo <= point <= hi
            or np.isclose(point, lo, atol=0.05)
            or np.isclose(point, hi, atol=0.05)
        )


def test_run_perception_bootstrap_structure(train_df, test_df):
    scm1, scm2 = fit_scms(train_df, CHIAPPA_FULL, CHIAPPA_NO_AY)
    res = run_perception_bootstrap(scm1, scm2, test_df, "A", [0, 1], B=50)
    assert set(res) == {"between", "within_scm1", "within_scm2"}
    assert set(res["within_scm1"]) == {"W2", "KL", "TV"}
    # each entry is a (point, lo, hi) triple
    assert len(res["within_scm1"]["W2"]) == 3


def test_format_distances_smoke(train_df, test_df, capsys):
    scm1, scm2 = fit_scms(train_df, CHIAPPA_FULL, CHIAPPA_NO_AY)
    res = run_perception(scm1, scm2, test_df, "A", [0, 1])
    format_distances(res, label="TEST")
    out = capsys.readouterr().out
    assert "PERCEPTION TEST" in out
    assert "Aggregated" in out
