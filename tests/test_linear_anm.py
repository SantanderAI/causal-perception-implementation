# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.linear_anm."""

import numpy as np
import pytest

from src.linear_anm import (
    CHIAPPA_FULL,
    CHIAPPA_NO_AY,
    CHIAPPA_NO_CY,
    LinearANM,
    compute_standard_errors,
    topological_sort,
)


def test_topological_sort_valid_order():
    order = topological_sort(CHIAPPA_FULL)
    pos = {node: i for i, node in enumerate(order)}
    assert set(order) == set(CHIAPPA_FULL)
    for node, parents in CHIAPPA_FULL.items():
        for p in parents:
            assert pos[p] < pos[node]


def test_topological_sort_detects_cycle():
    cyclic = {"A": ["B"], "B": ["A"]}
    with pytest.raises(ValueError):
        topological_sort(cyclic)


def test_init_derives_roots_and_mediators():
    scm = LinearANM(edges=CHIAPPA_FULL)
    assert scm.roots == ["A", "C"]
    assert set(scm.mediators) == {"S1", "S2", "S3", "R1", "R2"}
    assert scm.outcome == "Y"


def test_fit_populates_models_and_root_stats(train_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    assert set(scm.root_stats) == {"A", "C"}
    # one model per mediator + outcome
    assert set(scm.models) == {"S1", "S2", "S3", "R1", "R2", "Y"}


def test_intervene_returns_probabilities(train_df, test_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    result, probs = scm.intervene(test_df, {"A": 1})
    assert len(probs) == len(test_df)
    assert np.all((probs >= 0) & (probs <= 1))
    # intervened variable is fixed
    assert np.all(result["A"].values == 1)


def test_intervene_without_intervention_keeps_roots(train_df, test_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    result, _ = scm.intervene(test_df, {})
    assert np.allclose(result["A"].values, test_df["A"].values)
    assert np.allclose(result["C"].values, test_df["C"].values)


def test_counterfactual_shapes_and_range(train_df, test_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    result, probs = scm.counterfactual(test_df, {"A": 0})
    assert len(probs) == len(test_df)
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.all(result["A"].values == 0)


def test_abduct_returns_noise_per_mediator(train_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    noise = scm.abduct(train_df)
    assert set(noise) == {"S1", "S2", "S3", "R1", "R2"}
    for arr in noise.values():
        assert len(arr) == len(train_df)


def test_alternative_dags_change_y_parents(train_df, test_df):
    scm_full = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    scm_no_ay = LinearANM(edges=CHIAPPA_NO_AY).fit(train_df)
    scm_no_cy = LinearANM(edges=CHIAPPA_NO_CY).fit(train_df)
    _, p_full = scm_full.intervene(test_df, {"A": 1})
    _, p_no_ay = scm_no_ay.intervene(test_df, {"A": 1})
    _, p_no_cy = scm_no_cy.intervene(test_df, {"A": 1})
    # All valid probability vectors of the right length.
    for p in (p_full, p_no_ay, p_no_cy):
        assert len(p) == len(test_df)
        assert np.all((p >= 0) & (p <= 1))


def test_custom_y_model_is_used(train_df, test_df):
    from sklearn.ensemble import GradientBoostingClassifier

    gbm = GradientBoostingClassifier(n_estimators=10, random_state=0)
    scm = LinearANM(edges=CHIAPPA_FULL, y_model=gbm).fit(train_df)
    _, probs = scm.intervene(test_df, {"A": 1})
    assert np.all((probs >= 0) & (probs <= 1))


def test_compute_standard_errors_shape_and_sign(train_df):
    scm = LinearANM(edges=CHIAPPA_FULL).fit(train_df)
    model = scm.models["Y"]
    X = train_df[CHIAPPA_FULL["Y"]].values
    se = compute_standard_errors(model, X)
    # intercept + one per parent
    assert len(se) == X.shape[1] + 1
    assert np.all(se > 0)
