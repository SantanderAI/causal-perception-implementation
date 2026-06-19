# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.distances."""

import numpy as np
import pytest

from src.distances import (
    compute_all_distances,
    kl_divergence,
    print_distances,
    total_variation,
    wasserstein2,
)


def test_wasserstein2_identical_is_zero():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert wasserstein2(x, x) == pytest.approx(0.0)


def test_wasserstein2_constant_shift():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = x + 2.0
    assert wasserstein2(x, y) == pytest.approx(2.0)


def test_wasserstein2_is_order_invariant():
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    assert wasserstein2(x, y) == pytest.approx(0.0)


def test_wasserstein2_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        wasserstein2(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_kl_divergence_same_distribution_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=500)
    b = rng.normal(0.5, 0.1, size=500)
    assert abs(kl_divergence(a, b)) < 0.2


def test_kl_divergence_larger_for_shifted():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, size=500)
    near = rng.normal(0.5, 0.1, size=500)
    far = rng.normal(1.5, 0.1, size=500)
    assert kl_divergence(a, far) > kl_divergence(a, near)


def test_total_variation_identical_is_zero():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert total_variation(x, x) == pytest.approx(0.0)


def test_total_variation_disjoint_supports_near_one():
    a = np.zeros(100)
    b = np.ones(100)
    tv = total_variation(a, b)
    assert tv == pytest.approx(1.0, abs=1e-6)


def test_total_variation_in_unit_interval():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=300)
    b = rng.normal(0.7, 1.0, size=300)
    tv = total_variation(a, b)
    assert 0.0 <= tv <= 1.0


def test_compute_all_distances_keys():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.1, size=200)
    b = rng.normal(0.6, 0.1, size=200)
    d = compute_all_distances(a, b)
    assert set(d) == {"W2", "KL", "TV"}
    assert all(isinstance(v, float) for v in d.values())


def test_print_distances_smoke(capsys):
    print_distances("label", {"W2": 0.1, "KL": 0.2, "TV": 0.3})
    out = capsys.readouterr().out
    assert "W2" in out and "label" in out
