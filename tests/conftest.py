# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: synthetic German-Credit-shaped data and a fake OpenML bunch.

Tests never touch the network: the OpenML fetch is mocked everywhere.
"""

import numpy as np
import pandas as pd
import pytest

COLUMNS = ["A", "C", "S1", "S2", "S3", "R1", "R2", "Y"]


def _make_frame(n, seed):
    """Build a synthetic frame with the Chiappa variable schema.

    Y depends on the covariates (plus noise) so logistic regression has signal
    and both classes are present.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n).astype(float)
    C = rng.normal(40, 11, size=n)
    S1 = rng.integers(0, 4, size=n).astype(float)
    S2 = rng.integers(0, 5, size=n).astype(float)
    S3 = rng.integers(0, 3, size=n).astype(float)
    R1 = rng.normal(3000, 1500, size=n)
    R2 = rng.normal(20, 10, size=n)
    logit = 0.5 * A - 0.03 * (C - 40) + 0.4 * S1 + 0.2 * S2 - 0.0003 * R1
    p = 1.0 / (1.0 + np.exp(-logit))
    Y = (rng.random(n) < p).astype(int)
    # Guarantee both classes for a stable LogisticRegression fit.
    Y[0], Y[1] = 0, 1
    return pd.DataFrame({"A": A, "C": C, "S1": S1, "S2": S2, "S3": S3, "R1": R1, "R2": R2, "Y": Y})


@pytest.fixture
def train_df():
    return _make_frame(200, seed=1)


@pytest.fixture
def test_df():
    return _make_frame(80, seed=2)


@pytest.fixture
def fake_openml_bunch():
    """A minimal object mimicking sklearn's fetch_openml(as_frame=True) result.

    Contains the raw OpenML 'credit-g' columns used by data_prep.preprocess,
    with category labels that exactly match the encoding maps.
    """
    n = 40
    rng = np.random.default_rng(7)

    class _Bunch:
        pass

    data = pd.DataFrame(
        {
            "personal_status": rng.choice(
                ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"],
                size=n,
            ),
            "age": rng.integers(19, 70, size=n).astype(float),
            "checking_status": rng.choice(["no checking", "<0", "0<=X<200", ">=200"], size=n),
            "savings_status": rng.choice(
                ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"],
                size=n,
            ),
            "housing": rng.choice(["for free", "rent", "own"], size=n),
            "credit_amount": rng.normal(3000, 1500, size=n),
            "duration": rng.normal(20, 10, size=n),
        }
    )
    target = pd.Series(rng.choice(["good", "bad"], size=n))

    bunch = _Bunch()
    bunch.data = data
    bunch.target = target
    return bunch
