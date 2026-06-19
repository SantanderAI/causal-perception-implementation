# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.data_prep (OpenML fetch is mocked — no network access)."""

import os

import pandas as pd

from src import data_prep

CHIAPPA_COLUMNS = ["A", "C", "S1", "S2", "S3", "R1", "R2", "Y"]


def test_preprocess_maps_to_chiappa_schema(monkeypatch, fake_openml_bunch):
    monkeypatch.setattr(data_prep, "fetch_openml", lambda *a, **k: fake_openml_bunch)
    df = data_prep.preprocess()
    assert list(df.columns) == CHIAPPA_COLUMNS
    # A is binary {0, 1}
    assert set(df["A"].unique()).issubset({0, 1})
    # Y is binary {0, 1}
    assert set(df["Y"].unique()).issubset({0, 1})
    # No missing values (dropna applied)
    assert not df.isna().any().any()
    # Ordinal encodings within expected ranges
    assert df["S1"].between(0, 3).all()
    assert df["S2"].between(0, 4).all()
    assert df["S3"].between(0, 2).all()


def test_split_and_save_writes_three_files(monkeypatch, tmp_path, fake_openml_bunch):
    monkeypatch.setattr(data_prep, "fetch_openml", lambda *a, **k: fake_openml_bunch)
    out_full = tmp_path / "full.csv"
    out_train = tmp_path / "train.csv"
    out_test = tmp_path / "test.csv"
    monkeypatch.setattr(data_prep, "OUT_FULL", str(out_full))
    monkeypatch.setattr(data_prep, "OUT_TRAIN", str(out_train))
    monkeypatch.setattr(data_prep, "OUT_TEST", str(out_test))

    df = data_prep.preprocess()
    train, test = data_prep.split_and_save(df)

    assert out_full.exists() and out_train.exists() and out_test.exists()
    assert len(train) + len(test) == len(df)
    # 70/30 stratified split
    assert len(test) == round(0.3 * len(df))


def test_ensure_data_generates_when_missing(monkeypatch, tmp_path, fake_openml_bunch):
    monkeypatch.setattr(data_prep, "fetch_openml", lambda *a, **k: fake_openml_bunch)
    monkeypatch.setattr(data_prep, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_prep, "OUT_FULL", str(tmp_path / "full.csv"))
    monkeypatch.setattr(data_prep, "OUT_TRAIN", str(tmp_path / "train.csv"))
    monkeypatch.setattr(data_prep, "OUT_TEST", str(tmp_path / "test.csv"))

    full, train, test = data_prep.ensure_data()
    assert os.path.exists(full) and os.path.exists(train) and os.path.exists(test)


def test_ensure_data_is_idempotent(monkeypatch, tmp_path, fake_openml_bunch):
    calls = {"n": 0}

    def counting_fetch(*a, **k):
        calls["n"] += 1
        return fake_openml_bunch

    monkeypatch.setattr(data_prep, "fetch_openml", counting_fetch)
    monkeypatch.setattr(data_prep, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_prep, "OUT_FULL", str(tmp_path / "full.csv"))
    monkeypatch.setattr(data_prep, "OUT_TRAIN", str(tmp_path / "train.csv"))
    monkeypatch.setattr(data_prep, "OUT_TEST", str(tmp_path / "test.csv"))

    data_prep.ensure_data()
    data_prep.ensure_data()  # second call must not re-fetch
    assert calls["n"] == 1


def test_load_data_returns_train_test(monkeypatch, tmp_path, fake_openml_bunch):
    monkeypatch.setattr(data_prep, "fetch_openml", lambda *a, **k: fake_openml_bunch)
    monkeypatch.setattr(data_prep, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_prep, "OUT_FULL", str(tmp_path / "full.csv"))
    monkeypatch.setattr(data_prep, "OUT_TRAIN", str(tmp_path / "train.csv"))
    monkeypatch.setattr(data_prep, "OUT_TEST", str(tmp_path / "test.csv"))

    train, test = data_prep.load_data()
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    assert list(train.columns) == CHIAPPA_COLUMNS
    assert len(train) > len(test)
