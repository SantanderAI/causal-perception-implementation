# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Download and preprocess the German Credit dataset based on Chiappa (2019)'s DAG:
  A  = Sex (binary: 1=male, 0=female)
  C  = Age (continuous)
  S1 = Checking account status (ordinal)
  S2 = Savings account (ordinal)
  S3 = Housing (ordinal)
  R1 = Credit amount (continuous)
  R2 = Duration in months (continuous)
  Y  = Credit risk (binary: 1=good, 0=bad)

Outputs:
  data/german_credit_chiappa.csv
  data/german_credit_train.csv
  data/german_credit_test.csv
"""

import os

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# -- Paths --------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_FULL = os.path.join(DATA_DIR, "german_credit_chiappa.csv")
OUT_TRAIN = os.path.join(DATA_DIR, "german_credit_train.csv")
OUT_TEST = os.path.join(DATA_DIR, "german_credit_test.csv")

# -- Ordinal encoding maps (OpenML category labels) --------------------------
# Checking account status: no account < low < moderate < high
CHECKING_MAP = {"no checking": 0, "<0": 1, "0<=X<200": 2, ">=200": 3}

# Savings account: no savings < low < moderate < quite rich < rich
SAVINGS_MAP = {"no known savings": 0, "<100": 1, "100<=X<500": 2, "500<=X<1000": 3, ">=1000": 4}

# Housing: for free < rent < own
HOUSING_MAP = {"for free": 0, "rent": 1, "own": 2}

# Sex extraction from personal_status field
SEX_MAP = {
    "male div/sep": 1,
    "female div/dep/mar": 0,
    "male single": 1,
    "male mar/wid": 1,
}


def preprocess():
    """Fetch German Credit via OpenML and map to Chiappa's variables."""
    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = bunch.data.copy()
    df["target"] = bunch.target  # 'good' or 'bad'
    processed = pd.DataFrame()
    # A (Sex): extract from personal_status field, binary (0=female, 1=male)
    processed["A"] = df["personal_status"].map(SEX_MAP)
    # C (Age): continuous
    processed["C"] = df["age"].astype(float)
    # S1 (Checking account status): ordinal
    processed["S1"] = df["checking_status"].map(CHECKING_MAP)
    # S2 (Savings): ordinal
    processed["S2"] = df["savings_status"].map(SAVINGS_MAP)
    # S3 (Housing): ordinal
    processed["S3"] = df["housing"].map(HOUSING_MAP)
    # R1 (Credit amount): continuous
    processed["R1"] = df["credit_amount"].astype(float)
    # R2 (Duration): continuous, in months
    processed["R2"] = df["duration"].astype(float)
    # Y (Credit risk): 1=good, 0=bad
    processed["Y"] = (df["target"] == "good").astype(int)
    # Drop any rows with missing values
    processed = processed.dropna().reset_index(drop=True)
    return processed


def split_and_save(df):
    """Stratified 70/30 split and save to CSV."""
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Y"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    df.to_csv(OUT_FULL, index=False)
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print(f"Full dataset:  {len(df)} rows  -> {OUT_FULL}")
    print(f"Training set:  {len(train)} rows -> {OUT_TRAIN}")
    print(f"Test set:      {len(test)} rows  -> {OUT_TEST}")
    print("\nClass balance (Y):")
    print(f"  Train: {train['Y'].mean():.3f} good")
    print(f"  Test:  {test['Y'].mean():.3f} good")

    return train, test


def ensure_data():
    """Generate the CSV splits from OpenML if they are not already present.

    Returns the three output paths (full, train, test). Idempotent: if all
    three files already exist they are left untouched.
    """
    if all(os.path.exists(p) for p in (OUT_FULL, OUT_TRAIN, OUT_TEST)):
        return OUT_FULL, OUT_TRAIN, OUT_TEST
    os.makedirs(DATA_DIR, exist_ok=True)
    df = preprocess()
    split_and_save(df)
    return OUT_FULL, OUT_TRAIN, OUT_TEST


def load_data():
    """Return the (train, test) DataFrames, generating them on first use.

    The German Credit dataset is not redistributed with this repository; it is
    fetched from OpenML and cached as CSVs under ``data/`` the first time this
    function (or ``python -m src.data_prep``) runs.
    """
    ensure_data()
    train = pd.read_csv(OUT_TRAIN)
    test = pd.read_csv(OUT_TEST)
    return train, test


if __name__ == "__main__":
    df = preprocess()
    print(f"\nProcessed dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescriptive statistics:")
    print(df.describe())
    train, test = split_and_save(df)
