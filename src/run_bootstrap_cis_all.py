# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Bootstrap Confidence Intervals for All Experiments.

Computes 95% bootstrap CIs (B=1000, percentile method) for W2, KL, and TV
distances across all perception experiments:

  1. Structural perception (A->Y, main text)
  2. Parametrical perception (-2SE, main text)
  3. Parametrical perception (+1SE and +2SE, Appendix B.1)
  4. Alternative structural perception (C->Y, Appendix B.2)
  5. Nonlinear robustness / GBM (Appendix B.3)

Paired resampling: the same bootstrap index is drawn for both distributions
in each comparison, preserving individual-level structure.

Each experiment section uses an independent RNG (seed=42) so that results
are reproducible and order-invariant.

Outputs: results/bootstrap_cis_all.csv
"""

import copy
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.data_prep import load_data
from src.linear_anm import (
    CHIAPPA_FULL,
    CHIAPPA_NO_AY,
    CHIAPPA_NO_CY,
    LinearANM,
    compute_standard_errors,
)
from src.perception import bootstrap_ci, fit_scms


def run_comparisons(comparisons, B, rng, rows):
    """Run bootstrap CI for a list of comparisons and append to rows."""
    for label, exp, level, s1, s2 in comparisons:
        ci = bootstrap_ci(s1, s2, B=B, rng=rng)
        print(f"\n{label} | {level}:")
        for metric in ["W2", "KL", "TV"]:
            pt, lo, hi = ci[metric]
            print(f"  {metric}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")
            rows.append(
                {
                    "experiment": exp,
                    "comparison": label,
                    "level": level,
                    "metric": metric,
                    "point": round(pt, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                }
            )


def add_aggregated(rows, experiment, do0_label, do1_label, agg_label):
    """Compute aggregated (mean) CIs from do0 and do1 rows."""
    do0_rows = [
        r
        for r in rows
        if r["experiment"] == experiment
        and r["comparison"] == "M1 vs M2"
        and r["level"] == do0_label
    ]
    do1_rows = [
        r
        for r in rows
        if r["experiment"] == experiment
        and r["comparison"] == "M1 vs M2"
        and r["level"] == do1_label
    ]
    for d0, d1 in zip(do0_rows, do1_rows):
        rows.append(
            {
                "experiment": experiment,
                "comparison": "M1 vs M2",
                "level": agg_label,
                "metric": d0["metric"],
                "point": round((d0["point"] + d1["point"]) / 2, 4),
                "ci_lo": round((d0["ci_lo"] + d1["ci_lo"]) / 2, 4),
                "ci_hi": round((d0["ci_hi"] + d1["ci_hi"]) / 2, 4),
            }
        )


def run_parametrical_variant(scm1, test, a_idx, new_beta, variant_name, B, rng, rows):
    """Run bootstrap CIs for a parametrical variant (+1SE or +2SE)."""
    scm2 = copy.deepcopy(scm1)
    scm2.models["Y"].coef_[0][a_idx] = new_beta

    print(f"\n  {variant_name}: M2 beta = {new_beta:.4f}")

    _, m1_do0 = scm1.intervene(test, {"A": 0})
    _, m1_do1 = scm1.intervene(test, {"A": 1})
    _, m2_do0 = scm2.intervene(test, {"A": 0})
    _, m2_do1 = scm2.intervene(test, {"A": 1})

    _, m1_cf0 = scm1.counterfactual(test, {"A": 0})
    _, m1_cf1 = scm1.counterfactual(test, {"A": 1})
    _, m2_cf0 = scm2.counterfactual(test, {"A": 0})
    _, m2_cf1 = scm2.counterfactual(test, {"A": 1})

    comparisons = [
        ("M1 vs M2", variant_name, "interventional_do0", m1_do0, m2_do0),
        ("M1 vs M2", variant_name, "interventional_do1", m1_do1, m2_do1),
        ("Within M1", variant_name, "interventional_within", m1_do0, m1_do1),
        ("Within M2", variant_name, "interventional_within", m2_do0, m2_do1),
        ("M1 vs M2", variant_name, "counterfactual_do0", m1_cf0, m2_cf0),
        ("M1 vs M2", variant_name, "counterfactual_do1", m1_cf1, m2_cf1),
        ("Within M1", variant_name, "counterfactual_within", m1_cf0, m1_cf1),
        ("Within M2", variant_name, "counterfactual_within", m2_cf0, m2_cf1),
    ]

    run_comparisons(comparisons, B, rng, rows)

    add_aggregated(
        rows, variant_name, "interventional_do0", "interventional_do1", "interventional_agg"
    )
    add_aggregated(
        rows, variant_name, "counterfactual_do0", "counterfactual_do1", "counterfactual_agg"
    )


# -- Main --------------------------------------------------------------------


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data()
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    B = 1000
    rows = []

    # 1. STRUCTURAL PERCEPTION (A->Y, main text)
    print("=" * 70)
    print("1. STRUCTURAL PERCEPTION (A->Y) - Bootstrap CIs (B={})".format(B))
    print("=" * 70)

    rng = np.random.default_rng(42)

    scm1_s, scm2_s = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)

    _, m1_do0 = scm1_s.intervene(test, {"A": 0})
    _, m1_do1 = scm1_s.intervene(test, {"A": 1})
    _, m2_do0 = scm2_s.intervene(test, {"A": 0})
    _, m2_do1 = scm2_s.intervene(test, {"A": 1})

    _, m1_cf0 = scm1_s.counterfactual(test, {"A": 0})
    _, m1_cf1 = scm1_s.counterfactual(test, {"A": 1})
    _, m2_cf0 = scm2_s.counterfactual(test, {"A": 0})
    _, m2_cf1 = scm2_s.counterfactual(test, {"A": 1})

    structural_comparisons = [
        ("M1 vs M2", "structural", "interventional_do0", m1_do0, m2_do0),
        ("M1 vs M2", "structural", "interventional_do1", m1_do1, m2_do1),
        ("Within M1", "structural", "interventional_within", m1_do0, m1_do1),
        ("Within M2", "structural", "interventional_within", m2_do0, m2_do1),
        ("M1 vs M2", "structural", "counterfactual_do0", m1_cf0, m2_cf0),
        ("M1 vs M2", "structural", "counterfactual_do1", m1_cf1, m2_cf1),
        ("Within M1", "structural", "counterfactual_within", m1_cf0, m1_cf1),
        ("Within M2", "structural", "counterfactual_within", m2_cf0, m2_cf1),
    ]

    run_comparisons(structural_comparisons, B, rng, rows)

    add_aggregated(
        rows, "structural", "interventional_do0", "interventional_do1", "interventional_agg"
    )
    add_aggregated(
        rows, "structural", "counterfactual_do0", "counterfactual_do1", "counterfactual_agg"
    )

    # 2. PARAMETRICAL PERCEPTION (-2SE, main text)
    print("\n" + "=" * 70)
    print("2. PARAMETRICAL PERCEPTION (-2SE) - Bootstrap CIs (B={})".format(B))
    print("=" * 70)

    rng = np.random.default_rng(42)

    scm1_p = LinearANM(edges=CHIAPPA_FULL).fit(train)
    parents_y = CHIAPPA_FULL["Y"]
    X_train_y = train[parents_y].values
    se_all = compute_standard_errors(scm1_p.models["Y"], X_train_y)
    a_idx = parents_y.index("A")
    se_A = se_all[a_idx + 1]
    beta_A = scm1_p.models["Y"].coef_[0][a_idx]

    scm2_p = copy.deepcopy(scm1_p)
    scm2_p.models["Y"].coef_[0][a_idx] = beta_A - 2 * se_A

    print(f"A->Y: beta={beta_A:.4f}, SE={se_A:.4f}, M2 beta={beta_A - 2*se_A:.4f}")

    _, pm1_do0 = scm1_p.intervene(test, {"A": 0})
    _, pm1_do1 = scm1_p.intervene(test, {"A": 1})
    _, pm2_do0 = scm2_p.intervene(test, {"A": 0})
    _, pm2_do1 = scm2_p.intervene(test, {"A": 1})

    _, pm1_cf0 = scm1_p.counterfactual(test, {"A": 0})
    _, pm1_cf1 = scm1_p.counterfactual(test, {"A": 1})
    _, pm2_cf0 = scm2_p.counterfactual(test, {"A": 0})
    _, pm2_cf1 = scm2_p.counterfactual(test, {"A": 1})

    parametrical_comparisons = [
        ("M1 vs M2", "parametrical", "interventional_do0", pm1_do0, pm2_do0),
        ("M1 vs M2", "parametrical", "interventional_do1", pm1_do1, pm2_do1),
        ("Within M1", "parametrical", "interventional_within", pm1_do0, pm1_do1),
        ("Within M2", "parametrical", "interventional_within", pm2_do0, pm2_do1),
        ("M1 vs M2", "parametrical", "counterfactual_do0", pm1_cf0, pm2_cf0),
        ("M1 vs M2", "parametrical", "counterfactual_do1", pm1_cf1, pm2_cf1),
        ("Within M1", "parametrical", "counterfactual_within", pm1_cf0, pm1_cf1),
        ("Within M2", "parametrical", "counterfactual_within", pm2_cf0, pm2_cf1),
    ]

    run_comparisons(parametrical_comparisons, B, rng, rows)

    add_aggregated(
        rows, "parametrical", "interventional_do0", "interventional_do1", "interventional_agg"
    )
    add_aggregated(
        rows, "parametrical", "counterfactual_do0", "counterfactual_do1", "counterfactual_agg"
    )

    # 3. PARAMETRICAL PERCEPTION (+1SE and +2SE, Appendix B.1)
    print("\n" + "=" * 70)
    print("3. PARAMETRICAL +1SE / +2SE - Bootstrap CIs (B={})".format(B))
    print("=" * 70)

    rng = np.random.default_rng(42)

    scm1_pvar = LinearANM(edges=CHIAPPA_FULL).fit(train)

    run_parametrical_variant(
        scm1_pvar, test, a_idx, beta_A + 1 * se_A, "param_plus1SE", B, rng, rows
    )

    scm1_pvar = LinearANM(edges=CHIAPPA_FULL).fit(train)

    run_parametrical_variant(
        scm1_pvar, test, a_idx, beta_A + 2 * se_A, "param_plus2SE", B, rng, rows
    )

    # 4. ALTERNATIVE STRUCTURAL (C->Y, Appendix B.2)
    print("\n" + "=" * 70)
    print("4. STRUCTURAL C->Y - Bootstrap CIs (B={})".format(B))
    print("=" * 70)

    rng = np.random.default_rng(42)

    scm1_a, scm2_a = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_CY)

    c_young, c_old = 26, 43

    _, a1_do_young = scm1_a.intervene(test, {"C": c_young})
    _, a1_do_old = scm1_a.intervene(test, {"C": c_old})
    _, a2_do_young = scm2_a.intervene(test, {"C": c_young})
    _, a2_do_old = scm2_a.intervene(test, {"C": c_old})

    _, a1_cf_young = scm1_a.counterfactual(test, {"C": c_young})
    _, a1_cf_old = scm1_a.counterfactual(test, {"C": c_old})
    _, a2_cf_young = scm2_a.counterfactual(test, {"C": c_young})
    _, a2_cf_old = scm2_a.counterfactual(test, {"C": c_old})

    age_comparisons = [
        ("M1 vs M2", "age", "interventional_young", a1_do_young, a2_do_young),
        ("M1 vs M2", "age", "interventional_old", a1_do_old, a2_do_old),
        ("Within M1", "age", "interventional_within", a1_do_young, a1_do_old),
        ("Within M2", "age", "interventional_within", a2_do_young, a2_do_old),
        ("M1 vs M2", "age", "counterfactual_young", a1_cf_young, a2_cf_young),
        ("M1 vs M2", "age", "counterfactual_old", a1_cf_old, a2_cf_old),
        ("Within M1", "age", "counterfactual_within", a1_cf_young, a1_cf_old),
        ("Within M2", "age", "counterfactual_within", a2_cf_young, a2_cf_old),
    ]

    run_comparisons(age_comparisons, B, rng, rows)

    add_aggregated(rows, "age", "interventional_young", "interventional_old", "interventional_agg")
    add_aggregated(rows, "age", "counterfactual_young", "counterfactual_old", "counterfactual_agg")

    # 5. NONLINEAR ROBUSTNESS / GBM (Appendix B.3)
    print("\n" + "=" * 70)
    print("5. NONLINEAR GBM - Bootstrap CIs (B={})".format(B))
    print("=" * 70)

    rng = np.random.default_rng(42)

    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    scm1_b, scm2_b = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY, y_model=gbm)

    _, b1_do0 = scm1_b.intervene(test, {"A": 0})
    _, b1_do1 = scm1_b.intervene(test, {"A": 1})
    _, b2_do0 = scm2_b.intervene(test, {"A": 0})
    _, b2_do1 = scm2_b.intervene(test, {"A": 1})

    _, b1_cf0 = scm1_b.counterfactual(test, {"A": 0})
    _, b1_cf1 = scm1_b.counterfactual(test, {"A": 1})
    _, b2_cf0 = scm2_b.counterfactual(test, {"A": 0})
    _, b2_cf1 = scm2_b.counterfactual(test, {"A": 1})

    gbm_comparisons = [
        ("M1 vs M2", "gbm", "interventional_do0", b1_do0, b2_do0),
        ("M1 vs M2", "gbm", "interventional_do1", b1_do1, b2_do1),
        ("Within M1", "gbm", "interventional_within", b1_do0, b1_do1),
        ("Within M2", "gbm", "interventional_within", b2_do0, b2_do1),
        ("M1 vs M2", "gbm", "counterfactual_do0", b1_cf0, b2_cf0),
        ("M1 vs M2", "gbm", "counterfactual_do1", b1_cf1, b2_cf1),
        ("Within M1", "gbm", "counterfactual_within", b1_cf0, b1_cf1),
        ("Within M2", "gbm", "counterfactual_within", b2_cf0, b2_cf1),
    ]

    run_comparisons(gbm_comparisons, B, rng, rows)

    add_aggregated(rows, "gbm", "interventional_do0", "interventional_do1", "interventional_agg")
    add_aggregated(rows, "gbm", "counterfactual_do0", "counterfactual_do1", "counterfactual_agg")

    # -- Save ----------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "bootstrap_cis_all.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
