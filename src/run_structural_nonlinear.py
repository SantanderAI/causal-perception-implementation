# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Structural Perception - Nonlinear Robustness Check (GBM for Y).

Same two structural DAGs (CHIAPPA_FULL vs CHIAPPA_NO_AY) as in the main text,
but replacing logistic regression for Y with GradientBoostingClassifier.
Mediators remain OLS (linear). This tests whether the qualitative perception
pattern survives when the outcome model is nonlinear.

Outputs distances (W2, KL, TV) for interventional and counterfactual
distributions, matching the main-text format.
"""

from sklearn.ensemble import GradientBoostingClassifier

from src.data_prep import load_data
from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_AY
from src.perception import fit_scms, format_distances, run_perception


def main():
    train, test = load_data()

    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )

    scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY, y_model=gbm)

    int_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="interventional")
    cf_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="counterfactual")
    format_distances(int_r, label="STRUCTURAL GBM (interventional)")
    format_distances(cf_r, label="STRUCTURAL GBM (counterfactual)")

    # -- Comparison with logistic baseline --------------------------------
    scm1_lr, scm2_lr = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)
    lr_r = run_perception(scm1_lr, scm2_lr, test, "A", [0, 1], rung="interventional")

    print(f"\n{'=' * 60}")
    print("COMPARISON: Logistic vs GBM (aggregated interventional)")
    print("=" * 60)
    lr_agg = lr_r["aggregated"]
    gbm_agg = int_r["aggregated"]
    print(f"  Logistic: W2={lr_agg['W2']:.4f}, KL={lr_agg['KL']:.4f}, TV={lr_agg['TV']:.4f}")
    print(f"  GBM:      W2={gbm_agg['W2']:.4f}, KL={gbm_agg['KL']:.4f}, TV={gbm_agg['TV']:.4f}")


if __name__ == "__main__":
    main()
