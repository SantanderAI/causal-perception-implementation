# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Fair Decisions Experiment - Accuracy + Fairness.

For the structural pair (R1: full DAG, R2: no A->Y), compute factual
predicted risk Y_hat on the 300 test individuals and compare against
ground-truth Y labels.

Outputs:
  - ROC curves (both receivers, one plot) with AUC
  - Precision-Recall curves (both receivers, one plot) with AUC
  - Fairness metrics: DP gap, EO gaps, decision disagreement
  - CSV with per-receiver metrics
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src.data_prep import load_data
from src.fairness import decision_disagreement, fairness_metrics
from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_AY
from src.perception import fit_scms


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data()
    output_dir = os.path.join(project_root, "output")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    y_true = test["Y"].values
    a = test["A"].values
    tau = 0.70  # base rate of the test set

    print(f"Train: {len(train)} rows, Test: {len(test)} rows")
    print(f"Test: {(a==0).sum()} female, {(a==1).sum()} male")
    print(f"Test base rate (good credit): {y_true.mean():.3f}")
    print(f"Decision threshold: tau = {tau}\n")

    # -- Fit structural SCMs --------------------------------------------------
    scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)

    # Factual predictions (no intervention - propagate observed values)
    _, y_hat_r1 = scm1.intervene(test, {})
    _, y_hat_r2 = scm2.intervene(test, {})

    print(f"R1 (full DAG)  - mean Y_hat: {y_hat_r1.mean():.4f} +/- {y_hat_r1.std():.4f}")
    print(f"R2 (no A->Y)   - mean Y_hat: {y_hat_r2.mean():.4f} +/- {y_hat_r2.std():.4f}\n")

    # ROC CURVES
    fpr1, tpr1, _ = roc_curve(y_true, y_hat_r1)
    fpr2, tpr2, _ = roc_curve(y_true, y_hat_r2)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)

    print("=" * 60)
    print("ROC AUC")
    print("=" * 60)
    print(f"  R1 (full DAG): {roc_auc1:.3f}")
    print(f"  R2 (no A->Y):  {roc_auc2:.3f}")
    print(f"  Delta AUC:         {abs(roc_auc1 - roc_auc2):.3f}")

    # PRECISION-RECALL CURVES
    prec1, rec1, _ = precision_recall_curve(y_true, y_hat_r1)
    prec2, rec2, _ = precision_recall_curve(y_true, y_hat_r2)
    pr_auc1 = average_precision_score(y_true, y_hat_r1)
    pr_auc2 = average_precision_score(y_true, y_hat_r2)

    print(f"\n{'=' * 60}")
    print("PR AUC (Average Precision)")
    print("=" * 60)
    print(f"  R1 (full DAG): {pr_auc1:.3f}")
    print(f"  R2 (no A->Y):  {pr_auc2:.3f}")
    print(f"  Delta AUC:         {abs(pr_auc1 - pr_auc2):.3f}")

    # FAIRNESS METRICS (factual, tau=0.70)
    m_r1 = fairness_metrics(y_true, y_hat_r1, a, tau)
    m_r2 = fairness_metrics(y_true, y_hat_r2, a, tau)

    print(f"\n{'=' * 60}")
    print(f"FAIRNESS METRICS (factual, tau = {tau})")
    print("=" * 60)

    for label, m in [("R1 (full DAG)", m_r1), ("R2 (no A->Y)", m_r2)]:
        print(f"\n{label}:")
        print(
            f"  Accept rate:  {m['accept_all']:.3f} (F: {m['accept_female']:.3f}, M: {m['accept_male']:.3f})"
        )
        print(f"  DP gap (F-M): {m['dp_gap']:+.3f}")
        print(
            f"  TPR:          F: {m['tpr_female']:.3f}, M: {m['tpr_male']:.3f} -> gap: {m['tpr_gap']:+.3f}"
        )
        print(
            f"  FPR:          F: {m['fpr_female']:.3f}, M: {m['fpr_male']:.3f} -> gap: {m['fpr_gap']:+.3f}"
        )

    # Decision disagreement
    dd = decision_disagreement(y_hat_r1, y_hat_r2, a, tau)
    disagree_rate = dd["disagree_rate"]
    disagree_f = dd["disagree_female"]
    disagree_m = dd["disagree_male"]
    n_disagree = dd["n_disagree"]

    print("\nDecision disagreement (R1 vs R2):")
    print(f"  Overall:  {disagree_rate:.3f} ({n_disagree} / {len(test)} applicants)")
    print(f"  Female:   {disagree_f:.3f} ({dd['n_disagree_female']} / {(a==0).sum()})")
    print(f"  Male:     {disagree_m:.3f} ({dd['n_disagree_male']} / {(a==1).sum()})")
    print(f"  R1 grants, R2 denies: {dd['r1_grant_r2_deny']}")
    print(f"  R1 denies, R2 grants: {dd['r1_deny_r2_grant']}")

    # PLOT: 1x2 figure (ROC + PR)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    ax.plot(
        fpr1,
        tpr1,
        color="steelblue",
        lw=2,
        label=rf"$R_1$ ($\mathcal{{M}}_1$): AUC = {roc_auc1:.3f}",
    )
    ax.plot(
        fpr2, tpr2, color="coral", lw=2, label=rf"$R_2$ ($\mathcal{{M}}_2$): AUC = {roc_auc2:.3f}"
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC Curve")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # Precision-Recall
    ax = axes[1]
    ax.plot(
        rec1,
        prec1,
        color="steelblue",
        lw=2,
        label=rf"$R_1$ ($\mathcal{{M}}_1$): AP = {pr_auc1:.3f}",
    )
    ax.plot(
        rec2, prec2, color="coral", lw=2, label=rf"$R_2$ ($\mathcal{{M}}_2$): AP = {pr_auc2:.3f}"
    )
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", ls="--", lw=1, alpha=0.5, label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "fair_decisions_accuracy.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {fig_path}")
    plt.close()

    # -- Save CSV -------------------------------------------------------------
    rows = []
    for label, m in [("R1_structural", m_r1), ("R2_structural", m_r2)]:
        row = {
            "receiver": label,
            "ROC_AUC": roc_auc1 if "R1" in label else roc_auc2,
            "PR_AUC": pr_auc1 if "R1" in label else pr_auc2,
        }
        row.update(m)
        rows.append(row)
    rows.append(
        {
            "receiver": "disagreement",
            "accept_all": disagree_rate,
            "accept_female": disagree_f,
            "accept_male": disagree_m,
        }
    )
    results = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "fair_decisions_structural.csv")
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
