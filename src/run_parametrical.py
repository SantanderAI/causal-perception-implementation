# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Parametrical Perception Experiment.

Compares two SCMs with the SAME DAG (Chiappa full) but different A->Y coefficients:
  M1: baseline (point estimate beta)
  M2: perturbed (beta - 2*SE)

Both interventional and counterfactual distributions are computed.
Outputs: CSV tables, combined 2x2 figure.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import load_data
from src.linear_anm import CHIAPPA_FULL, LinearANM, compute_standard_errors
from src.perception import format_distances, run_perception


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data()
    output_dir = os.path.join(project_root, "output")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Train: {len(train)} rows, Test: {len(test)} rows\n")

    # -- Fit baseline SCM (M1) and create perturbed M2 ------------------------
    scm1 = LinearANM(edges=CHIAPPA_FULL).fit(train)
    parents_y = CHIAPPA_FULL["Y"]
    X_train_y = train[parents_y].values
    se_all = compute_standard_errors(scm1.models["Y"], X_train_y)
    a_idx = parents_y.index("A")
    se_A = se_all[a_idx + 1]
    beta_A = scm1.models["Y"].coef_[0][a_idx]

    print(f"A->Y coefficient: beta = {beta_A:.4f}, SE = {se_A:.4f}")
    print(f"M1: beta = {beta_A:.4f}")
    print(f"M2: beta - 2*SE = {beta_A - 2 * se_A:.4f}\n")

    scm2 = copy.deepcopy(scm1)
    scm2.models["Y"].coef_[0][a_idx] = beta_A - 2 * se_A

    # -- Run perception via shared engine -------------------------------------
    int_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="interventional")
    cf_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="counterfactual")
    format_distances(int_r, label="PARAMETRICAL (interventional)")
    format_distances(cf_r, label="PARAMETRICAL (counterfactual)")

    # -- Save CSV results -----------------------------------------------------
    for rung_label, r in [("interventional", int_r), ("counterfactual", cf_r)]:
        rows = []
        for v, d in r["between"].items():
            rows.append({"comparison": f"parametrical_do{v}", **d})
        rows.append({"comparison": "parametrical_agg", **r["aggregated"]})
        rows.append({"comparison": "within_M1", **r["within_scm1"]})
        rows.append({"comparison": "within_M2", **r["within_scm2"]})
        path = os.path.join(output_dir, f"parametrical_perception_{rung_label}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Results saved to {path}")

    # -- Combined 2x2 figure --------------------------------------------------
    m1_do0, m1_do1 = int_r["probs"][("scm1", 0)], int_r["probs"][("scm1", 1)]
    m2_do0, m2_do1 = int_r["probs"][("scm2", 0)], int_r["probs"][("scm2", 1)]
    m1_cf0, m1_cf1 = cf_r["probs"][("scm1", 0)], cf_r["probs"][("scm1", 1)]
    m2_cf0, m2_cf1 = cf_r["probs"][("scm2", 0)], cf_r["probs"][("scm2", 1)]

    all_probs = np.concatenate(
        [
            m1_do0,
            m1_do1,
            m2_do0,
            m2_do1,
            m1_cf0,
            m1_cf1,
            m2_cf0,
            m2_cf1,
        ]
    )
    bin_lo, bin_hi = all_probs.min() - 0.02, all_probs.max() + 0.02
    bins = np.linspace(bin_lo, bin_hi, 35)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")

    ax = axes[0, 0]
    ax.hist(m1_do0, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_1$", color="steelblue")
    ax.hist(m2_do0, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_2$", color="coral")
    ax.set_ylabel("Density")
    ax.set_title(r"(a) Interventional dist. under do($A$=0)")
    ax.legend(fontsize=10)

    ax = axes[0, 1]
    ax.hist(m1_do1, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_1$", color="steelblue")
    ax.hist(m2_do1, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_2$", color="coral")
    ax.set_title(r"(b) Interventional dist. under do($A$=1)")
    ax.legend(fontsize=10)

    ax = axes[1, 0]
    ax.hist(m1_cf0, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_1$", color="steelblue")
    ax.hist(m2_cf0, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_2$", color="coral")
    ax.set_xlabel(r"$\hat{Y}$  (predicted risk probability)")
    ax.set_ylabel("Density")
    ax.set_title(r"(c) Counterfactual dist. under do($A$=0)")
    ax.legend(fontsize=10)

    ax = axes[1, 1]
    ax.hist(m1_cf1, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_1$", color="steelblue")
    ax.hist(m2_cf1, bins=bins, alpha=0.5, density=True, label=r"$\mathcal{M}_2$", color="coral")
    ax.set_xlabel(r"$\hat{Y}$  (predicted risk probability)")
    ax.set_title(r"(d) Counterfactual dist. under do($A$=1)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "parametrical_perception_combined.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
