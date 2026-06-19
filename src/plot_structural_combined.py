# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Generate a combined 2x3 figure for structural perception.
Top row: interventional distributions.
Bottom row: counterfactual distributions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import load_data
from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_AY
from src.perception import fit_scms, format_distances, run_perception


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data()
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)
    int_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="interventional")
    cf_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="counterfactual")
    format_distances(int_r, label="STRUCTURAL (interventional)")
    format_distances(cf_r, label="STRUCTURAL (counterfactual)")

    # Extract probability arrays for plotting
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

    # -- Plot 2x2 figure -----------------------------------------------------
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")

    # -- Top row: Interventional ----------------------------------------------
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

    # -- Bottom row: Counterfactual -------------------------------------------
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
    fig_path = os.path.join(fig_dir, "structural_perception_combined.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Figure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
