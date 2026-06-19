# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Causal DP gap sweep over thresholds.

For each receiver (R1: full DAG, R2: no A->Y), compute interventional and
counterfactual Y_hat under do(A=0) and do(A=1), then sweep tau from 0 to 1
and plot the DP gap = P(Y_hat >= tau | do(A=0)) - P(Y_hat >= tau | do(A=1)).

Outputs:
  - 1x2 figure: (a) interventional DP gap vs tau, (b) counterfactual DP gap vs tau
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import load_data
from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_AY
from src.perception import fit_scms, run_perception


def dp_gap_sweep(y_hat_do0, y_hat_do1, taus):
    """Compute DP gap at each threshold: P(Y_hat >= tau | do(A=0)) - P(Y_hat >= tau | do(A=1))."""
    gaps = []
    for tau in taus:
        accept_0 = np.mean(y_hat_do0 >= tau)
        accept_1 = np.mean(y_hat_do1 >= tau)
        gaps.append(accept_0 - accept_1)
    return np.array(gaps)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data()
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)

    int_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="interventional")
    cf_r = run_perception(scm1, scm2, test, "A", [0, 1], rung="counterfactual")

    # Extract probability arrays
    int_r1_do0 = int_r["probs"][("scm1", 0)]
    int_r1_do1 = int_r["probs"][("scm1", 1)]
    int_r2_do0 = int_r["probs"][("scm2", 0)]
    int_r2_do1 = int_r["probs"][("scm2", 1)]
    cf_r1_do0 = cf_r["probs"][("scm1", 0)]
    cf_r1_do1 = cf_r["probs"][("scm1", 1)]
    cf_r2_do0 = cf_r["probs"][("scm2", 0)]
    cf_r2_do1 = cf_r["probs"][("scm2", 1)]

    # -- Threshold sweep ------------------------------------------------------
    taus = np.linspace(0.01, 0.99, 200)

    int_dp_r1 = dp_gap_sweep(int_r1_do0, int_r1_do1, taus)
    int_dp_r2 = dp_gap_sweep(int_r2_do0, int_r2_do1, taus)
    cf_dp_r1 = dp_gap_sweep(cf_r1_do0, cf_r1_do1, taus)
    cf_dp_r2 = dp_gap_sweep(cf_r2_do0, cf_r2_do1, taus)

    # Print key values at tau = 0.70
    tau_ref = 0.70
    for label, do0, do1 in [
        ("R1 int", int_r1_do0, int_r1_do1),
        ("R2 int", int_r2_do0, int_r2_do1),
        ("R1 cf", cf_r1_do0, cf_r1_do1),
        ("R2 cf", cf_r2_do0, cf_r2_do1),
    ]:
        gap = np.mean(do0 >= tau_ref) - np.mean(do1 >= tau_ref)
        print(
            f"{label} @ tau={tau_ref}: accept_do0={np.mean(do0>=tau_ref):.3f}, "
            f"accept_do1={np.mean(do1>=tau_ref):.3f}, DP gap={gap:+.3f}"
        )

    # -- Plot -----------------------------------------------------------------
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # (a) Interventional
    ax = axes[0]
    ax.plot(taus, int_dp_r1, color="steelblue", lw=2, label=r"$R_1$ ($\mathcal{M}_1$)")
    ax.plot(taus, int_dp_r2, color="coral", lw=2, label=r"$R_2$ ($\mathcal{M}_2$)")
    ax.axhline(y=0, color="k", ls="-", lw=0.5, alpha=0.5)
    ax.axvline(x=0.70, color="gray", ls="--", lw=1, alpha=0.7, label=r"$\tau = 0.70$")
    ax.set_xlabel(r"Threshold $\tau$")
    ax.set_ylabel("DP gap (do(A=0, female) - do(A=1, male))")
    ax.set_title("(a) Interventional")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])

    # (b) Counterfactual
    ax = axes[1]
    ax.plot(taus, cf_dp_r1, color="steelblue", lw=2, label=r"$R_1$ ($\mathcal{M}_1$)")
    ax.plot(taus, cf_dp_r2, color="coral", lw=2, label=r"$R_2$ ($\mathcal{M}_2$)")
    ax.axhline(y=0, color="k", ls="-", lw=0.5, alpha=0.5)
    ax.axvline(x=0.70, color="gray", ls="--", lw=1, alpha=0.7, label=r"$\tau = 0.70$")
    ax.set_xlabel(r"Threshold $\tau$")
    ax.set_title("(b) Counterfactual")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "causal_dp_sweep.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
