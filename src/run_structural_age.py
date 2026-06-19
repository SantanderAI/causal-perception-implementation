# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Structural Perception - Alternative Disagreement on C->Y (Age).

Receivers disagree on whether age (C) directly affects credit risk (Y).
  M1: CHIAPPA_FULL   (includes C->Y)
  M2: CHIAPPA_NO_CY  (removes C->Y; age affects Y only through mediators)

Interventions: do(C=26) ("young", P25) and do(C=43) ("old", P75).

Outputs distances (W2, KL, TV) for interventional and counterfactual
distributions, mirroring the main-text structural perception experiment.
"""

from src.data_prep import load_data
from src.linear_anm import CHIAPPA_FULL, CHIAPPA_NO_CY
from src.perception import fit_scms, format_distances, run_perception


def main():
    train, test = load_data()

    scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_CY)

    int_r = run_perception(scm1, scm2, test, "C", [26, 43], rung="interventional")
    cf_r = run_perception(scm1, scm2, test, "C", [26, 43], rung="counterfactual")
    format_distances(int_r, label="STRUCTURAL C->Y (interventional)")
    format_distances(cf_r, label="STRUCTURAL C->Y (counterfactual)")


if __name__ == "__main__":
    main()
