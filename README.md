# causal-perception-implementation

**Open source by Santander AI Lab** — **machine learning** research code for
**causal inference**: comparing competing structural causal models (SCMs) on
the German Credit dataset.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/SantanderAI/causal-perception-implementation/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/causal-perception-implementation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Open source by Santander AI Lab.** This repository is the **machine learning**
research code for *causal perception* — comparing competing **structural causal
models (SCMs)** through their interventional and counterfactual distributions,
applied to fair credit decisions on the German Credit dataset.

It implements a linear Additive Noise Model with a configurable causal DAG
(Chiappa, 2019), Pearl's `do`-operator and counterfactual (abduction →
intervention → prediction) inference, three 1-D distribution distances
(Wasserstein-2, KL via KDE, Total Variation), bootstrap confidence intervals,
and a fair-decisions experiment (demographic-parity / equal-opportunity gaps,
ROC/PR, decision disagreement).

> This is research code accompanying a (forthcoming) academic paper. Copyright
> is held by the author, **José M. Álvarez**; the open-source release is
> distributed and maintained by **Santander AI Lab** with the author's consent
> (see [`NOTICE`](NOTICE)).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Requirements

- Python 3.10, 3.11, or 3.12
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`
- Plotting experiments additionally require `matplotlib` (installed via the
  `viz` extra)
- Internet access **on first run only**, to fetch the German Credit dataset
  from OpenML

## Installation

```bash
# Core engine only
pip install -e .

# With plotting support for the experiment scripts
pip install -e ".[viz]"

# Development (tests, linters, type checker)
pip install -e ".[dev,viz]"
```

## Data

The German Credit (Statlog) dataset is **not redistributed** in this
repository. It is fetched from [OpenML](https://www.openml.org/d/31)
(`credit-g`) and cached under `data/` automatically on first use — both
`src.data_prep.load_data()` and every experiment script call it transparently.

To (re)generate the CSV splits explicitly:

```bash
python -m src.data_prep
```

See [`data/README.md`](data/README.md) for the variable mapping (Chiappa's
DAG), provenance, and citation.

## Usage

Each experiment is runnable as a module from the repository root:

```bash
python -m src.run_fair_decisions        # accuracy + fairness (DP/EO, ROC/PR)
python -m src.run_parametrical          # parametrical perception (Δβ on A→Y)
python -m src.run_causal_dp_sweep       # DP gap vs decision threshold
python -m src.run_structural_age        # alternative disagreement on C→Y (age)
python -m src.run_structural_nonlinear  # nonlinear (GBM) robustness check
python -m src.run_bootstrap_cis_all     # 95% bootstrap CIs for all experiments
python -m src.plot_structural_combined  # combined structural-perception figure
```

Minimal programmatic example:

```python
from src.data_prep import load_data
from src.linear_anm import LinearANM, CHIAPPA_FULL, CHIAPPA_NO_AY
from src.perception import fit_scms, run_perception

train, test = load_data()
scm1, scm2 = fit_scms(train, CHIAPPA_FULL, CHIAPPA_NO_AY)
result = run_perception(scm1, scm2, test, variable="A", values=[0, 1])
print(result["aggregated"])  # {"W2": ..., "KL": ..., "TV": ...}
```

## Repository structure

```
src/
  data_prep.py              # OpenML fetch + Chiappa variable mapping + load_data()
  linear_anm.py             # Linear ANM: DAGs, fit, do-operator, counterfactuals
  distances.py              # W2, KL (KDE), Total Variation between 1-D samples
  perception.py             # competing-SCM engine + bootstrap CIs
  run_*.py                  # experiment entrypoints (require the `viz` extra)
  plot_structural_combined.py
tests/                      # pytest suite (OpenML fetch mocked — no network)
data/                       # generated CSVs (git-ignored) + data/README.md
```

## Citation

If you use this code, please cite the accompanying paper and the dataset (see
[`CITATION.cff`](CITATION.cff) and [`data/README.md`](data/README.md)). The
paper is forthcoming; the citation file will be updated with the venue and DOI
once published.

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). All
contributors must sign the CLA (handled automatically by the CLA Assistant bot)
and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please report vulnerabilities privately as described in
[`.github/SECURITY.md`](.github/SECURITY.md). Do **not** open public issues for
security reports.

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
copyright and third-party data attribution.
