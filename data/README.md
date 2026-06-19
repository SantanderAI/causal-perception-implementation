# Data

This directory holds the dataset used by the experiments. **No data files are
redistributed in this repository.** The German Credit (Statlog) dataset is
third-party data and is fetched at runtime from [OpenML](https://www.openml.org/d/31)
by the data-preparation script.

## How to generate the data

From the repository root:

```bash
python -m src.data_prep
```

This downloads the German Credit dataset via `sklearn.datasets.fetch_openml("credit-g", version=1)`
and writes three CSV files into this directory:

| File | Rows | Description |
| --- | --- | --- |
| `german_credit_chiappa.csv` | 1000 | Full dataset mapped to Chiappa's variables |
| `german_credit_train.csv` | 700 | Stratified 70% train split (`random_state=42`) |
| `german_credit_test.csv` | 300 | Stratified 30% test split (`random_state=42`) |

The experiment scripts and `src/data_prep.load_data()` regenerate these files
automatically if they are missing, so you normally do not need to run the
command above by hand.

## Variable mapping (Chiappa, 2019)

The raw OpenML features are mapped to the causal DAG variables as follows:

| Var | Meaning | Source feature | Encoding |
| --- | --- | --- | --- |
| `A` | Sex | `personal_status` | 1 = male, 0 = female |
| `C` | Age | `age` | continuous |
| `S1` | Checking account status | `checking_status` | ordinal 0–3 |
| `S2` | Savings account | `savings_status` | ordinal 0–4 |
| `S3` | Housing | `housing` | ordinal 0–2 |
| `R1` | Credit amount | `credit_amount` | continuous |
| `R2` | Duration (months) | `duration` | continuous |
| `Y` | Credit risk | `class` | 1 = good, 0 = bad |

## Provenance and citation

- **Hofmann, H. (1994). Statlog (German Credit Data).** UCI Machine Learning
  Repository. <https://doi.org/10.24432/C5NC77>. Distributed via OpenML as
  dataset `credit-g` (<https://www.openml.org/d/31>).
- **Chiappa, S. (2019). Path-Specific Counterfactual Fairness.** Proceedings of
  the AAAI Conference on Artificial Intelligence, 33(01). Source of the causal
  DAG over the German Credit variables.

The dataset is provided by its original distributors under their own terms;
please consult OpenML / the UCI repository before any redistribution.
