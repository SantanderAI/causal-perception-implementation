# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/fairness.py` with `fairness_metrics` and `decision_disagreement` helpers
  for demographic-parity / equal-opportunity gaps and receiver disagreement.
- Apache 2.0 `LICENSE` + `NOTICE` with third-party data attribution (German
  Credit / OpenML `credit-g`, Chiappa 2019 DAG).
- `data/README.md` documenting dataset provenance and the variable mapping.
- `src/data_prep.load_data()` / `ensure_data()` helpers that fetch the German
  Credit dataset from OpenML and cache it under `data/` on first use, so the
  CSVs are no longer redistributed in the repository.
- Community/governance files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `.github/SECURITY.md`, `CODEOWNERS`.
- `pyproject.toml` packaging the engine with pinned dependencies, a `viz` extra
  for the plotting scripts, and Black/Ruff/mypy/pytest/coverage configuration.
- `CITATION.cff` for software + the accompanying (forthcoming) paper.
- `tests/` pytest suite covering `distances`, `linear_anm`, `perception`,
  `data_prep`, and `fairness` (with the OpenML fetch mocked — no network access
  required).
- SPDX headers (`Copyright (c) 2026 José M. Álvarez` / `Apache-2.0`) on all
  Python source files.
- GitHub Actions workflows (third-party actions pinned to SHA digests):
  `ci`, `codeql`, `dep-scan`, `license-check`, `pattern-check`, `cla`,
  `stale`, `release`.
- `.github/dependabot.yml` (monthly Python + GitHub Actions updates),
  issue templates (bug, feature, config) and a PR template.
- `.github/pattern-check-allowlist.txt` for the internal-pattern scan.

### Changed
- Experiment scripts and the `linear_anm` sanity check now load data via
  `src.data_prep.load_data()` instead of reading committed CSVs directly.

### Removed
- The three pre-generated German Credit CSVs (`data/*.csv`) — regenerated on
  demand from OpenML.

[Unreleased]: https://github.com/SantanderAI/causal-perception-implementation/commits/main
