# Contributing to causal-perception-implementation

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to **opensource@gruposantander.com**.

## How to Contribute

This is research code accompanying an academic paper on causal perception.
Contributions that improve reproducibility, portability, tests, and
documentation are especially welcome.

### Reporting Bugs

1. **Check existing issues** — Search the [issue tracker](../../issues) to see if the bug has already been reported.
2. **Open a new issue** — If not, open a new issue using the **Bug Report** template. Include:
   - A clear and descriptive title
   - Steps to reproduce the behavior
   - Expected behavior vs actual behavior
   - Environment details (OS, Python version, package versions)
   - Any relevant logs or screenshots

### Proposing Features

1. **Open an issue** using the **Feature Request** template.
2. Describe the problem you are trying to solve.
3. Describe your proposed solution.
4. Discuss the feature with maintainers before implementing.

## Pull Request Process

### For External Contributors

1. **Fork** the repository to your GitHub account.
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b fix/cross-platform-paths
   ```
3. **Make your changes** following the [Code Style](#code-style) guidelines.
4. **Add tests** for any new functionality.
5. **Update documentation** if your changes affect the public API.
6. **Commit** with clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add Mahalanobis distance to distances
   fix: guard KDE against zero-variance samples
   docs: document the counterfactual flow
   ```
7. **Push** your branch and open a Pull Request against `main`.
8. **Sign the CLA** when prompted by the CLA Assistant bot.
9. **Wait for review** — A maintainer will review your PR within 2 weeks (SLA).

### For Internal Contributors (Santander)

1. **Create a branch** from `main` (no fork needed if you are a member of the org).
2. Follow steps 3-7 above.
3. Request review from the maintainer team in `CODEOWNERS`.

### PR Requirements

All pull requests must pass the following automated checks before merge:

- [ ] **CI lint and tests** (`ci`) — Ruff, Black, mypy, pytest matrix
- [ ] **Security scan** (`codeql`, `dep-scan`) — SAST and dependency audit
- [ ] **License check** (`license-check`) — Dependency license compatibility + SPDX headers
- [ ] **Pattern check** (`pattern-check`) — No internal URLs, IPs, or corporate email addresses
- [ ] **CLA signed** (for external contributors)

Additionally:

- At least **1 maintainer approval** is required.
- All review conversations must be resolved.
- The branch must be up to date with `main`.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100).
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Use [mypy](https://mypy-lang.org/) for type checking.
- All public functions and classes must have docstrings (NumPy style, as used
  throughout `src/`).

### File Headers

Every Python source file must include the copyright header:

```python
# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use |
|:---|:---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring (no feature/fix) |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance tasks |

## Testing

- Write tests for all new functionality.
- Use [pytest](https://docs.pytest.org/) as the test framework.
- Place tests in the `tests/` directory.
- Run the full test suite before submitting a PR:
  ```bash
  pytest tests/ -v --cov=src
  ```
- Minimum code coverage target: **80%**.

> Tests must not require network access: the German Credit dataset loader is
> mocked in the test suite. Only the experiment scripts fetch from OpenML.

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to the terms of our Contributor License Agreement. The [CLA Assistant](https://cla-assistant.io/) bot will automatically check your PR and ask you to sign the CLA if you have not already done so.

The CLA ensures that contributions can be distributed under the project's Apache 2.0 license.

## Release Process

This project follows [Semantic Versioning (SemVer)](https://semver.org/):

- **MAJOR** — Incompatible API changes
- **MINOR** — New features (backward-compatible)
- **PATCH** — Bug fixes (backward-compatible)

Releases are managed by maintainers. If you believe a release is warranted, open an issue to discuss.

---

Thank you for contributing to **causal-perception-implementation**!
