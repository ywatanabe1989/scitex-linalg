---
description: |
  [TOPIC] Installation
  [DETAILS] pip install scitex-linalg. Pure-Python; depends on numpy + scipy + sympy. Optional torch extra for `geometric_median`.
tags: [scitex-linalg-installation]
---

# Installation

## Standard

```bash
pip install scitex-linalg
```

Pulls `numpy`, `scipy`, and `sympy`. No system deps.

## Optional extras

| Extra | Purpose |
|---|---|
| `torch` | `geometric_median()` GPU-friendly path via PyTorch |
| `dev` | Test + lint tooling |
| `docs` | Sphinx + RTD theme |
| `all` | Everything above |

```bash
pip install 'scitex-linalg[torch]'
```

## Verify

```bash
python -c "import scitex_linalg; print(scitex_linalg.__version__)"
python -c "from scitex_linalg import euclidean_distance, cdist; print('ok')"
```

## Editable install (development)

```bash
git clone https://github.com/ywatanabe1989/scitex-linalg
cd scitex-linalg
pip install -e '.[dev]'
```
