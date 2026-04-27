# scitex-linalg

<!-- scitex-badges:start -->
[![PyPI](https://img.shields.io/pypi/v/scitex-linalg.svg)](https://pypi.org/project/scitex-linalg/)
[![Python](https://img.shields.io/pypi/pyversions/scitex-linalg.svg)](https://pypi.org/project/scitex-linalg/)
[![Tests](https://github.com/ywatanabe1989/scitex-linalg/actions/workflows/test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-linalg/actions/workflows/test.yml)
[![Install Test](https://github.com/ywatanabe1989/scitex-linalg/actions/workflows/install-test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-linalg/actions/workflows/install-test.yml)
[![Coverage](https://codecov.io/gh/ywatanabe1989/scitex-linalg/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/scitex-linalg)
[![Docs](https://readthedocs.org/projects/scitex-linalg/badge/?version=latest)](https://scitex-linalg.readthedocs.io/en/latest/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- scitex-badges:end -->


Small linear-algebra helpers extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-linalg            # core (numpy/scipy/sympy)
pip install "scitex-linalg[torch]"   # + geometric_median (torch + geom-median)
```

## API

```python
import scitex_linalg as sxl

sxl.euclidean_distance(u, v, axis=0)      # element-wise Euclidean distance
sxl.cdist(u, v)                           # pairwise distances
sxl.edist(u, v)                           # alias for cdist
sxl.cosine(v1, v2)                        # cosine similarity (NaN-safe)
sxl.nannorm(v, axis=-1)                   # NaN-aware vector norm
sxl.rebase_a_vec(v, v_base)               # project v onto v_base basis
sxl.three_line_lengths_to_coords(a, b, c) # triangle side lengths -> 2-D coords
sxl.geometric_median(xx, dim=-1)          # torch geometric median (requires [torch] extra)
```

## Status

Standalone fork of `scitex.linalg` — intended to remain importable as
`scitex.linalg` via the SciTeX umbrella package's bridge module. Decorators
(`numpy_fn`, `torch_fn`, `wrap`) are vendored under `_vendor_decorators/`
to keep the package free of `scitex.*` runtime deps; when `scitex-decorators`
is split out, those will be replaced with a direct dependency.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).
