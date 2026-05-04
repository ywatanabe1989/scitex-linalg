---
description: |
  [TOPIC] Python API
  [DETAILS] All public callables — distances, similarity, geometric median, vector rebasing, triangle-side coordinate solver.
tags: [scitex-linalg-python-api]
---

# Python API

```python
import scitex_linalg as sla
```

## Distances

| Callable | Purpose |
|---|---|
| `euclidean_distance(a, b)` | Scalar L2 distance between two vectors |
| `edist(a, b)` | Short-name alias |
| `cdist(A, B)` | Pairwise distance matrix (numpy-friendly wrapper around `scipy.spatial.distance.cdist`) |

## Similarity

| Callable | Purpose |
|---|---|
| `cosine(a, b)` | Cosine similarity |
| `nannorm(v)` | NaN-aware unit-norm |

## Geometry

| Callable | Purpose |
|---|---|
| `geometric_median(X, backend="numpy" or "torch")` | Robust multivariate median (Weiszfeld iterative) |
| `rebase_a_vec(v, basis)` | Express `v` in a new orthonormal basis |
| `three_line_lengths_to_coords(a, b, c)` | Solve triangle side-lengths to 2-D coordinates |

## Notes

- `geometric_median(..., backend="torch")` requires the `torch` extra; raises
  `ImportError` if torch is missing.
- All functions accept and return plain numpy arrays unless noted.

## See also

- `scipy.spatial.distance` — heavyweight alternative for huge `cdist` cases
- `scitex-stats` — for statistical tests on the distances
