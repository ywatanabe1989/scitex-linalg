---
description: |
  [TOPIC] Quick start
  [DETAILS] Smallest example — pairwise distances, cosine similarity, geometric median.
tags: [scitex-linalg-quick-start]
---

# Quick Start

## Pairwise distances

```python
import numpy as np
import scitex_linalg as sla

A = np.random.randn(100, 3)
B = np.random.randn(50, 3)

D = sla.cdist(A, B)            # shape (100, 50), Euclidean
d = sla.euclidean_distance(A[0], B[0])
```

## Cosine + NaN-safe norm

```python
sim = sla.cosine(A[0], B[0])
unit = sla.nannorm(vec_with_nans)
```

## Geometric median (robust centroid)

```python
gm = sla.geometric_median(A)              # numpy
gm = sla.geometric_median(A, backend="torch")   # requires [torch] extra
```

## Coordinates from triangle side lengths

```python
xy_a, xy_b, xy_c = sla.three_line_lengths_to_coords(2.0, 3.0, 4.0)
```

## Next

- [03_python-api.md](03_python-api.md) — full public surface
- [SKILL.md](SKILL.md) — overview
