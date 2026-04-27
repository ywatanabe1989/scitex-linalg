"""Quickstart for scitex_linalg.

Demonstrates Euclidean distance and geometric median on small numpy arrays.
"""

import numpy as np

import scitex_linalg as sl


def main() -> int:
    rng = np.random.default_rng(0)

    # Two random vectors -> Euclidean distance
    u = rng.normal(size=(3, 100))
    v = rng.normal(size=(3, 100))
    dist_per_sample = sl.euclidean_distance(u, v, axis=0)
    print(f"Euclidean distance per sample (first 5): {dist_per_sample[:5]}")
    print(f"Mean distance: {dist_per_sample.mean():.4f}")

    # Geometric median of a point cloud (robust centre)
    # Requires the [torch] extra (torch + geom-median). Skip gracefully if absent.
    if sl.geometric_median is None:
        print("geometric_median unavailable (install scitex-linalg[torch]); skipping.")
        return 0

    points = rng.normal(loc=[1.0, 2.0, -1.0], scale=0.5, size=(200, 3)).T
    gmed = sl.geometric_median(points, dim=-1)
    print(f"Geometric median (≈ centre): {np.round(gmed, 3)}")

    # Compare to arithmetic mean (sensitive to outliers)
    points_with_outliers = np.concatenate(
        [points, rng.normal(loc=50.0, size=(3, 5))], axis=-1
    )
    arith_mean = points_with_outliers.mean(axis=-1)
    robust_med = sl.geometric_median(points_with_outliers, dim=-1)
    print(f"Mean with outliers:   {np.round(arith_mean, 3)}")
    print(f"Median with outliers: {np.round(robust_med, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
