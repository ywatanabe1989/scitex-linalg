"""SciTeX linalg — small linear-algebra helpers (distance, geometric median, misc)."""

from ._distance import cdist, edist, euclidean_distance
from ._misc import cosine, nannorm, rebase_a_vec, three_line_lengths_to_coords

# geometric_median requires torch + geom-median (optional extra: pip install
# scitex-linalg[torch]). Lazy-load so bare install + import works.
try:
    from ._geometric_median import geometric_median
except ImportError:
    geometric_median = None

__version__ = "0.1.0"

__all__ = [
    "euclidean_distance",
    "cdist",
    "edist",
    "geometric_median",
    "cosine",
    "nannorm",
    "rebase_a_vec",
    "three_line_lengths_to_coords",
]
