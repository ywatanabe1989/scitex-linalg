---
name: scitex-linalg
description: |
  [WHAT] Small linear-algebra helpers.
  [WHEN] Use when working with scitex-linalg APIs or when the user mentions scitex.linalg..
  [HOW] `import scitex_linalg` for the Python API; see leaf skills for entry points.
tags: [scitex-linalg]
primary_interface: python
interfaces:
  python: 2
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-linalg/src/scitex_linalg/_skills/scitex-linalg/SKILL.md
---


> **Interfaces:** Python ⭐⭐ · CLI — · MCP — · Skills ⭐⭐ · Hook — · HTTP —

# scitex-linalg

Small linear-algebra helpers — `cdist`, `edist`, `euclidean_distance`, `cosine`, `nannorm`, `rebase_a_vec`, `three_line_lengths_to_coords`, `geometric_median` (optional torch). Drop-in replacement for one-line scipy.spatial dependencies and bespoke geometric-median implementations.

See README.md and the package's public `__init__.py` for the full
function list. This skill leaf exists so agents discover the package
exists and roughly what shape it has — refer to the source for
signatures.
