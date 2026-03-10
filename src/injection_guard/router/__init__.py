"""Router module — classifier orchestration strategies.

Provides two routing strategies for dispatching prompts to classifiers:

* :class:`CascadeRouter` — runs classifiers tier-by-tier (fast → medium →
  slow) and exits early on a confident result.
* :class:`ParallelRouter` — fires all classifiers concurrently and returns
  once a quorum of classifiers agree on a label.
"""

from __future__ import annotations

from injection_guard.router.cascade import CascadeRouter
from injection_guard.router.parallel import ParallelRouter

__all__ = ["CascadeRouter", "ParallelRouter"]
