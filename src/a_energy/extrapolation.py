"""Convergence diagnostics for a sequence eta_k computed on nested meshes (h halved each step)."""
from typing import Dict, Optional, Sequence

import numpy as np


def observed_order_and_limit(etas: Sequence[float], assumed_order: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Observed order p from the last three levels and the Richardson-type limit.

    d1 = eta_{k-2} - eta_{k-1},  d2 = eta_{k-1} - eta_k  (same sign and |d2| < |d1| required),
    p = log2(d1 / d2),  limit = eta_k - d2 / (2^p - 1).
    If ``assumed_order`` is given, the limit with that order is also returned (``limit_assumed``).
    This is a HEURISTIC (it assumes the asymptotic regime); it is reported next to, never instead of,
    the raw ladder.
    """
    e = [float(v) for v in etas]
    out: Dict[str, Optional[float]] = {"order": None, "limit": None, "limit_assumed": None}
    if len(e) < 3:
        return out
    d1, d2 = e[-3] - e[-2], e[-2] - e[-1]
    if d1 * d2 > 0 and abs(d2) < abs(d1):
        p = float(np.log2(d1 / d2))
        out["order"] = p
        out["limit"] = float(e[-1] - d2 / (2.0**p - 1.0))
    if assumed_order is not None and d2 != 0.0:
        out["limit_assumed"] = float(e[-1] - d2 / (2.0**assumed_order - 1.0))
    return out
