"""Phase IV-C §5: Geometry-prioritized asynchronous DP.

Priority scoring combines geometry gain with residual-based prioritization.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_geometry_priority(
    residuals: NDArray[np.float64],
    natural_shifts: NDArray[np.float64],
    mode: str = "combined",
) -> NDArray[np.float64]:
    """Compute priority scores for async DP backup ordering.

    Parameters
    ----------
    residuals : NDArray[np.float64], shape (|S|,) or (|S|, |A|)
        Bellman residuals per state(-action).
    natural_shifts : NDArray[np.float64], same shape as residuals
        Natural-shift magnitudes |u(s)| or |u(s,a)|.
    mode : str
        Priority mode. Supported: ``"residual_only"``,
        ``"geometry_only"``, ``"combined"``. Default is ``"combined"``.

    Returns
    -------
    NDArray[np.float64], same shape as residuals
        Priority scores; higher values indicate higher backup priority.
    """
    raise NotImplementedError("Phase IV-C implementation pending")
