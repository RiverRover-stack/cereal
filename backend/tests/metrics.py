"""Shared forecast-accuracy metrics for `tests.backtest` and `tests.test_forecasting`.

MAE and RMSE are in the target's own units (dollars); MAPE is scale-free, which is
what makes it comparable across folds that land on different revenue levels.
"""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, as a 0-100 number.

    Days where actual revenue is 0 are excluded: dividing by zero makes the
    percentage undefined, not large, and folding it into the mean would distort
    every other day's contribution.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    )
