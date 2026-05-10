"""Modelling utilities for CMAPSS remaining-useful-life regression.

This module is deliberately small and scikit-learn-native. The pipeline
returned by :func:`build_baseline_regressor` is a regular
:class:`sklearn.pipeline.Pipeline` and can be cross-validated, persisted,
or wrapped in any other sklearn meta-estimator without modification.

The two evaluation metrics are the ones the CMAPSS literature
universally reports:

* :func:`rmse` — root mean squared error in cycles.
* :func:`s_score` — the asymmetric scoring function from the original
  PHM 2008 challenge. Late predictions (predicting more remaining life
  than the engine actually has) are penalised exponentially harder than
  early predictions, reflecting the operational cost asymmetry: missing
  a failure is worse than scheduling maintenance a few cycles early.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "build_baseline_regressor",
    "make_xy",
    "rmse",
    "s_score",
]


def make_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str = "RUL",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a model-ready ``(X, y)`` pair from a long-format DataFrame.

    Args:
        df: Long-format DataFrame produced by :mod:`pdm.data` /
            :mod:`pdm.features`.
        feature_columns: Names of the columns to include as features.
        target_column: Name of the regression target column.
            Defaults to ``"RUL"``.

    Returns:
        A pair of numpy arrays: ``X`` of shape ``(n_samples, n_features)``
        and ``y`` of shape ``(n_samples,)``.

    Raises:
        KeyError: If any of ``feature_columns`` or ``target_column`` is
            missing from the DataFrame.
        ValueError: If ``feature_columns`` is empty.
    """
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    missing = [c for c in [*feature_columns, target_column] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    X = df[list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
    y = df[target_column].to_numpy(dtype=np.float64, copy=True)
    return X, y


def build_baseline_regressor(*, alpha: float = 1.0) -> Pipeline:
    """Build the baseline RUL regression pipeline.

    The pipeline is intentionally simple: standardise features, then fit
    an L2-regularised linear regressor. It serves as the floor against
    which more elaborate models (gradient boosting, sequence models)
    must demonstrably improve.

    Args:
        alpha: L2 regularisation strength. Higher values shrink
            coefficients more aggressively. Defaults to ``1.0``.

    Returns:
        A scikit-learn :class:`Pipeline` ready for ``fit`` / ``predict``.

    Raises:
        ValueError: If ``alpha`` is negative.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in the same units as the target (cycles)."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def s_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    early_decay: float = 13.0,
    late_decay: float = 10.0,
) -> float:
    r"""CMAPSS asymmetric score (Saxena et al., 2008).

    For each sample with prediction error ``d = y_pred - y_true``,

    .. math::

        s_i = \\begin{cases}
            \\exp(-d_i / a) - 1 & d_i < 0 \\quad (\\text{early}) \\\\
            \\exp( d_i / b) - 1 & d_i \\geq 0 \\quad (\\text{late})
        \\end{cases}

    with the standard challenge constants ``a = 13`` and ``b = 10``.
    Lower is better; a perfect prediction returns ``0``.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.
        early_decay: Decay constant for early predictions. Defaults
            to ``13.0`` (the official challenge value).
        late_decay: Decay constant for late predictions. Defaults
            to ``10.0`` (the official challenge value).

    Returns:
        Sum of per-sample asymmetric losses.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes,
            or if either decay constant is non-positive.
    """
    if early_decay <= 0 or late_decay <= 0:
        raise ValueError(
            f"Decay constants must be strictly positive; "
            f"got early_decay={early_decay}, late_decay={late_decay}."
        )

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}")

    diff = y_pred_arr - y_true_arr
    early = np.exp(-diff[diff < 0] / early_decay) - 1.0
    late = np.exp(diff[diff >= 0] / late_decay) - 1.0
    return float(early.sum() + late.sum())
