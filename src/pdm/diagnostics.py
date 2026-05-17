"""Prediction-error diagnostics for CMAPSS RUL models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.models import rmse, s_score

__all__ = [
    "prediction_error_breakdown",
    "prediction_error_by_group",
    "prediction_error_by_rul_band",
    "prediction_error_with_target_caps",
    "prediction_rows",
]


def prediction_rows(
    *,
    subset: str,
    model: str,
    seed: int | None,
    unit_ids: np.ndarray,
    end_cycles: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build one row per test-unit prediction with signed error columns."""
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    unit_array = np.asarray(unit_ids, dtype=np.int64)
    cycle_array = np.asarray(end_cycles, dtype=np.int64)
    expected_shape = y_true_array.shape
    if y_true_array.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got {y_true_array.shape}.")
    if y_pred_array.shape != expected_shape:
        raise ValueError(f"y_pred must have shape {expected_shape}, got {y_pred_array.shape}.")
    if unit_array.shape != expected_shape:
        raise ValueError(f"unit_ids must have shape {expected_shape}, got {unit_array.shape}.")
    if cycle_array.shape != expected_shape:
        raise ValueError(f"end_cycles must have shape {expected_shape}, got {cycle_array.shape}.")

    error = y_pred_array - y_true_array
    return pd.DataFrame(
        {
            "subset": subset,
            "model": model,
            "seed": seed,
            "unit_id": unit_array,
            "end_cycle": cycle_array,
            "y_true": y_true_array,
            "y_pred": y_pred_array,
            "error": error,
            "abs_error": np.abs(error),
            "late": error > 0.0,
        }
    )


def _metric_row(group: pd.DataFrame, label: str) -> dict[str, str | int | float]:
    y_true = group["y_true"].to_numpy(dtype=np.float64)
    y_pred = group["y_pred"].to_numpy(dtype=np.float64)
    errors = group["error"].to_numpy(dtype=np.float64)
    return {
        "segment": label,
        "n": len(group),
        "rmse": rmse(y_true, y_pred),
        "s_score": s_score(y_true, y_pred),
        "mean_error": float(errors.mean()),
        "mean_abs_error": float(np.abs(errors).mean()),
        "max_abs_error": float(np.abs(errors).max()),
    }


def prediction_error_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize all, early, and late prediction errors."""
    required = {"y_true", "y_pred", "error", "late"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")

    rows = [_metric_row(predictions, "all")]
    early = predictions[~predictions["late"]]
    late = predictions[predictions["late"]]
    if not early.empty:
        rows.append(_metric_row(early, "early_or_exact"))
    if not late.empty:
        rows.append(_metric_row(late, "late"))
    return pd.DataFrame(rows)


def prediction_error_by_rul_band(
    predictions: pd.DataFrame,
    *,
    bins: tuple[float, ...] = (0.0, 50.0, 100.0, 125.0, float("inf")),
    labels: tuple[str, ...] = ("0-50", "50-100", "100-125", "125+"),
) -> pd.DataFrame:
    """Summarize prediction errors by true-RUL band."""
    required = {"y_true", "y_pred", "error"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")
    if len(labels) != len(bins) - 1:
        raise ValueError("labels must contain exactly len(bins) - 1 entries.")

    labelled = predictions.copy()
    labelled["rul_band"] = pd.cut(
        labelled["y_true"],
        bins=list(bins),
        labels=list(labels),
        include_lowest=True,
        right=True,
    )
    rows = []
    for band, group in labelled.groupby("rul_band", observed=True, sort=False):
        row = _metric_row(group, str(band))
        row["segment"] = f"rul_{band}"
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_error_by_group(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    segment_prefix: str | None = None,
) -> pd.DataFrame:
    """Summarize prediction errors by a categorical prediction column."""
    required = {"y_true", "y_pred", "error", group_column}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    labelled = predictions.dropna(subset=[group_column])
    if labelled.empty:
        raise ValueError(f"predictions must contain at least one non-null {group_column} value.")

    rows = []
    prefix = segment_prefix or group_column
    for group_value, group in labelled.groupby(group_column, observed=True, sort=True):
        row = _metric_row(group, str(group_value))
        row["segment"] = f"{prefix}_{group_value}"
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_error_with_target_caps(
    predictions: pd.DataFrame,
    *,
    caps: tuple[float | None, ...] = (None, 125.0),
) -> pd.DataFrame:
    """Compare metrics under raw and capped target conventions.

    ``None`` means raw labels and raw predictions. Numeric caps clip both
    ``y_true`` and ``y_pred`` to the cap before scoring, matching a capped-RUL
    evaluation convention.
    """
    required = {"y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")

    rows = []
    for cap in caps:
        y_true = predictions["y_true"].to_numpy(dtype=np.float64)
        y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
        if cap is None:
            label = "raw"
        else:
            label = f"cap_{cap:g}"
            y_true = np.minimum(y_true, cap)
            y_pred = np.minimum(y_pred, cap)
        error = y_pred - y_true
        rows.append(
            {
                "target_convention": label,
                "n": len(predictions),
                "rmse": rmse(y_true, y_pred),
                "s_score": s_score(y_true, y_pred),
                "mean_error": float(error.mean()),
                "mean_abs_error": float(np.abs(error).mean()),
                "max_abs_error": float(np.abs(error).max()),
            }
        )
    return pd.DataFrame(rows)
