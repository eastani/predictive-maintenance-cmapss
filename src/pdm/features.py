"""Feature engineering for CMAPSS run-to-failure trajectories.

This module collects the feature-level transformations that consistently
appear in well-performing CMAPSS solutions, packaged as small composable
functions that operate on the long-format DataFrames returned by
:mod:`pdm.data`.

Three transformations are exposed:

* :func:`drop_constant_sensors` — discards sensor channels whose variance
  is essentially zero, since they carry no degradation information and
  inject noise into downstream models.
* :func:`clip_rul` — applies the piecewise-linear RUL relabelling that
  has become standard practice on CMAPSS: the degradation signal is
  approximately flat early on, so capping the target prevents the model
  from chasing label noise in the long healthy regime.
* :func:`add_rolling_features` — computes per-unit rolling mean and
  standard deviation over a configurable set of window sizes. The rolling
  statistics are computed *within* each engine so that information from
  one trajectory never leaks into another.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

__all__ = [
    "add_rolling_features",
    "clip_rul",
    "drop_constant_sensors",
]


def drop_constant_sensors(
    df: pd.DataFrame,
    *,
    threshold: float = 1e-6,
    candidate_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop sensor columns whose variance is below ``threshold``.

    On CMAPSS, several sensor channels (e.g., sensors 1, 5, 6, 10, 16, 18,
    19 in FD001) are reported as constants across the entire fleet. They
    add no information and frequently destabilise gradient-based learners.

    Args:
        df: Input long-format DataFrame.
        threshold: Variance below which a column is considered constant.
            Defaults to ``1e-6``.
        candidate_columns: Optional explicit list of columns to consider.
            When ``None``, all columns whose name starts with ``sensor_``
            are considered.

    Returns:
        Tuple of ``(filtered_df, dropped_columns)`` where ``filtered_df``
        is a copy of ``df`` with constant sensor columns removed and
        ``dropped_columns`` is the sorted list of names that were dropped.
    """
    if candidate_columns is None:
        candidate_columns = [c for c in df.columns if c.startswith("sensor_")]

    variances = df[list(candidate_columns)].var(numeric_only=True)
    dropped = sorted(variances[variances < threshold].index.tolist())
    return df.drop(columns=dropped), dropped


def clip_rul(rul: pd.Series, *, max_rul: int = 125) -> pd.Series:
    """Apply piecewise-linear RUL relabelling.

    The damage propagation that CMAPSS simulates is dominated by the final
    portion of the trajectory; the first many cycles have negligible
    degradation. Capping the RUL target therefore stops the regressor from
    investing capacity in fitting the flat early regime — a relabelling
    convention introduced by Heimes (2008) and used in essentially every
    competitive CMAPSS submission since.

    Args:
        rul: Series of integer remaining-useful-life values.
        max_rul: Maximum value to retain. All RULs above this cap are set
            to ``max_rul``. Common values in the literature are 125 or 130.

    Returns:
        Series of clipped RUL values, preserving the original index and
        name.

    Raises:
        ValueError: If ``max_rul`` is non-positive.
    """
    if max_rul <= 0:
        raise ValueError(f"max_rul must be strictly positive, got {max_rul}.")
    return rul.clip(upper=max_rul)


def add_rolling_features(
    df: pd.DataFrame,
    *,
    windows: Iterable[int] = (5, 10, 20),
    columns: Sequence[str] | None = None,
    unit_column: str = "unit_id",
    statistics: Sequence[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Append per-unit rolling statistics to a long-format DataFrame.

    For each window size ``w`` and statistic ``s`` (e.g., ``"mean"``), a
    new column ``"{column}_{statistic}_{w}"`` is created. Rolling
    computations are performed within each ``unit_id`` so that information
    from one trajectory never leaks into another.

    The first ``w - 1`` rows of each unit are filled with the corresponding
    expanding statistic to avoid leading NaNs that would otherwise force
    callers to drop large chunks of data.

    Args:
        df: Input DataFrame.
        windows: Window sizes to compute. Each must be a strictly positive
            integer.
        columns: Columns to compute statistics on. When ``None``, all
            columns whose name starts with ``sensor_`` are used.
        unit_column: Column identifying the engine / unit. Defaults to
            ``"unit_id"``.
        statistics: Statistics to compute. Each must be the name of a
            method exposed by :class:`pandas.api.typing.RollingGroupby`
            (e.g., ``"mean"``, ``"std"``, ``"min"``, ``"max"``).

    Returns:
        A new DataFrame with the rolling-statistic columns appended. The
        original columns are preserved in their original order.

    Raises:
        ValueError: If any window size is non-positive, or if no candidate
            columns are found.
    """
    if any(w <= 0 for w in windows):
        raise ValueError(f"All window sizes must be strictly positive, got {list(windows)}.")

    if columns is None:
        columns = [c for c in df.columns if c.startswith("sensor_")]
    if not columns:
        raise ValueError("No candidate columns found for rolling features.")

    out = df.copy()
    grouped = out.groupby(unit_column, sort=False)[list(columns)]

    for window in windows:
        for stat in statistics:
            rolling = getattr(grouped.rolling(window=window, min_periods=1), stat)()
            # `rolling` carries a (unit_id, original_index) MultiIndex; drop
            # the unit level so the values realign with `out`.
            rolling = rolling.reset_index(level=0, drop=True)
            rolling.columns = [f"{c}_{stat}_{window}" for c in rolling.columns]
            out = pd.concat([out, rolling], axis=1)

    # `std` with min_periods=1 yields NaN at the first row of each unit; fill
    # those with 0.0 so downstream models do not need to special-case them.
    std_cols = [c for c in out.columns if "_std_" in c]
    if std_cols:
        out[std_cols] = out[std_cols].fillna(0.0)

    return out
