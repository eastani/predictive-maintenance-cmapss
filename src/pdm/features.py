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
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "add_operating_regime",
    "add_rolling_features",
    "apply_regime_normalizer",
    "clip_rul",
    "drop_constant_sensors",
    "fit_operating_regime_model",
    "fit_regime_normalizer",
]


@dataclass(frozen=True)
class OperatingRegimeModel:
    """K-means operating-regime assignment fitted on scaled settings."""

    setting_columns: tuple[str, ...]
    centers: np.ndarray
    setting_means: np.ndarray
    setting_stds: np.ndarray
    regime_column: str = "op_regime"


@dataclass(frozen=True)
class RegimeNormalizer:
    """Per-regime normalization statistics for sensor channels."""

    columns: tuple[str, ...]
    regime_column: str
    means: pd.DataFrame
    stds: pd.DataFrame
    suffix: str = "_regime_z"


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


def _default_setting_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("op_setting_")]


def fit_operating_regime_model(
    df: pd.DataFrame,
    *,
    n_regimes: int = 6,
    setting_columns: Sequence[str] | None = None,
    random_state: int = 42,
    regime_column: str = "op_regime",
) -> OperatingRegimeModel:
    """Fit a K-means model over standardized operational settings.

    FD002 and FD004 contain multiple operating conditions. Assigning each
    row to an operating regime lets downstream feature engineering normalize
    sensors within comparable operating states instead of mixing regimes.
    The operational settings have very different numeric scales, so K-means
    is fitted on z-scored settings rather than raw values.

    Args:
        df: Input long-format DataFrame.
        n_regimes: Number of operating regimes to identify.
        setting_columns: Operational-setting columns. When ``None``, columns
            whose names start with ``"op_setting_"`` are used.
        random_state: K-means random seed.
        regime_column: Name to use when applying this model.

    Returns:
        Fitted :class:`OperatingRegimeModel` containing cluster centers.

    Raises:
        ValueError: If no setting columns are available, or ``n_regimes`` is
            invalid for the number of rows.
    """
    if n_regimes <= 0:
        raise ValueError(f"n_regimes must be strictly positive, got {n_regimes}.")
    if setting_columns is None:
        setting_columns = _default_setting_columns(df)
    if not setting_columns:
        raise ValueError("No operational-setting columns found.")
    if n_regimes > len(df):
        raise ValueError(f"n_regimes={n_regimes} exceeds number of rows ({len(df)}).")

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - scikit-learn is a core dependency
        raise ImportError("scikit-learn is required for operating-regime clustering.") from exc

    settings = df[list(setting_columns)].to_numpy(dtype=np.float64, copy=True)
    setting_means = settings.mean(axis=0)
    setting_stds = settings.std(axis=0)
    setting_stds = np.where(setting_stds == 0.0, 1.0, setting_stds)
    scaled_settings = (settings - setting_means) / setting_stds

    model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")
    model.fit(scaled_settings)
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    return OperatingRegimeModel(
        setting_columns=tuple(setting_columns),
        centers=centers,
        setting_means=setting_means,
        setting_stds=setting_stds,
        regime_column=regime_column,
    )


def add_operating_regime(df: pd.DataFrame, model: OperatingRegimeModel) -> pd.DataFrame:
    """Append an operating-regime label using a fitted regime model."""
    missing = [column for column in model.setting_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing operational-setting columns: {missing}")

    settings = df[list(model.setting_columns)].to_numpy(dtype=np.float64, copy=True)
    scaled_settings = (settings - model.setting_means) / model.setting_stds
    distances = ((scaled_settings[:, None, :] - model.centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1).astype("int64")

    out = df.copy()
    out[model.regime_column] = labels
    return out


def fit_regime_normalizer(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    regime_column: str = "op_regime",
    suffix: str = "_regime_z",
) -> RegimeNormalizer:
    """Fit per-regime mean/std statistics for sensor normalization."""
    if regime_column not in df.columns:
        raise KeyError(f"Missing regime column: {regime_column}")
    if columns is None:
        columns = [column for column in df.columns if column.startswith("sensor_")]
    if not columns:
        raise ValueError("No candidate columns found for regime normalization.")

    grouped = df.groupby(regime_column, sort=True)[list(columns)]
    means = grouped.mean()
    stds = grouped.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return RegimeNormalizer(
        columns=tuple(columns),
        regime_column=regime_column,
        means=means,
        stds=stds,
        suffix=suffix,
    )


def apply_regime_normalizer(
    df: pd.DataFrame,
    normalizer: RegimeNormalizer,
) -> pd.DataFrame:
    """Append per-regime z-score columns using fitted normalization stats."""
    if normalizer.regime_column not in df.columns:
        raise KeyError(f"Missing regime column: {normalizer.regime_column}")

    missing_columns = [column for column in normalizer.columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing normalization columns: {missing_columns}")

    regimes = set(df[normalizer.regime_column].unique())
    known_regimes = set(normalizer.means.index)
    unknown = sorted(regimes - known_regimes)
    if unknown:
        raise ValueError(f"Unknown operating regimes: {unknown}")

    out = df.copy()
    for column in normalizer.columns:
        out[f"{column}{normalizer.suffix}"] = [
            (float(value) - float(normalizer.means.loc[regime, column]))
            / float(normalizer.stds.loc[regime, column])
            for regime, value in zip(
                out[normalizer.regime_column],
                out[column],
                strict=True,
            )
        ]
    return out


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
