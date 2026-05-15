"""Reusable evaluation harness for CMAPSS RUL experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from pdm.data import CMAPSSData
from pdm.features import (
    add_operating_regime,
    add_rolling_features,
    apply_regime_normalizer,
    clip_rul,
    drop_constant_sensors,
    fit_operating_regime_model,
    fit_regime_normalizer,
)
from pdm.models import make_xy, rmse, s_score


class Regressor(Protocol):
    """Small estimator protocol shared by scikit-learn-compatible models."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Regressor:
        """Fit the estimator."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values."""


@dataclass(frozen=True)
class ModellingDataset:
    """Train/test matrices and feature metadata for one CMAPSS subset."""

    train: pd.DataFrame
    test_final: pd.DataFrame
    feature_columns: tuple[str, ...]
    dropped_sensors: tuple[str, ...]
    use_regime_features: bool


@dataclass(frozen=True)
class EvaluationResult:
    """Headline metrics for one model on one subset."""

    subset: str
    model_name: str
    rmse: float
    s_score: float
    n_train_samples: int
    n_test_units: int
    n_features: int
    use_regime_features: bool

    def as_dict(self) -> dict[str, str | int | float | bool]:
        """Return a tabular representation suitable for DataFrame construction."""
        return {
            "subset": self.subset,
            "model": self.model_name,
            "rmse": self.rmse,
            "s_score": self.s_score,
            "n_train_samples": self.n_train_samples,
            "n_test_units": self.n_test_units,
            "n_features": self.n_features,
            "use_regime_features": self.use_regime_features,
        }


def _last_test_cycle(test: pd.DataFrame, test_rul: pd.DataFrame) -> pd.DataFrame:
    last_cycle_index = test.groupby("unit_id")["cycle"].transform("max") == test["cycle"]
    final = test[last_cycle_index].sort_values("unit_id").reset_index(drop=True).copy()
    aligned_rul = test_rul.set_index("unit_id").loc[final["unit_id"]]["RUL"].to_numpy()
    final["RUL"] = aligned_rul
    return final


def _feature_columns(df: pd.DataFrame) -> tuple[str, ...]:
    columns = [column for column in df.columns if column.startswith("sensor_")]
    if not columns:
        raise ValueError("No feature columns found after preprocessing.")
    return tuple(columns)


def build_modelling_dataset(
    data: CMAPSSData,
    *,
    windows: Iterable[int] = (5, 10, 20),
    max_rul: int = 125,
    use_regime_features: bool = False,
    n_regimes: int = 6,
) -> ModellingDataset:
    """Build aligned train/test modelling frames for one CMAPSS subset.

    The test labels in CMAPSS apply only to each unit's final observed cycle.
    This function keeps that convention explicit by returning ``test_final``
    with one labelled row per test unit.
    """
    train, dropped = drop_constant_sensors(data.train)
    test = data.test.drop(columns=dropped, errors="ignore")
    sensor_columns = tuple(column for column in train.columns if column.startswith("sensor_"))

    if use_regime_features:
        regime_model = fit_operating_regime_model(train, n_regimes=n_regimes)
        train = add_operating_regime(train, regime_model)
        test = add_operating_regime(test, regime_model)
        normalizer = fit_regime_normalizer(train, columns=sensor_columns)
        train = apply_regime_normalizer(train, normalizer)
        test = apply_regime_normalizer(test, normalizer)
        sensor_columns = tuple(column for column in train.columns if column.startswith("sensor_"))

    train = add_rolling_features(train, windows=windows, columns=sensor_columns)
    test = add_rolling_features(test, windows=windows, columns=sensor_columns)
    train = train.assign(RUL=clip_rul(train["RUL"], max_rul=max_rul))
    test_final = _last_test_cycle(test, data.test_rul)

    return ModellingDataset(
        train=train,
        test_final=test_final,
        feature_columns=_feature_columns(train),
        dropped_sensors=tuple(dropped),
        use_regime_features=use_regime_features,
    )


def evaluate_regressor(
    data: CMAPSSData,
    *,
    model_name: str,
    model_factory: Callable[[], Regressor],
    windows: Iterable[int] = (5, 10, 20),
    max_rul: int = 125,
    use_regime_features: bool = False,
    n_regimes: int = 6,
) -> EvaluationResult:
    """Fit and evaluate one model on a CMAPSS subset."""
    modelling = build_modelling_dataset(
        data,
        windows=windows,
        max_rul=max_rul,
        use_regime_features=use_regime_features,
        n_regimes=n_regimes,
    )
    x_train, y_train = make_xy(modelling.train, modelling.feature_columns)
    x_test, y_test = make_xy(modelling.test_final, modelling.feature_columns)

    model = model_factory()
    model.fit(x_train, y_train)
    predictions = np.clip(model.predict(x_test), a_min=0.0, a_max=None)

    return EvaluationResult(
        subset=data.subset,
        model_name=model_name,
        rmse=rmse(y_test, predictions),
        s_score=s_score(y_test, predictions),
        n_train_samples=x_train.shape[0],
        n_test_units=x_test.shape[0],
        n_features=x_train.shape[1],
        use_regime_features=use_regime_features,
    )
