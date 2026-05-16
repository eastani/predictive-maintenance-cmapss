"""Sequence-window builders for CMAPSS recurrent models.

CMAPSS test labels are attached only to the final observed cycle of each
test engine. Sequence models therefore need a different data contract from
the tabular regressors: training can use many labelled cycle-ending windows,
but evaluation must use exactly one final-cycle window per test unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from pdm.data import CMAPSSData
from pdm.features import (
    add_operating_regime,
    apply_regime_normalizer,
    clip_rul,
    drop_constant_sensors,
    fit_operating_regime_model,
    fit_regime_normalizer,
)

__all__ = [
    "SequenceDataset",
    "SequenceWindows",
    "build_sequence_dataset",
    "make_sequence_windows",
]


class SequenceWindows(NamedTuple):
    """Padded sequence windows and the metadata needed to audit them."""

    x: np.ndarray
    y: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class SequenceDataset:
    """Train/test sequence tensors for one CMAPSS subset."""

    train_x: np.ndarray
    train_y: np.ndarray
    train_unit_ids: np.ndarray
    train_end_cycles: np.ndarray
    train_lengths: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_unit_ids: np.ndarray
    test_end_cycles: np.ndarray
    test_lengths: np.ndarray
    feature_columns: tuple[str, ...]
    dropped_sensors: tuple[str, ...]
    sequence_length: int
    padding_value: float
    use_regime_features: bool

    @property
    def n_features(self) -> int:
        """Number of features per sequence step."""
        return len(self.feature_columns)


def _validate_window_args(sequence_length: int, stride: int) -> None:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be strictly positive, got {sequence_length}.")
    if stride <= 0:
        raise ValueError(f"stride must be strictly positive, got {stride}.")


def _feature_columns(df: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(column for column in df.columns if column.startswith("sensor_"))
    if not columns:
        raise ValueError("No feature columns found after preprocessing.")
    return columns


def _make_padded_window(
    values: np.ndarray,
    *,
    end_index: int,
    sequence_length: int,
    padding_value: float,
) -> tuple[np.ndarray, int]:
    start_index = max(0, end_index + 1 - sequence_length)
    raw_window = values[start_index : end_index + 1]
    valid_length = raw_window.shape[0]

    padded = np.full(
        shape=(sequence_length, values.shape[1]),
        fill_value=padding_value,
        dtype=np.float64,
    )
    padded[-valid_length:] = raw_window
    return padded, valid_length


def make_sequence_windows(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    sequence_length: int,
    stride: int = 1,
    target_column: str = "RUL",
    unit_column: str = "unit_id",
    cycle_column: str = "cycle",
    padding_value: float = 0.0,
) -> SequenceWindows:
    """Create left-padded per-unit sequence windows from a labelled frame.

    Each output row contains the previous ``sequence_length`` observations up
    to and including the current cycle. Early-cycle windows are left-padded
    and accompanied by a ``lengths`` vector so recurrent models can use masks
    or packed sequences rather than treating padding as real sensor history.
    """
    _validate_window_args(sequence_length=sequence_length, stride=stride)
    required = [unit_column, cycle_column, target_column, *feature_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []
    lengths: list[int] = []

    ordered = df.sort_values([unit_column, cycle_column], ignore_index=True)
    for unit_id, group in ordered.groupby(unit_column, sort=False):
        values = group.loc[:, list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
        target_values = group.loc[:, target_column].to_numpy(dtype=np.float64, copy=True)
        cycle_values = group.loc[:, cycle_column].to_numpy(dtype=np.int64, copy=True)

        for end_index in range(0, len(group), stride):
            window, valid_length = _make_padded_window(
                values,
                end_index=end_index,
                sequence_length=sequence_length,
                padding_value=padding_value,
            )
            windows.append(window)
            targets.append(float(target_values[end_index]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(cycle_values[end_index]))
            lengths.append(valid_length)

    return SequenceWindows(
        x=np.stack(windows).astype(np.float64, copy=False),
        y=np.asarray(targets, dtype=np.float64),
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        end_cycles=np.asarray(end_cycles, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
    )


def _make_final_test_windows(
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    sequence_length: int,
    unit_column: str = "unit_id",
    cycle_column: str = "cycle",
    padding_value: float = 0.0,
) -> SequenceWindows:
    required = [unit_column, cycle_column, *feature_columns]
    missing = [column for column in required if column not in test.columns]
    if missing:
        raise KeyError(f"Missing columns in test DataFrame: {missing}")
    if {unit_column, "RUL"} - set(test_rul.columns):
        raise KeyError("test_rul must contain unit_id and RUL columns.")

    labels = test_rul.set_index(unit_column)["RUL"]
    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []
    lengths: list[int] = []

    ordered = test.sort_values([unit_column, cycle_column], ignore_index=True)
    for unit_id, group in ordered.groupby(unit_column, sort=False):
        if unit_id not in labels.index:
            raise ValueError(f"Missing test RUL label for unit_id={unit_id}.")

        values = group.loc[:, list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
        cycle_values = group.loc[:, cycle_column].to_numpy(dtype=np.int64, copy=True)
        end_index = len(group) - 1
        window, valid_length = _make_padded_window(
            values,
            end_index=end_index,
            sequence_length=sequence_length,
            padding_value=padding_value,
        )

        windows.append(window)
        targets.append(float(labels.loc[unit_id]))
        unit_ids.append(int(unit_id))
        end_cycles.append(int(cycle_values[end_index]))
        lengths.append(valid_length)

    return SequenceWindows(
        x=np.stack(windows).astype(np.float64, copy=False),
        y=np.asarray(targets, dtype=np.float64),
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        end_cycles=np.asarray(end_cycles, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
    )


def _preprocess_sequence_frames(
    data: CMAPSSData,
    *,
    max_rul: int,
    use_regime_features: bool,
    n_regimes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
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

    train = train.assign(RUL=clip_rul(train["RUL"], max_rul=max_rul))
    feature_columns = _feature_columns(train)
    return train, test, feature_columns, tuple(dropped)


def build_sequence_dataset(
    data: CMAPSSData,
    *,
    sequence_length: int = 30,
    stride: int = 1,
    max_rul: int = 125,
    use_regime_features: bool = False,
    n_regimes: int = 6,
    padding_value: float = 0.0,
) -> SequenceDataset:
    """Build train/test sequence tensors for a CMAPSS subset.

    Training windows are generated at cycle level. Test windows are generated
    only at each unit's final observed cycle, matching the official CMAPSS
    label semantics and avoiding inflated test sample counts.
    """
    _validate_window_args(sequence_length=sequence_length, stride=stride)
    train, test, feature_columns, dropped = _preprocess_sequence_frames(
        data,
        max_rul=max_rul,
        use_regime_features=use_regime_features,
        n_regimes=n_regimes,
    )

    train_windows = make_sequence_windows(
        train,
        feature_columns,
        sequence_length=sequence_length,
        stride=stride,
        padding_value=padding_value,
    )
    test_windows = _make_final_test_windows(
        test,
        data.test_rul,
        feature_columns,
        sequence_length=sequence_length,
        padding_value=padding_value,
    )

    return SequenceDataset(
        train_x=train_windows.x,
        train_y=train_windows.y,
        train_unit_ids=train_windows.unit_ids,
        train_end_cycles=train_windows.end_cycles,
        train_lengths=train_windows.lengths,
        test_x=test_windows.x,
        test_y=test_windows.y,
        test_unit_ids=test_windows.unit_ids,
        test_end_cycles=test_windows.end_cycles,
        test_lengths=test_windows.lengths,
        feature_columns=feature_columns,
        dropped_sensors=dropped,
        sequence_length=sequence_length,
        padding_value=padding_value,
        use_regime_features=use_regime_features,
    )
