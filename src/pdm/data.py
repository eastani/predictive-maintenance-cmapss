"""CMAPSS dataset loader.

The four CMAPSS sub-datasets share a common, fixed schema:

* ``train_FD00X.txt`` and ``test_FD00X.txt`` — whitespace-separated, 26
  columns: unit id, operating cycle, three operational settings, then 21
  sensor channels.
* ``RUL_FD00X.txt`` — one row per test unit, a single integer giving the
  remaining useful life at the *last* recorded cycle of the test trajectory.

This module hides those mechanics behind a single :func:`load_subset` entry
point that returns a fully labelled, schema-validated :class:`CMAPSSData`
container ready for feature engineering.

References:
    A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation
    Modeling for Aircraft Engine Run-to-Failure Simulation*, International
    Conference on Prognostics and Health Management, 2008.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

SubsetName = Literal["FD001", "FD002", "FD003", "FD004"]
"""Identifier for one of the four CMAPSS sub-datasets."""

OPERATIONAL_SETTING_COLUMNS: tuple[str, ...] = (
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
)

SENSOR_COLUMNS: tuple[str, ...] = tuple(f"sensor_{i:02d}" for i in range(1, 22))

ALL_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "cycle",
    *OPERATIONAL_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
)


@dataclass(frozen=True)
class CMAPSSData:
    """A loaded CMAPSS sub-dataset.

    Attributes:
        subset: Which sub-dataset (``"FD001"`` … ``"FD004"``) was loaded.
        train: Training trajectories with a derived ``RUL`` column.
            One row per (unit_id, cycle); units run until failure so the
            final cycle of each unit has ``RUL == 0``.
        test: Test trajectories, truncated before failure. No ``RUL``
            column — labels live in :attr:`test_rul`.
        test_rul: One row per test ``unit_id`` giving the remaining useful
            life at the *last* observed cycle of that unit.
    """

    subset: SubsetName
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame

    @property
    def n_train_units(self) -> int:
        """Number of distinct engines in the training set."""
        return int(self.train["unit_id"].nunique())

    @property
    def n_test_units(self) -> int:
        """Number of distinct engines in the test set."""
        return int(self.test["unit_id"].nunique())


class CMAPSSDataError(ValueError):
    """Raised when CMAPSS files are missing, malformed, or fail validation."""


def _read_trajectory_file(path: Path) -> pd.DataFrame:
    """Read a ``train_FD00X.txt`` / ``test_FD00X.txt`` file.

    The official files use repeated whitespace as a delimiter and contain
    two trailing whitespace columns. We treat those as a parser quirk and
    drop them after parsing.
    """
    if not path.is_file():
        raise CMAPSSDataError(f"CMAPSS trajectory file not found: {path}")

    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # The released files occasionally end each line with two extra blanks
    # which pandas parses as all-NaN trailing columns. Drop them.
    raw = raw.dropna(axis="columns", how="all")

    if raw.shape[1] != len(ALL_COLUMNS):
        raise CMAPSSDataError(
            f"Unexpected column count in {path.name}: "
            f"expected {len(ALL_COLUMNS)}, got {raw.shape[1]}"
        )

    raw.columns = list(ALL_COLUMNS)
    raw = raw.astype({"unit_id": "int64", "cycle": "int64"})
    return raw.sort_values(["unit_id", "cycle"], ignore_index=True)


def _read_rul_file(path: Path) -> pd.DataFrame:
    """Read a ``RUL_FD00X.txt`` file — one integer per line, in unit-id order."""
    if not path.is_file():
        raise CMAPSSDataError(f"CMAPSS RUL file not found: {path}")

    rul_values = pd.read_csv(path, header=None, names=["RUL"]).astype({"RUL": "int64"})
    rul_values.insert(0, "unit_id", range(1, len(rul_values) + 1))
    return rul_values


def _attach_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Add a ``RUL`` column to the training trajectories.

    For training units, the trajectory ends at failure, so for each row the
    remaining useful life is ``max_cycle_for_unit - current_cycle``.
    """
    final_cycle = train.groupby("unit_id")["cycle"].transform("max")
    return train.assign(RUL=(final_cycle - train["cycle"]).astype("int64"))


def load_subset(subset: SubsetName, data_dir: str | Path) -> CMAPSSData:
    """Load one CMAPSS sub-dataset from a directory of raw files.

    Args:
        subset: Which sub-dataset to load (``"FD001"`` … ``"FD004"``).
        data_dir: Directory containing the raw CMAPSS files for this
            subset (``train_<subset>.txt``, ``test_<subset>.txt``,
            ``RUL_<subset>.txt``).

    Returns:
        Fully labelled :class:`CMAPSSData` with derived training RUL.

    Raises:
        CMAPSSDataError: If any expected file is missing or malformed.
    """
    if subset not in ("FD001", "FD002", "FD003", "FD004"):
        raise CMAPSSDataError(f"Unknown CMAPSS subset: {subset!r}")

    root = Path(data_dir)
    train = _attach_train_rul(_read_trajectory_file(root / f"train_{subset}.txt"))
    test = _read_trajectory_file(root / f"test_{subset}.txt")
    rul = _read_rul_file(root / f"RUL_{subset}.txt")

    expected_test_units = set(test["unit_id"].unique())
    rul_units = set(rul["unit_id"].unique())
    if expected_test_units != rul_units:
        missing = expected_test_units.symmetric_difference(rul_units)
        raise CMAPSSDataError(
            f"Mismatch between test units and RUL labels for {subset}: "
            f"differing units = {sorted(missing)}"
        )

    return CMAPSSData(subset=subset, train=train, test=test, test_rul=rul)
