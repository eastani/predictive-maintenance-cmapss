"""Tests for the CMAPSS loader.

These tests run against synthetic fixtures generated at runtime so they
work without the real NASA dataset available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.data import (
    ALL_COLUMNS,
    SENSOR_COLUMNS,
    CMAPSSDataError,
    load_subset,
)


def _write_synthetic_subset(
    data_dir: Path,
    subset: str = "FD001",
    n_units: int = 3,
    cycles_per_unit: int = 20,
    test_truncate_at: int = 12,
) -> None:
    """Create a synthetic CMAPSS-format set of files inside ``data_dir``."""
    rng = np.random.default_rng(seed=42)

    train_rows = []
    test_rows = []
    rul_rows = []

    n_settings = len(("op_setting_1", "op_setting_2", "op_setting_3"))
    n_sensors = len(SENSOR_COLUMNS)

    def make_row(unit: int, cycle: int) -> list[float]:
        settings = rng.normal(size=n_settings).round(4).tolist()
        sensors = rng.normal(size=n_sensors).round(4).tolist()
        return [float(unit), float(cycle), *settings, *sensors]

    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            train_rows.append(make_row(unit, cycle))
        for cycle in range(1, test_truncate_at + 1):
            test_rows.append(make_row(unit, cycle))
        rul_rows.append(cycles_per_unit - test_truncate_at)

    pd.DataFrame(train_rows, columns=list(ALL_COLUMNS)).to_csv(
        data_dir / f"train_{subset}.txt", sep=" ", header=False, index=False
    )
    pd.DataFrame(test_rows, columns=list(ALL_COLUMNS)).to_csv(
        data_dir / f"test_{subset}.txt", sep=" ", header=False, index=False
    )
    pd.DataFrame(rul_rows, columns=["RUL"]).to_csv(
        data_dir / f"RUL_{subset}.txt", header=False, index=False
    )


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    _write_synthetic_subset(tmp_path)
    return tmp_path


class TestLoadSubset:
    def test_returns_three_dataframes(self, synthetic_dir: Path) -> None:
        data = load_subset("FD001", synthetic_dir)
        assert isinstance(data.train, pd.DataFrame)
        assert isinstance(data.test, pd.DataFrame)
        assert isinstance(data.test_rul, pd.DataFrame)

    def test_train_columns_match_schema(self, synthetic_dir: Path) -> None:
        data = load_subset("FD001", synthetic_dir)
        assert list(data.train.columns) == [*ALL_COLUMNS, "RUL"]

    def test_train_has_strictly_decreasing_rul(self, synthetic_dir: Path) -> None:
        """Within each training unit, RUL must decrease cycle by cycle to zero."""
        data = load_subset("FD001", synthetic_dir)
        for _, group in data.train.groupby("unit_id"):
            ruls = group.sort_values("cycle")["RUL"].to_list()
            assert ruls == sorted(ruls, reverse=True)
            assert ruls[-1] == 0

    def test_unit_counts(self, synthetic_dir: Path) -> None:
        data = load_subset("FD001", synthetic_dir)
        assert data.n_train_units == 3
        assert data.n_test_units == 3

    def test_test_rul_aligned_to_test_units(self, synthetic_dir: Path) -> None:
        data = load_subset("FD001", synthetic_dir)
        assert set(data.test_rul["unit_id"]) == set(data.test["unit_id"])

    @pytest.mark.parametrize("subset", ["FD000", "fd001", "FD005", ""])
    def test_invalid_subset_raises(self, synthetic_dir: Path, subset: str) -> None:
        with pytest.raises(CMAPSSDataError, match="Unknown"):
            load_subset(subset, synthetic_dir)  # type: ignore[arg-type]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CMAPSSDataError, match="not found"):
            load_subset("FD001", tmp_path)

    def test_malformed_file_raises(self, tmp_path: Path, synthetic_dir: Path) -> None:
        # Truncate a column to simulate a malformed file.
        bad = (synthetic_dir / "train_FD001.txt").read_text().splitlines()
        bad = [" ".join(line.split()[:5]) for line in bad]
        (synthetic_dir / "train_FD001.txt").write_text("\n".join(bad))
        with pytest.raises(CMAPSSDataError, match="column count"):
            load_subset("FD001", synthetic_dir)
