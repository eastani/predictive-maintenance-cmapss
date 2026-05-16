"""Tests for CMAPSS sequence-window builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.data import CMAPSSData
from pdm.sequences import build_sequence_dataset, make_sequence_windows


def _synthetic_data() -> CMAPSSData:
    train_rows = []
    test_rows = []
    rul_rows = []
    for unit in range(1, 4):
        high_regime = unit == 3
        setting = 10.0 if high_regime else 0.0
        offset = 100.0 if high_regime else 0.0
        for cycle in range(1, 6):
            train_rows.append(
                {
                    "unit_id": unit,
                    "cycle": cycle,
                    "op_setting_1": setting,
                    "op_setting_2": setting,
                    "op_setting_3": 0.0,
                    "sensor_01": 1.0,
                    "sensor_02": offset + cycle,
                    "sensor_03": offset + cycle * 10,
                    "RUL": 5 - cycle,
                }
            )
        for cycle in range(1, 4):
            test_rows.append(
                {
                    "unit_id": unit,
                    "cycle": cycle,
                    "op_setting_1": setting,
                    "op_setting_2": setting,
                    "op_setting_3": 0.0,
                    "sensor_01": 1.0,
                    "sensor_02": offset + cycle,
                    "sensor_03": offset + cycle * 10,
                }
            )
        rul_rows.append({"unit_id": unit, "RUL": 2})

    return CMAPSSData(
        subset="FD002",
        train=pd.DataFrame(train_rows),
        test=pd.DataFrame(test_rows),
        test_rul=pd.DataFrame(rul_rows),
    )


class TestMakeSequenceWindows:
    def test_left_pads_early_cycle_windows(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1],
                "cycle": [1, 2, 3],
                "sensor_02": [10.0, 20.0, 30.0],
                "RUL": [2, 1, 0],
            }
        )

        windows = make_sequence_windows(df, ("sensor_02",), sequence_length=4)

        assert windows.x.shape == (3, 4, 1)
        assert windows.lengths.tolist() == [1, 2, 3]
        np.testing.assert_allclose(windows.x[0, :, 0], [0.0, 0.0, 0.0, 10.0])
        np.testing.assert_allclose(windows.x[2, :, 0], [0.0, 10.0, 20.0, 30.0])
        assert windows.y.tolist() == [2.0, 1.0, 0.0]

    def test_stride_controls_training_window_density(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 1, 1],
                "cycle": [1, 2, 3, 4, 5],
                "sensor_02": [1, 2, 3, 4, 5],
                "RUL": [4, 3, 2, 1, 0],
            }
        )

        windows = make_sequence_windows(df, ("sensor_02",), sequence_length=2, stride=2)

        assert windows.end_cycles.tolist() == [1, 3, 5]
        assert windows.y.tolist() == [4.0, 2.0, 0.0]

    @pytest.mark.parametrize("sequence_length,stride", [(0, 1), (3, 0), (-1, 1), (3, -1)])
    def test_rejects_non_positive_window_args(self, sequence_length: int, stride: int) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_02": [1.0],
                "RUL": [0],
            }
        )

        with pytest.raises(ValueError, match="strictly positive"):
            make_sequence_windows(
                df,
                ("sensor_02",),
                sequence_length=sequence_length,
                stride=stride,
            )


class TestBuildSequenceDataset:
    def test_builds_cycle_level_train_and_final_cycle_test_windows(self) -> None:
        dataset = build_sequence_dataset(_synthetic_data(), sequence_length=4)

        assert dataset.train_x.shape == (15, 4, 2)
        assert dataset.test_x.shape == (3, 4, 2)
        assert dataset.test_unit_ids.tolist() == [1, 2, 3]
        assert dataset.test_end_cycles.tolist() == [3, 3, 3]
        assert dataset.test_y.tolist() == [2.0, 2.0, 2.0]
        assert dataset.test_lengths.tolist() == [3, 3, 3]
        assert "sensor_01" not in dataset.feature_columns
        assert dataset.n_features == 2

    def test_can_include_regime_normalized_sequence_features(self) -> None:
        dataset = build_sequence_dataset(
            _synthetic_data(),
            sequence_length=3,
            use_regime_features=True,
            n_regimes=2,
        )

        assert dataset.use_regime_features is True
        assert "sensor_02_regime_z" in dataset.feature_columns
        assert dataset.train_x.shape[2] == len(dataset.feature_columns)
