"""Tests for the reusable CMAPSS evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from pdm.data import CMAPSSData
from pdm.evaluation import build_modelling_dataset, evaluate_regressor


def _synthetic_data() -> CMAPSSData:
    train_rows = []
    test_rows = []
    rul_rows = []
    for unit in range(1, 5):
        high_regime = unit > 2
        setting = 10.0 if high_regime else 0.0
        offset = 100.0 if high_regime else 0.0
        for cycle in range(1, 7):
            train_rows.append(
                {
                    "unit_id": unit,
                    "cycle": cycle,
                    "op_setting_1": setting,
                    "op_setting_2": setting,
                    "op_setting_3": 0.0,
                    "sensor_01": 1.0,
                    "sensor_02": offset + cycle,
                    "sensor_03": offset + cycle * 2,
                    "RUL": 6 - cycle,
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
                    "sensor_03": offset + cycle * 2,
                }
            )
        rul_rows.append({"unit_id": unit, "RUL": 3})

    return CMAPSSData(
        subset="FD002",
        train=pd.DataFrame(train_rows),
        test=pd.DataFrame(test_rows),
        test_rul=pd.DataFrame(rul_rows),
    )


class TestBuildModellingDataset:
    def test_aligns_one_test_row_per_unit(self) -> None:
        modelling = build_modelling_dataset(_synthetic_data(), windows=(2,))

        assert len(modelling.test_final) == 4
        assert modelling.test_final["cycle"].to_list() == [3, 3, 3, 3]
        assert modelling.test_final["RUL"].to_list() == [3, 3, 3, 3]
        assert "sensor_01" not in modelling.feature_columns
        assert "sensor_02_mean_2" in modelling.feature_columns

    def test_can_add_regime_features(self) -> None:
        modelling = build_modelling_dataset(
            _synthetic_data(),
            windows=(2,),
            use_regime_features=True,
            n_regimes=2,
        )

        assert modelling.use_regime_features is True
        assert "op_regime" in modelling.train.columns
        assert "sensor_02_regime_z" in modelling.feature_columns
        assert "sensor_02_regime_z_mean_2" in modelling.feature_columns


class TestEvaluateRegressor:
    def test_returns_headline_metrics(self) -> None:
        result = evaluate_regressor(
            _synthetic_data(),
            model_name="dummy-mean",
            model_factory=lambda: DummyRegressor(strategy="mean"),
            windows=(2,),
        )

        assert result.subset == "FD002"
        assert result.model_name == "dummy-mean"
        assert result.n_train_samples == 24
        assert result.n_test_units == 4
        assert result.n_features > 0
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.s_score)
        assert result.as_dict()["model"] == "dummy-mean"
