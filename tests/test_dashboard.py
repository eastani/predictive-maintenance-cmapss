"""Tests for benchmark dashboard data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdm.dashboard import load_benchmark_results, model_delta_summary


class TestLoadBenchmarkResults:
    def test_loads_default_measured_results(self) -> None:
        results = load_benchmark_results()

        assert set(results["subset"]) == {"FD001", "FD002", "FD003", "FD004"}
        assert set(results["model"]) == {"ridge", "xgboost"}
        assert len(results) == 8

    def test_loads_csv_with_required_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        pd.DataFrame(
            [
                {
                    "subset": "FD001",
                    "model": "ridge",
                    "rmse": 1.0,
                    "s_score": 2.0,
                    "n_train_samples": 3,
                    "n_test_units": 4,
                    "n_features": 5,
                    "use_regime_features": False,
                }
            ]
        ).to_csv(path, index=False)

        results = load_benchmark_results(path)

        assert results.loc[0, "subset"] == "FD001"
        assert results.loc[0, "rmse"] == 1.0

    def test_rejects_csv_missing_required_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"subset": "FD001"}]).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing columns"):
            load_benchmark_results(path)


class TestModelDeltaSummary:
    def test_computes_xgboost_minus_ridge_deltas(self) -> None:
        results = load_benchmark_results()
        deltas = model_delta_summary(results)

        fd002 = deltas.set_index("subset").loc["FD002"]
        assert fd002["rmse_delta"] < 0
        assert fd002["s_score_delta"] < 0
        assert bool(fd002["xgboost_better_rmse"]) is True
        assert bool(fd002["xgboost_better_s_score"]) is True

    def test_skips_subsets_without_both_models(self) -> None:
        results = load_benchmark_results()
        only_ridge = results[results["model"] == "ridge"]

        assert model_delta_summary(only_ridge).empty
