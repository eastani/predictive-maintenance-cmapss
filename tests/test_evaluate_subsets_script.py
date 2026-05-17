"""Tests for the cross-subset evaluation CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scripts.evaluate_subsets import (
    _regime_options,
    build_run_metadata,
    summarize_operating_regime_diagnostics,
    summarize_s_score_diagnostics,
    summarize_target_cap_diagnostics,
)


class TestRegimeOptions:
    def test_auto_enables_regimes_only_for_multi_condition_subsets(self) -> None:
        assert _regime_options("FD001", "auto") == [False]
        assert _regime_options("FD002", "auto") == [True]
        assert _regime_options("FD003", "auto") == [False]
        assert _regime_options("FD004", "auto") == [True]

    def test_forced_modes(self) -> None:
        assert _regime_options("FD001", "on") == [True]
        assert _regime_options("FD004", "off") == [False]

    def test_both_mode_runs_ablation_in_stable_order(self) -> None:
        assert _regime_options("FD002", "both") == [False, True]


class TestRunMetadata:
    def test_builds_reproducibility_metadata_without_changing_result_schema(self) -> None:
        args = argparse.Namespace(
            data_dir=Path("data/raw"),
            subsets=["FD001", "FD002"],
            out=Path("reports/cross_subset_results.csv"),
            predictions_out=None,
            target_cap_diagnostics_out=None,
            regime_diagnostics_out=None,
            s_score_diagnostics_out=None,
            metadata_out=None,
            with_xgboost=True,
            max_rul=125,
            n_regimes=6,
            regime_mode="auto",
        )

        metadata = build_run_metadata(
            args,
            command=["scripts/evaluate_subsets.py", "--with-xgboost"],
            output_files={
                "results": Path("reports/cross_subset_results.csv"),
                "metadata": Path("reports/cross_subset_results_metadata.json"),
            },
            started_at="2026-05-17T00:00:00+00:00",
            finished_at="2026-05-17T00:01:00+00:00",
            n_result_rows=4,
            n_prediction_rows=520,
            git_commit="abc123",
        )

        assert metadata["schema_version"] == 1
        assert metadata["git"] == {"commit": "abc123"}
        assert metadata["inputs"] == {"data_dir": "data/raw", "subsets": ["FD001", "FD002"]}
        assert metadata["configuration"]["max_rul"] == 125
        assert "out" not in metadata["configuration"]
        assert "metadata_out" not in metadata["configuration"]
        assert metadata["target_convention"]["headline_metrics"] == "raw_test_rul"
        assert [model["name"] for model in metadata["models"]] == ["ridge", "xgboost"]
        assert metadata["outputs"]["metadata"] == "reports/cross_subset_results_metadata.json"
        assert metadata["counts"] == {"result_rows": 4, "prediction_rows": 520}


class TestSummarizeTargetCapDiagnostics:
    def test_aggregates_tabular_predictions_without_seed(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
                    "use_regime_features": False,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 90.0,
                    "error": -10.0,
                    "abs_error": 10.0,
                    "late": False,
                },
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
                    "use_regime_features": False,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 180.0,
                    "y_pred": 120.0,
                    "error": -60.0,
                    "abs_error": 60.0,
                    "late": False,
                },
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
                    "use_regime_features": True,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 95.0,
                    "error": -5.0,
                    "abs_error": 5.0,
                    "late": False,
                },
            ]
        )

        diagnostics = summarize_target_cap_diagnostics(predictions, max_rul=125)

        assert diagnostics["target_convention"].to_list() == ["raw", "cap_125", "raw", "cap_125"]
        assert diagnostics["use_regime_features"].to_list() == [False, False, True, True]
        assert diagnostics["seed"].isna().all()


class TestSummarizeOperatingRegimeDiagnostics:
    def test_aggregates_by_operating_regime(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD002",
                    "model": "xgboost",
                    "seed": None,
                    "use_regime_features": True,
                    "operating_regime": 0,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 90.0,
                    "error": -10.0,
                    "abs_error": 10.0,
                    "late": False,
                },
                {
                    "subset": "FD002",
                    "model": "xgboost",
                    "seed": None,
                    "use_regime_features": True,
                    "operating_regime": 1,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 180.0,
                    "y_pred": 120.0,
                    "error": -60.0,
                    "abs_error": 60.0,
                    "late": False,
                },
            ]
        )

        diagnostics = summarize_operating_regime_diagnostics(predictions)

        assert diagnostics["segment"].to_list() == ["op_regime_0", "op_regime_1"]
        assert diagnostics["use_regime_features"].to_list() == [True, True]
        assert diagnostics["seed"].isna().all()


class TestSummarizeSScoreDiagnostics:
    def test_aggregates_s_score_contributions_without_seed(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
                    "use_regime_features": True,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 90.0,
                    "error": -10.0,
                    "abs_error": 10.0,
                    "late": False,
                },
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
                    "use_regime_features": True,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 100.0,
                    "y_pred": 110.0,
                    "error": 10.0,
                    "abs_error": 10.0,
                    "late": True,
                },
            ]
        )

        diagnostics = summarize_s_score_diagnostics(predictions)

        assert diagnostics["segment"].to_list() == ["all", "early", "late"]
        assert diagnostics["use_regime_features"].to_list() == [True, True, True]
        assert diagnostics["seed"].isna().all()
