"""Tests for prediction-error diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pdm.diagnostics import (
    prediction_error_breakdown,
    prediction_error_by_group,
    prediction_error_by_rul_band,
    prediction_error_with_target_caps,
    prediction_rows,
    prediction_s_score_contributions,
)


class TestPredictionRows:
    def test_builds_signed_error_columns(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1, 2]),
            end_cycles=np.array([31, 47]),
            y_true=np.array([10.0, 20.0]),
            y_pred=np.array([12.0, 15.0]),
        )

        assert rows["error"].to_list() == [2.0, -5.0]
        assert rows["abs_error"].to_list() == [2.0, 5.0]
        assert rows["late"].to_list() == [True, False]

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="y_pred"):
            prediction_rows(
                subset="FD001",
                model="lstm",
                seed=None,
                unit_ids=np.array([1, 2]),
                end_cycles=np.array([31, 47]),
                y_true=np.array([10.0, 20.0]),
                y_pred=np.array([12.0]),
            )


class TestPredictionErrorBreakdown:
    def test_summarizes_all_early_and_late_segments(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1, 2, 3]),
            end_cycles=np.array([31, 47, 52]),
            y_true=np.array([10.0, 20.0, 30.0]),
            y_pred=np.array([12.0, 15.0, 30.0]),
        )

        summary = prediction_error_breakdown(rows)

        assert summary["segment"].to_list() == ["all", "early_or_exact", "late"]
        assert summary.loc[summary["segment"] == "all", "n"].iloc[0] == 3
        assert summary.loc[summary["segment"] == "late", "mean_error"].iloc[0] == 2.0

    def test_rejects_empty_predictions(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1]),
            end_cycles=np.array([31]),
            y_true=np.array([10.0]),
            y_pred=np.array([12.0]),
        ).iloc[0:0]

        with pytest.raises(ValueError, match="must not be empty"):
            prediction_error_breakdown(rows)


class TestPredictionErrorByRulBand:
    def test_summarizes_true_rul_bands(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1, 2, 3, 4]),
            end_cycles=np.array([31, 47, 52, 71]),
            y_true=np.array([25.0, 75.0, 110.0, 180.0]),
            y_pred=np.array([30.0, 60.0, 100.0, 120.0]),
        )

        summary = prediction_error_by_rul_band(rows)

        assert summary["segment"].to_list() == ["rul_0-50", "rul_50-100", "rul_100-125", "rul_125+"]
        assert summary["n"].to_list() == [1, 1, 1, 1]
        assert summary.loc[summary["segment"] == "rul_125+", "mean_error"].iloc[0] == -60.0

    def test_rejects_label_count_mismatch(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1]),
            end_cycles=np.array([31]),
            y_true=np.array([25.0]),
            y_pred=np.array([30.0]),
        )

        with pytest.raises(ValueError, match="labels"):
            prediction_error_by_rul_band(rows, bins=(0.0, 50.0), labels=("low", "extra"))


class TestPredictionErrorByGroup:
    def test_summarizes_categorical_groups(self) -> None:
        rows = prediction_rows(
            subset="FD002",
            model="xgboost",
            seed=None,
            unit_ids=np.array([1, 2, 3]),
            end_cycles=np.array([31, 47, 52]),
            y_true=np.array([10.0, 20.0, 30.0]),
            y_pred=np.array([12.0, 15.0, 27.0]),
        )
        rows["operating_regime"] = np.array([0, 1, 1])

        summary = prediction_error_by_group(
            rows,
            group_column="operating_regime",
            segment_prefix="op_regime",
        )

        assert summary["segment"].to_list() == ["op_regime_0", "op_regime_1"]
        assert summary["n"].to_list() == [1, 2]
        assert summary.loc[summary["segment"] == "op_regime_1", "mean_error"].iloc[0] == -4.0

    def test_rejects_missing_group_column(self) -> None:
        rows = prediction_rows(
            subset="FD002",
            model="xgboost",
            seed=None,
            unit_ids=np.array([1]),
            end_cycles=np.array([31]),
            y_true=np.array([10.0]),
            y_pred=np.array([12.0]),
        )

        with pytest.raises(KeyError, match="operating_regime"):
            prediction_error_by_group(rows, group_column="operating_regime")


class TestPredictionErrorWithTargetCaps:
    def test_compares_raw_and_capped_conventions(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1, 2]),
            end_cycles=np.array([31, 47]),
            y_true=np.array([100.0, 180.0]),
            y_pred=np.array([90.0, 120.0]),
        )

        summary = prediction_error_with_target_caps(rows, caps=(None, 125.0))

        assert summary["target_convention"].to_list() == ["raw", "cap_125"]
        assert summary.loc[0, "mean_error"] == -35.0
        assert summary.loc[1, "mean_error"] == -7.5

    def test_rejects_empty_predictions(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="lstm",
            seed=42,
            unit_ids=np.array([1]),
            end_cycles=np.array([31]),
            y_true=np.array([100.0]),
            y_pred=np.array([90.0]),
        ).iloc[0:0]

        with pytest.raises(ValueError, match="must not be empty"):
            prediction_error_with_target_caps(rows)


class TestPredictionSScoreContributions:
    def test_breaks_down_early_and_late_contributions(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="ridge",
            seed=None,
            unit_ids=np.array([1, 2, 3]),
            end_cycles=np.array([31, 47, 52]),
            y_true=np.array([10.0, 20.0, 30.0]),
            y_pred=np.array([5.0, 30.0, 30.0]),
        )

        summary = prediction_s_score_contributions(rows)

        assert summary["segment"].to_list() == ["all", "early", "late"]
        assert summary["n"].to_list() == [3, 1, 2]
        assert summary.loc[summary["segment"] == "all", "s_score"].iloc[0] == pytest.approx(
            summary.loc[summary["segment"] == "early", "s_score"].iloc[0]
            + summary.loc[summary["segment"] == "late", "s_score"].iloc[0]
        )
        assert summary.loc[summary["segment"] == "late", "s_score_share"].iloc[0] > 0.5

    def test_rejects_empty_predictions(self) -> None:
        rows = prediction_rows(
            subset="FD001",
            model="ridge",
            seed=None,
            unit_ids=np.array([1]),
            end_cycles=np.array([31]),
            y_true=np.array([10.0]),
            y_pred=np.array([10.0]),
        ).iloc[0:0]

        with pytest.raises(ValueError, match="must not be empty"):
            prediction_s_score_contributions(rows)
