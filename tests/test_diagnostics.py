"""Tests for prediction-error diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pdm.diagnostics import prediction_error_breakdown, prediction_rows


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
