"""Tests for the modelling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pdm.models import (
    build_baseline_regressor,
    build_gradient_boosted_regressor,
    make_xy,
    rmse,
    s_score,
)


# --------------------------------------------------------------------------- #
# make_xy
# --------------------------------------------------------------------------- #
class TestMakeXY:
    def test_returns_arrays_with_expected_shapes(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "RUL": [10, 20, 30]})
        X, y = make_xy(df, feature_columns=["a", "b"])
        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_values_match_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "RUL": [10, 20]})
        X, y = make_xy(df, feature_columns=["a", "b"])
        np.testing.assert_array_equal(X, np.array([[1.0, 3.0], [2.0, 4.0]]))
        np.testing.assert_array_equal(y, np.array([10.0, 20.0]))

    def test_arrays_are_independent_copies(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "RUL": [10, 20]})
        X, y = make_xy(df, feature_columns=["a"])
        X[0, 0] = 99.0
        y[0] = 99.0
        # Mutations must not propagate back to the source DataFrame.
        assert df.loc[0, "a"] == 1.0
        assert df.loc[0, "RUL"] == 10

    def test_custom_target_column(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "lifetime": [50, 100]})
        _, y = make_xy(df, feature_columns=["a"], target_column="lifetime")
        np.testing.assert_array_equal(y, np.array([50.0, 100.0]))

    def test_empty_feature_columns_raises(self) -> None:
        df = pd.DataFrame({"RUL": [1, 2]})
        with pytest.raises(ValueError, match="must not be empty"):
            make_xy(df, feature_columns=[])

    def test_missing_columns_raise(self) -> None:
        df = pd.DataFrame({"a": [1.0], "RUL": [10]})
        with pytest.raises(KeyError, match="Missing columns"):
            make_xy(df, feature_columns=["a", "b"])
        with pytest.raises(KeyError, match="Missing columns"):
            make_xy(df, feature_columns=["a"], target_column="lifetime")


# --------------------------------------------------------------------------- #
# build_baseline_regressor
# --------------------------------------------------------------------------- #
class TestBuildBaselineRegressor:
    def test_returns_pipeline(self) -> None:
        model = build_baseline_regressor()
        assert isinstance(model, Pipeline)
        assert [name for name, _ in model.steps] == ["scaler", "ridge"]

    def test_can_fit_and_predict(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        # Linear ground truth so ridge should fit it well.
        true_w = np.array([2.0, -1.0, 0.5, 3.0])
        y = X @ true_w + rng.normal(scale=0.1, size=50)
        model = build_baseline_regressor(alpha=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)
        assert rmse(y, preds) < 0.5

    @pytest.mark.parametrize("bad_alpha", [-0.1, -1.0, -1e-9])
    def test_negative_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            build_baseline_regressor(alpha=bad_alpha)


# --------------------------------------------------------------------------- #
# build_gradient_boosted_regressor
# --------------------------------------------------------------------------- #
xgboost = pytest.importorskip("xgboost", reason="optional 'boost' extra not installed")


class TestBuildGradientBoostedRegressor:
    def test_returns_pipeline_with_expected_steps(self) -> None:
        model = build_gradient_boosted_regressor(n_estimators=10, max_depth=3)
        assert isinstance(model, Pipeline)
        assert [name for name, _ in model.steps] == ["scaler", "xgboost"]

    def test_can_fit_and_predict(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 5))
        true_w = np.array([2.0, -1.0, 0.5, 3.0, 0.1])
        y = X @ true_w + rng.normal(scale=0.1, size=120)

        model = build_gradient_boosted_regressor(n_estimators=80, max_depth=4, learning_rate=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (120,)
        # XGBoost should easily under-fit a near-linear target with this size.
        assert rmse(y, preds) < 0.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_estimators": 0},
            {"n_estimators": -1},
            {"max_depth": 0},
            {"max_depth": -3},
            {"learning_rate": 0.0},
            {"learning_rate": -0.01},
            {"subsample": 0.0},
            {"subsample": -0.1},
            {"subsample": 1.5},
            {"colsample_bytree": 0.0},
            {"colsample_bytree": 1.5},
        ],
    )
    def test_invalid_hyperparameters_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            build_gradient_boosted_regressor(**kwargs)


# --------------------------------------------------------------------------- #
# rmse
# --------------------------------------------------------------------------- #
class TestRMSE:
    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 4.0, 4.0])
        # MSE = (4 + 16 + 16) / 3 = 12 → sqrt(12) ≈ 3.464
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.0))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# s_score
# --------------------------------------------------------------------------- #
class TestSScore:
    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        assert s_score(y, y) == 0.0

    def test_late_prediction_penalised_more_than_early(self) -> None:
        """A 10-cycle late prediction must cost strictly more than a 10-cycle
        early one — that asymmetry is the point of the metric."""
        y_true = np.array([50.0])
        early = s_score(y_true, np.array([40.0]))
        late = s_score(y_true, np.array([60.0]))
        assert late > early
        # Sanity: with the canonical constants exp(10/10) - 1 ≈ 1.718
        # vs exp(10/13) - 1 ≈ 1.146.
        assert late == pytest.approx(np.exp(1.0) - 1.0, rel=1e-6)
        assert early == pytest.approx(np.exp(10.0 / 13.0) - 1.0, rel=1e-6)

    def test_score_is_additive_across_samples(self) -> None:
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        # Compute each sample independently and sum.
        s1 = s_score(np.array([10.0]), np.array([12.0]))
        s2 = s_score(np.array([20.0]), np.array([18.0]))
        assert s_score(y_true, y_pred) == pytest.approx(s1 + s2, rel=1e-9)

    def test_zero_error_at_boundary(self) -> None:
        """When y_pred == y_true exactly, the late branch contributes zero."""
        s = s_score(np.array([5.0]), np.array([5.0]))
        assert s == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"early_decay": 0.0},
            {"early_decay": -1.0},
            {"late_decay": 0.0},
            {"late_decay": -1.0},
        ],
    )
    def test_invalid_decay_raises(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            s_score(np.array([1.0]), np.array([1.0]), **kwargs)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            s_score(np.array([1.0]), np.array([1.0, 2.0]))
