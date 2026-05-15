"""Tests for model serving helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from pdm.serving import (
    ModelArtifact,
    load_model_artifact,
    predict_rul,
    save_model_artifact,
    vectorize_features,
)


def _fit_model() -> Ridge:
    x = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([100.0, 80.0, 60.0, 40.0])
    model = Ridge(alpha=0.0)
    model.fit(x, y)
    return model


class TestModelArtifact:
    def test_rejects_empty_feature_columns(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ModelArtifact(model=_fit_model(), feature_columns=())

    def test_rejects_duplicate_feature_columns(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            ModelArtifact(model=_fit_model(), feature_columns=("a", "a"))

    def test_round_trip_joblib_artifact(self, tmp_path: Path) -> None:
        artifact = ModelArtifact(
            model=_fit_model(),
            feature_columns=("sensor_02_mean_5", "sensor_03_mean_5"),
            model_version="unit-test",
        )
        path = tmp_path / "model.joblib"

        save_model_artifact(artifact, path)
        loaded = load_model_artifact(path)

        assert loaded.model_version == "unit-test"
        assert loaded.feature_columns == artifact.feature_columns
        assert predict_rul(
            loaded, {"sensor_02_mean_5": 1.0, "sensor_03_mean_5": 2.0}
        ) == pytest.approx(80.0)


class TestVectorizeFeatures:
    def test_orders_features_by_model_columns(self) -> None:
        matrix = vectorize_features({"b": 2.0, "a": 1.0}, ("a", "b"))
        np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0]]))

    def test_missing_required_feature_raises(self) -> None:
        with pytest.raises(KeyError, match="Missing required"):
            vectorize_features({"a": 1.0}, ("a", "b"))
