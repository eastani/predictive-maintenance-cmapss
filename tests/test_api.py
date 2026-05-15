"""Tests for the FastAPI inference service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pdm.api import create_app


class TestInferenceAPI:
    def test_health_without_model_reports_not_ready_for_prediction(self) -> None:
        client = TestClient(create_app())
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "model_loaded": False,
            "model_version": None,
            "feature_count": None,
        }

    def test_predict_with_injected_predictor(self) -> None:
        def predictor(features: dict[str, float]) -> float:
            return 120.0 - features["sensor_02_mean_5"] * 2.0

        client = TestClient(create_app(predictor=predictor, model_version="unit-test"))
        response = client.post(
            "/predict-rul",
            json={
                "unit_id": 7,
                "cycle": 42,
                "features": {"sensor_02_mean_5": 10.0},
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "rul": 100.0,
            "model_version": "unit-test",
            "unit_id": 7,
            "cycle": 42,
        }

    def test_predict_without_model_returns_503(self) -> None:
        client = TestClient(create_app())
        response = client.post("/predict-rul", json={"features": {"x": 1.0}})

        assert response.status_code == 503
        assert response.json()["detail"] == "No model artifact is loaded."

    def test_create_app_rejects_ambiguous_model_sources(self) -> None:
        with pytest.raises(ValueError, match="Provide only one"):
            create_app(artifact_path="model.joblib", predictor=lambda _: 1.0)
