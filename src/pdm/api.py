"""FastAPI application for serving CMAPSS RUL predictions."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pdm.serving import ModelArtifact, load_model_artifact, predict_rul

Predictor = Callable[[dict[str, float]], float]


class PredictionRequest(BaseModel):
    """Feature payload for one RUL prediction."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="Mapping from engineered feature name to numeric value.",
    )
    unit_id: int | None = Field(default=None, ge=1)
    cycle: int | None = Field(default=None, ge=1)


class PredictionResponse(BaseModel):
    """RUL prediction response."""

    model_config = ConfigDict(extra="forbid")

    rul: float
    model_version: str
    unit_id: int | None = None
    cycle: int | None = None


class HealthResponse(BaseModel):
    """Health-check response."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_version: str | None = None
    feature_count: int | None = None


def _load_optional_artifact(path: str | Path | None) -> ModelArtifact | None:
    if path is None:
        env_path = os.getenv("PDM_MODEL_PATH")
        path = env_path if env_path else None
    if path is None:
        return None
    return load_model_artifact(path)


def create_app(
    *,
    artifact_path: str | Path | None = None,
    artifact: ModelArtifact | None = None,
    predictor: Predictor | None = None,
    model_version: str = "test-double",
) -> FastAPI:
    """Create the RUL inference API.

    Args:
        artifact_path: Optional path to a persisted :class:`ModelArtifact`.
            If omitted, ``PDM_MODEL_PATH`` is used when present.
        artifact: Optional in-memory artifact. Useful for tests and notebooks.
        predictor: Optional callable accepting the raw feature mapping. This
            is intended for tests; production deployments should use an
            artifact.
        model_version: Version label returned when ``predictor`` is used.

    Returns:
        Configured FastAPI application.
    """
    if sum(item is not None for item in (artifact_path, artifact, predictor)) > 1:
        raise ValueError("Provide only one of artifact_path, artifact, or predictor.")

    loaded_artifact = artifact if artifact is not None else _load_optional_artifact(artifact_path)

    app = FastAPI(
        title="Predictive Maintenance RUL API",
        version="0.1.0",
        description="Minimal API for serving remaining-useful-life predictions.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return service and model readiness."""
        if loaded_artifact is not None:
            return HealthResponse(
                status="ok",
                model_loaded=True,
                model_version=loaded_artifact.model_version,
                feature_count=len(loaded_artifact.feature_columns),
            )
        return HealthResponse(
            status="ok",
            model_loaded=predictor is not None,
            model_version=model_version if predictor is not None else None,
            feature_count=None,
        )

    @app.post("/predict-rul", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Predict remaining useful life for one engineered feature vector."""
        try:
            if loaded_artifact is not None:
                rul = predict_rul(loaded_artifact, request.features)
                response_model_version = loaded_artifact.model_version
            elif predictor is not None:
                rul = predictor(request.features)
                response_model_version = model_version
            else:
                raise HTTPException(status_code=503, detail="No model artifact is loaded.")
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictionResponse(
            rul=rul,
            model_version=response_model_version,
            unit_id=request.unit_id,
            cycle=request.cycle,
        )

    return app


app = create_app()
"""ASGI application used by ``uvicorn pdm.api:app``."""
