"""Model artifact helpers for serving RUL predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np


class Regressor(Protocol):
    """Minimal prediction protocol used by persisted scikit-learn models."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for a two-dimensional feature matrix."""


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable bundle used by the inference API.

    Attributes:
        model: A fitted estimator exposing ``predict``.
        feature_columns: Ordered feature names expected by the estimator.
        model_version: Human-readable model version or training run label.
        metadata: Optional free-form details such as subset, metric values,
            preprocessing settings, or training timestamp.
    """

    model: Regressor
    feature_columns: tuple[str, ...]
    model_version: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate feature metadata."""
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique.")


def save_model_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    """Persist a model artifact with joblib."""
    joblib.dump(artifact, Path(path))


def load_model_artifact(path: str | Path) -> ModelArtifact:
    """Load and validate a model artifact from disk."""
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, ModelArtifact):
        raise TypeError(f"Expected ModelArtifact, got {type(loaded).__name__}.")
    return loaded


def vectorize_features(
    features: dict[str, float],
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """Convert a feature mapping into a single-row model matrix."""
    missing = [name for name in feature_columns if name not in features]
    if missing:
        raise KeyError(f"Missing required feature values: {missing}")

    values = [features[name] for name in feature_columns]
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def predict_rul(artifact: ModelArtifact, features: dict[str, float]) -> float:
    """Predict remaining useful life from an ordered feature mapping."""
    x = vectorize_features(features, artifact.feature_columns)
    prediction = artifact.model.predict(x)
    return float(np.asarray(prediction, dtype=np.float64).reshape(-1)[0])
