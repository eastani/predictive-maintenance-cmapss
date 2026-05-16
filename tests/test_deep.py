"""Tests for optional deep-learning helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pdm.deep import (
    LSTMTrainingConfig,
    build_lstm_regressor,
    is_torch_available,
    left_padded_to_right_padded,
    train_lstm_regressor,
)


class TestLeftPaddedToRightPadded:
    def test_moves_valid_suffixes_to_prefixes(self) -> None:
        x = np.array(
            [
                [[0.0], [0.0], [10.0], [20.0]],
                [[0.0], [1.0], [2.0], [3.0]],
            ],
            dtype=np.float32,
        )
        lengths = np.array([2, 3])

        out = left_padded_to_right_padded(x, lengths)

        np.testing.assert_allclose(out[0, :, 0], [10.0, 20.0, 0.0, 0.0])
        np.testing.assert_allclose(out[1, :, 0], [1.0, 2.0, 3.0, 0.0])

    def test_rejects_lengths_longer_than_sequence(self) -> None:
        x = np.zeros((1, 3, 2), dtype=np.float32)
        lengths = np.array([4])

        with pytest.raises(ValueError, match="must not exceed"):
            left_padded_to_right_padded(x, lengths)


class TestBuildLSTMRegressor:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_features": 0},
            {"n_features": 2, "hidden_size": 0},
            {"n_features": 2, "num_layers": 0},
            {"n_features": 2, "dropout": -0.1},
            {"n_features": 2, "dropout": 1.0},
        ],
    )
    def test_invalid_model_args_raise_before_importing_torch(
        self, kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError):
            build_lstm_regressor(**kwargs)

    def test_valid_model_requires_deep_extra_when_torch_missing(self) -> None:
        if is_torch_available():
            pytest.skip("torch is installed in this environment")

        with pytest.raises(ImportError, match="uv sync --extra deep"):
            build_lstm_regressor(n_features=2)


class TestTrainLSTMRegressor:
    def test_rejects_bad_training_config_before_importing_torch(self) -> None:
        x = np.zeros((2, 3, 1), dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        lengths = np.array([1, 2])
        config = LSTMTrainingConfig(epochs=0)

        with pytest.raises(ValueError, match="epochs"):
            train_lstm_regressor(x, y, lengths, config=config)

    def test_valid_training_requires_deep_extra_when_torch_missing(self) -> None:
        if is_torch_available():
            pytest.skip("torch is installed in this environment")

        x = np.zeros((2, 3, 1), dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        lengths = np.array([1, 2])

        with pytest.raises(ImportError, match="uv sync --extra deep"):
            train_lstm_regressor(x, y, lengths, config=LSTMTrainingConfig(epochs=1))
