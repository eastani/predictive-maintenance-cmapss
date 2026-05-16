"""Tests for optional deep-learning helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pdm.deep import (
    LSTMTrainingConfig,
    build_lstm_regressor,
    is_torch_available,
    left_padded_to_right_padded,
    standardize_left_padded_sequences,
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


class TestStandardizeLeftPaddedSequences:
    def test_fits_scaler_on_valid_timesteps_only(self) -> None:
        x = np.array(
            [
                [[0.0], [0.0], [10.0], [20.0]],
                [[0.0], [30.0], [40.0], [50.0]],
            ],
            dtype=np.float32,
        )
        lengths = np.array([2, 3])

        out, means, stds = standardize_left_padded_sequences(x, lengths)

        np.testing.assert_allclose(means, np.array([30.0], dtype=np.float32))
        np.testing.assert_allclose(stds, np.array([np.sqrt(200.0)], dtype=np.float32))
        np.testing.assert_allclose(out[0, :2, 0], [0.0, 0.0])
        np.testing.assert_allclose(out[1, :1, 0], [0.0])
        assert out[0, 2, 0] < 0.0
        assert out[1, 3, 0] > 0.0

    def test_can_apply_existing_scaler(self) -> None:
        x = np.array([[[0.0], [2.0], [4.0]]], dtype=np.float32)
        lengths = np.array([2])

        out, means, stds = standardize_left_padded_sequences(
            x,
            lengths,
            feature_means=np.array([2.0], dtype=np.float32),
            feature_stds=np.array([2.0], dtype=np.float32),
        )

        np.testing.assert_allclose(out[0, :, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(means, np.array([2.0], dtype=np.float32))
        np.testing.assert_allclose(stds, np.array([2.0], dtype=np.float32))


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

    def test_can_train_tiny_model_when_torch_is_available(self) -> None:
        pytest.importorskip("torch", reason="optional 'deep' extra not installed")

        x = np.array(
            [
                [[0.0], [0.0], [1.0]],
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
            ],
            dtype=np.float32,
        )
        y = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        lengths = np.array([1, 2, 3])

        trained = train_lstm_regressor(
            x,
            y,
            lengths,
            config=LSTMTrainingConfig(
                hidden_size=4,
                batch_size=2,
                epochs=1,
                device="cpu",
            ),
        )

        assert len(trained.train_loss) == 1
        assert trained.feature_means.shape == (1,)
        assert trained.target_std > 0.0
