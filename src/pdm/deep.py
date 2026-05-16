"""Optional PyTorch sequence models for CMAPSS RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pdm.data import CMAPSSData
from pdm.evaluation import EvaluationResult
from pdm.models import rmse, s_score
from pdm.sequences import build_sequence_dataset

__all__ = [
    "LSTMEvaluationDetails",
    "LSTMTrainingConfig",
    "TrainedLSTMRegressor",
    "build_lstm_regressor",
    "evaluate_lstm_predictions",
    "evaluate_lstm_regressor",
    "is_torch_available",
    "left_padded_to_right_padded",
    "predict_lstm_regressor",
    "standardize_left_padded_sequences",
    "train_lstm_regressor",
]


@dataclass(frozen=True)
class LSTMTrainingConfig:
    """Training knobs for the lightweight LSTM baseline."""

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    gradient_clip_norm: float | None = 1.0
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class TrainedLSTMRegressor:
    """A fitted LSTM model plus its training loss history."""

    model: Any
    config: LSTMTrainingConfig
    train_loss: tuple[float, ...]
    device: str
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float
    target_std: float
    padding_value: float


@dataclass(frozen=True)
class LSTMEvaluationDetails:
    """Detailed LSTM evaluation output for diagnostics."""

    result: EvaluationResult
    y_true: np.ndarray
    y_pred: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    lengths: np.ndarray
    trained: TrainedLSTMRegressor


def is_torch_available() -> bool:
    """Return whether the optional PyTorch dependency is importable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for LSTM models. Install with: uv sync --extra deep"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _validate_model_args(
    *,
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    if n_features <= 0:
        raise ValueError(f"n_features must be strictly positive, got {n_features}.")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be strictly positive, got {hidden_size}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be strictly positive, got {num_layers}.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must lie in [0, 1), got {dropout}.")


def _validate_training_config(config: LSTMTrainingConfig) -> None:
    _validate_model_args(
        n_features=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be strictly positive, got {config.learning_rate}.")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {config.weight_decay}.")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be strictly positive, got {config.batch_size}.")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be strictly positive, got {config.epochs}.")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError(
            "gradient_clip_norm must be strictly positive when provided, "
            f"got {config.gradient_clip_norm}."
        )


def _validate_sequence_arrays(
    x: np.ndarray,
    y: np.ndarray | None,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    x_array = np.asarray(x, dtype=np.float32)
    if x_array.ndim != 3:
        raise ValueError(
            f"x must have shape (n_samples, sequence_length, n_features), got {x.shape}."
        )
    if x_array.shape[0] == 0:
        raise ValueError("x must contain at least one sample.")

    y_array = None if y is None else np.asarray(y, dtype=np.float32)
    if y_array is not None and y_array.shape != (x_array.shape[0],):
        raise ValueError(f"y must have shape ({x_array.shape[0]},), got {y_array.shape}.")

    lengths_array = np.asarray(lengths, dtype=np.int64)
    if lengths_array.shape != (x_array.shape[0],):
        raise ValueError(
            f"lengths must have shape ({x_array.shape[0]},), got {lengths_array.shape}."
        )
    if np.any(lengths_array <= 0):
        raise ValueError("lengths must all be strictly positive.")
    if np.any(lengths_array > x_array.shape[1]):
        raise ValueError("lengths must not exceed sequence_length.")

    return x_array, y_array, lengths_array


def left_padded_to_right_padded(
    x: np.ndarray,
    lengths: np.ndarray,
    *,
    padding_value: float = 0.0,
) -> np.ndarray:
    """Move valid suffixes from left-padded windows to right-padded windows.

    ``pdm.sequences`` stores early windows as ``[pad, pad, valid, valid]`` so
    the final timestep always corresponds to the current cycle. PyTorch packed
    sequences expect ``[valid, valid, pad, pad]``. This helper performs that
    conversion without changing the raw feature values.
    """
    x_array, _, lengths_array = _validate_sequence_arrays(x, None, lengths)
    out = np.full_like(x_array, fill_value=padding_value, dtype=np.float32)
    for row_index, length in enumerate(lengths_array):
        out[row_index, :length, :] = x_array[row_index, -length:, :]
    return out


def _valid_sequence_rows(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    valid_rows = []
    for row_index, length in enumerate(lengths):
        valid_rows.append(x[row_index, -length:, :])
    return np.concatenate(valid_rows, axis=0)


def _fit_sequence_standardization(
    x: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid_values = _valid_sequence_rows(x, lengths)
    means = valid_values.mean(axis=0).astype(np.float32, copy=False)
    stds = valid_values.std(axis=0).astype(np.float32, copy=False)
    stds = np.where(stds == 0.0, 1.0, stds).astype(np.float32, copy=False)
    return means, stds


def standardize_left_padded_sequences(
    x: np.ndarray,
    lengths: np.ndarray,
    *,
    feature_means: np.ndarray | None = None,
    feature_stds: np.ndarray | None = None,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize valid timesteps while preserving padded positions.

    The scaler is fitted only on valid suffix values according to ``lengths``;
    left-padding cells are left at ``padding_value``. This keeps padding from
    skewing sensor statistics and gives the LSTM a better-conditioned input
    space than raw CMAPSS sensor units.
    """
    x_array, _, lengths_array = _validate_sequence_arrays(x, None, lengths)
    if feature_means is None or feature_stds is None:
        means, stds = _fit_sequence_standardization(x_array, lengths_array)
    else:
        means = np.asarray(feature_means, dtype=np.float32)
        stds = np.asarray(feature_stds, dtype=np.float32)
        expected_shape = (x_array.shape[2],)
        if means.shape != expected_shape:
            raise ValueError(f"feature_means must have shape {expected_shape}, got {means.shape}.")
        if stds.shape != expected_shape:
            raise ValueError(f"feature_stds must have shape {expected_shape}, got {stds.shape}.")
        if np.any(stds <= 0.0):
            raise ValueError("feature_stds must all be strictly positive.")

    standardized = np.full_like(x_array, fill_value=padding_value, dtype=np.float32)
    for row_index, length in enumerate(lengths_array):
        valid = x_array[row_index, -length:, :]
        standardized[row_index, -length:, :] = (valid - means) / stds

    return standardized, means, stds


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested!r}.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested but torch.backends.mps.is_available() is false.")
    return requested


def _left_padded_to_right_padded_torch(
    torch: Any,
    x: Any,
    lengths: Any,
    *,
    padding_value: float,
) -> Any:
    out = torch.full_like(x, fill_value=padding_value)
    for row_index, length in enumerate(lengths.tolist()):
        out[row_index, :length, :] = x[row_index, -length:, :]
    return out


def _build_lstm_module(
    torch: Any,
    nn: Any,
    *,
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    padding_value: float,
) -> Any:
    recurrent_dropout = dropout if num_layers > 1 else 0.0

    class LSTMRegressor(nn.Module):  # type: ignore[misc]
        """Small packed-sequence LSTM regressor."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: Any, lengths: Any) -> Any:
            x_right_padded = _left_padded_to_right_padded_torch(
                torch,
                x,
                lengths,
                padding_value=padding_value,
            )
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_right_padded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            return self.head(hidden[-1]).squeeze(-1)

    return LSTMRegressor()


def build_lstm_regressor(
    *,
    n_features: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    padding_value: float = 0.0,
) -> Any:
    """Build an unfitted PyTorch LSTM regressor."""
    _validate_model_args(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    torch, nn, _, _ = _load_torch()
    return _build_lstm_module(
        torch,
        nn,
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        padding_value=padding_value,
    )


def train_lstm_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_lengths: np.ndarray,
    *,
    config: LSTMTrainingConfig | None = None,
    padding_value: float = 0.0,
) -> TrainedLSTMRegressor:
    """Fit the lightweight LSTM baseline on prebuilt sequence windows."""
    config = config or LSTMTrainingConfig()
    _validate_training_config(config)
    x_array, y_array, lengths_array = _validate_sequence_arrays(train_x, train_y, train_lengths)
    if y_array is None:  # pragma: no cover - impossible through the call signature
        raise ValueError("train_y is required.")
    x_array, feature_means, feature_stds = standardize_left_padded_sequences(
        x_array,
        lengths_array,
        padding_value=padding_value,
    )
    target_mean = float(y_array.mean())
    target_std = float(y_array.std())
    if target_std == 0.0:
        target_std = 1.0
    y_array = ((y_array - target_mean) / target_std).astype(np.float32, copy=False)

    torch, nn, DataLoader, TensorDataset = _load_torch()
    torch.manual_seed(config.random_state)
    device = _resolve_device(torch, config.device)

    x_tensor = torch.as_tensor(x_array, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_array, dtype=torch.float32)
    lengths_tensor = torch.as_tensor(lengths_array, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor, lengths_tensor)
    generator = torch.Generator().manual_seed(config.random_state)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = _build_lstm_module(
        torch,
        nn,
        n_features=x_array.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_value=padding_value,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    model.train()
    for _ in range(config.epochs):
        epoch_losses: list[float] = []
        for batch_x, batch_y, batch_lengths in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x, batch_lengths)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)))

    return TrainedLSTMRegressor(
        model=model,
        config=config,
        train_loss=tuple(losses),
        device=device,
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        padding_value=padding_value,
    )


def predict_lstm_regressor(
    trained: TrainedLSTMRegressor,
    x: np.ndarray,
    lengths: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict RUL values from a fitted LSTM model."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be strictly positive, got {batch_size}.")
    x_array, _, lengths_array = _validate_sequence_arrays(x, None, lengths)
    x_array, _, _ = standardize_left_padded_sequences(
        x_array,
        lengths_array,
        feature_means=trained.feature_means,
        feature_stds=trained.feature_stds,
        padding_value=trained.padding_value,
    )
    torch, _, DataLoader, TensorDataset = _load_torch()

    dataset = TensorDataset(
        torch.as_tensor(x_array, dtype=torch.float32),
        torch.as_tensor(lengths_array, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[np.ndarray] = []
    trained.model.eval()
    with torch.no_grad():
        for batch_x, batch_lengths in loader:
            batch_x = batch_x.to(trained.device)
            batch_lengths = batch_lengths.to(trained.device)
            batch_predictions = trained.model(batch_x, batch_lengths)
            unscaled_predictions = (
                batch_predictions.detach().cpu().numpy() * trained.target_std + trained.target_mean
            )
            predictions.append(unscaled_predictions)
    return np.concatenate(predictions).astype(np.float64, copy=False)


def evaluate_lstm_predictions(
    data: CMAPSSData,
    *,
    sequence_length: int = 30,
    stride: int = 1,
    max_rul: int = 125,
    use_regime_features: bool = False,
    n_regimes: int = 6,
    config: LSTMTrainingConfig | None = None,
) -> LSTMEvaluationDetails:
    """Train an LSTM and return predictions plus headline metrics."""
    dataset = build_sequence_dataset(
        data,
        sequence_length=sequence_length,
        stride=stride,
        max_rul=max_rul,
        use_regime_features=use_regime_features,
        n_regimes=n_regimes,
    )
    trained = train_lstm_regressor(
        dataset.train_x,
        dataset.train_y,
        dataset.train_lengths,
        config=config,
        padding_value=dataset.padding_value,
    )
    predictions = np.clip(
        predict_lstm_regressor(trained, dataset.test_x, dataset.test_lengths),
        a_min=0.0,
        a_max=None,
    )
    result = EvaluationResult(
        subset=data.subset,
        model_name="lstm",
        rmse=rmse(dataset.test_y, predictions),
        s_score=s_score(dataset.test_y, predictions),
        n_train_samples=dataset.train_x.shape[0],
        n_test_units=dataset.test_x.shape[0],
        n_features=dataset.n_features,
        use_regime_features=use_regime_features,
    )
    return LSTMEvaluationDetails(
        result=result,
        y_true=dataset.test_y,
        y_pred=predictions,
        unit_ids=dataset.test_unit_ids,
        end_cycles=dataset.test_end_cycles,
        lengths=dataset.test_lengths,
        trained=trained,
    )


def evaluate_lstm_regressor(
    data: CMAPSSData,
    *,
    sequence_length: int = 30,
    stride: int = 1,
    max_rul: int = 125,
    use_regime_features: bool = False,
    n_regimes: int = 6,
    config: LSTMTrainingConfig | None = None,
) -> EvaluationResult:
    """Train and evaluate an LSTM baseline using the official test labels."""
    return evaluate_lstm_predictions(
        data,
        sequence_length=sequence_length,
        stride=stride,
        max_rul=max_rul,
        use_regime_features=use_regime_features,
        n_regimes=n_regimes,
        config=config,
    ).result
