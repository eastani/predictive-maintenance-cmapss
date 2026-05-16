"""Evaluate the optional PyTorch LSTM baseline on CMAPSS subsets.

Examples:
    uv sync --extra deep
    uv run python scripts/evaluate_lstm.py --data-dir data/raw --subsets FD001
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from pdm.data import SubsetName, load_subset
from pdm.deep import LSTMTrainingConfig, evaluate_lstm_predictions
from pdm.diagnostics import (
    prediction_error_breakdown,
    prediction_error_by_rul_band,
    prediction_error_with_target_caps,
    prediction_rows,
)

RegimeMode = Literal["auto", "on", "off", "both"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default=["FD001"],
    )
    parser.add_argument("--out", type=Path, default=Path("reports/lstm_results.csv"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    parser.add_argument("--rul-band-diagnostics-out", type=Path, default=None)
    parser.add_argument("--target-cap-diagnostics-out", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--n-regimes", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--regime-mode",
        choices=["auto", "on", "off", "both"],
        default="auto",
        help=(
            "Regime-feature strategy: auto enables regimes only for FD002/FD004; "
            "on/off force one setting; both runs an ablation with and without regimes."
        ),
    )
    return parser.parse_args()


def _regime_options(subset: str, mode: RegimeMode) -> list[bool]:
    if mode == "auto":
        return [subset in {"FD002", "FD004"}]
    if mode == "on":
        return [True]
    if mode == "off":
        return [False]
    return [False, True]


def summarize_lstm_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated LSTM runs by subset and preprocessing setting."""
    group_columns = [
        "subset",
        "model",
        "use_regime_features",
        "sequence_length",
        "stride",
        "epochs",
    ]
    summary = (
        results.groupby(group_columns, as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            s_score_mean=("s_score", "mean"),
            s_score_std=("s_score", "std"),
            runs=("seed", "count"),
            n_train_samples=("n_train_samples", "first"),
            n_test_units=("n_test_units", "first"),
            n_features=("n_features", "first"),
        )
        .fillna({"rmse_std": 0.0, "s_score_std": 0.0})
        .sort_values(["subset", "use_regime_features", "model"])
        .reset_index(drop=True)
    )
    return summary


def summarize_prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction diagnostics by subset, model, and seed."""
    rows = []
    group_columns = ["subset", "model", "seed"]
    if "use_regime_features" in predictions.columns:
        group_columns.append("use_regime_features")
    for keys, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        breakdown = prediction_error_breakdown(group)
        for column in reversed(group_columns):
            breakdown.insert(0, column, key_values[column])
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


def summarize_rul_band_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction diagnostics by true-RUL band."""
    rows = []
    group_columns = ["subset", "model", "seed"]
    if "use_regime_features" in predictions.columns:
        group_columns.append("use_regime_features")
    for keys, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        breakdown = prediction_error_by_rul_band(group)
        for column in reversed(group_columns):
            breakdown.insert(0, column, key_values[column])
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


def summarize_target_cap_diagnostics(
    predictions: pd.DataFrame,
    *,
    max_rul: int,
) -> pd.DataFrame:
    """Aggregate raw-vs-capped target diagnostics by subset, model, and seed."""
    rows = []
    group_columns = ["subset", "model", "seed"]
    if "use_regime_features" in predictions.columns:
        group_columns.append("use_regime_features")
    for keys, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        breakdown = prediction_error_with_target_caps(group, caps=(None, float(max_rul)))
        for column in reversed(group_columns):
            breakdown.insert(0, column, key_values[column])
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Run the LSTM evaluation and write a CSV report."""
    args = parse_args()
    rows = []
    prediction_frames = []

    for subset in args.subsets:
        data = load_subset(cast(SubsetName, subset), args.data_dir)
        for use_regime_features in _regime_options(subset, cast(RegimeMode, args.regime_mode)):
            for seed in args.seeds:
                config = LSTMTrainingConfig(
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    random_state=seed,
                    device=args.device,
                )
                details = evaluate_lstm_predictions(
                    data,
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    max_rul=args.max_rul,
                    use_regime_features=use_regime_features,
                    n_regimes=args.n_regimes,
                    config=config,
                )
                result = details.result
                row = result.as_dict()
                row["sequence_length"] = args.sequence_length
                row["stride"] = args.stride
                row["epochs"] = args.epochs
                row["seed"] = seed
                rows.append(row)
                predictions = prediction_rows(
                    subset=result.subset,
                    model=result.model_name,
                    seed=seed,
                    unit_ids=details.unit_ids,
                    end_cycles=details.end_cycles,
                    y_true=details.y_true,
                    y_pred=details.y_pred,
                )
                predictions["use_regime_features"] = result.use_regime_features
                prediction_frames.append(predictions)

    results = (
        pd.DataFrame(rows)
        .sort_values(["subset", "use_regime_features", "model"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    summary = summarize_lstm_results(results)
    summary_out = args.summary_out or args.out.with_name(f"{args.out.stem}_summary.csv")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_out = args.predictions_out or args.out.with_name(f"{args.out.stem}_predictions.csv")
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)
    diagnostics = summarize_prediction_diagnostics(predictions)
    diagnostics_out = args.diagnostics_out or args.out.with_name(f"{args.out.stem}_diagnostics.csv")
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_out, index=False)
    rul_band_diagnostics = summarize_rul_band_diagnostics(predictions)
    rul_band_diagnostics_out = args.rul_band_diagnostics_out or args.out.with_name(
        f"{args.out.stem}_rul_band_diagnostics.csv"
    )
    rul_band_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    rul_band_diagnostics.to_csv(rul_band_diagnostics_out, index=False)
    target_cap_diagnostics = summarize_target_cap_diagnostics(predictions, max_rul=args.max_rul)
    target_cap_diagnostics_out = args.target_cap_diagnostics_out or args.out.with_name(
        f"{args.out.stem}_target_cap_diagnostics.csv"
    )
    target_cap_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    target_cap_diagnostics.to_csv(target_cap_diagnostics_out, index=False)

    print(results.round({"rmse": 2, "s_score": 2}).to_string(index=False))
    print(f"\nSaved results to {args.out}")
    print("\nSummary:")
    print(summary.round({"rmse_mean": 2, "rmse_std": 2, "s_score_mean": 2, "s_score_std": 2}))
    print(f"\nSaved summary to {summary_out}")
    print(f"Saved predictions to {predictions_out}")
    print(f"Saved diagnostics to {diagnostics_out}")
    print(f"Saved RUL-band diagnostics to {rul_band_diagnostics_out}")
    print(f"Saved target-cap diagnostics to {target_cap_diagnostics_out}")


if __name__ == "__main__":
    main()
