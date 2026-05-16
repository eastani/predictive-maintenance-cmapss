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
from pdm.deep import LSTMTrainingConfig, evaluate_lstm_regressor

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


def main() -> None:
    """Run the LSTM evaluation and write a CSV report."""
    args = parse_args()
    rows = []

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
                result = evaluate_lstm_regressor(
                    data,
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    max_rul=args.max_rul,
                    use_regime_features=use_regime_features,
                    n_regimes=args.n_regimes,
                    config=config,
                )
                row = result.as_dict()
                row["sequence_length"] = args.sequence_length
                row["stride"] = args.stride
                row["epochs"] = args.epochs
                row["seed"] = seed
                rows.append(row)

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

    print(results.round({"rmse": 2, "s_score": 2}).to_string(index=False))
    print(f"\nSaved results to {args.out}")
    print("\nSummary:")
    print(summary.round({"rmse_mean": 2, "rmse_std": 2, "s_score_mean": 2, "s_score_std": 2}))
    print(f"\nSaved summary to {summary_out}")


if __name__ == "__main__":
    main()
