"""Evaluate baseline models across CMAPSS subsets.

Examples:
    uv run python scripts/evaluate_subsets.py --data-dir data/raw \
        --out reports/cross_subset.csv
    uv run python scripts/evaluate_subsets.py --data-dir data/raw \
        --subsets FD002 FD004 --with-xgboost
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from pdm.data import SubsetName, load_subset
from pdm.diagnostics import (
    prediction_error_by_group,
    prediction_error_with_target_caps,
    prediction_rows,
    prediction_s_score_contributions,
)
from pdm.evaluation import Regressor, evaluate_regressor_predictions
from pdm.models import build_baseline_regressor, build_gradient_boosted_regressor

RegimeMode = Literal["auto", "on", "off", "both"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--out", type=Path, default=Path("reports/cross_subset_results.csv"))
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--target-cap-diagnostics-out", type=Path, default=None)
    parser.add_argument("--regime-diagnostics-out", type=Path, default=None)
    parser.add_argument("--s-score-diagnostics-out", type=Path, default=None)
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--n-regimes", type=int, default=6)
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


def _model_specs(include_xgboost: bool) -> list[tuple[str, Callable[[], Regressor]]]:
    specs: list[tuple[str, Callable[[], Regressor]]] = [
        ("ridge", lambda: build_baseline_regressor(alpha=1.0))
    ]
    if include_xgboost:
        specs.append(
            (
                "xgboost",
                lambda: build_gradient_boosted_regressor(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.03,
                ),
            )
        )
    return specs


def _regime_options(subset: str, mode: RegimeMode) -> list[bool]:
    if mode == "auto":
        return [subset in {"FD002", "FD004"}]
    if mode == "on":
        return [True]
    if mode == "off":
        return [False]
    return [False, True]


def summarize_target_cap_diagnostics(
    predictions: pd.DataFrame,
    *,
    max_rul: int,
) -> pd.DataFrame:
    """Aggregate raw-vs-capped target diagnostics by subset and model."""
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


def summarize_operating_regime_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction diagnostics by learned operating regime."""
    if "operating_regime" not in predictions.columns:
        raise KeyError("Missing prediction columns: ['operating_regime']")
    rows = []
    group_columns = ["subset", "model", "seed", "use_regime_features"]
    for keys, group in predictions.dropna(subset=["operating_regime"]).groupby(
        group_columns,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        breakdown = prediction_error_by_group(
            group,
            group_column="operating_regime",
            segment_prefix="op_regime",
        )
        for column in reversed(group_columns):
            breakdown.insert(0, column, key_values[column])
        rows.append(breakdown)
    if not rows:
        raise ValueError("predictions must contain at least one non-null operating_regime value.")
    return pd.concat(rows, ignore_index=True)


def summarize_s_score_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate early-vs-late S-score contributions by subset and model."""
    rows = []
    group_columns = ["subset", "model", "seed"]
    if "use_regime_features" in predictions.columns:
        group_columns.append("use_regime_features")
    for keys, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        breakdown = prediction_s_score_contributions(group)
        for column in reversed(group_columns):
            breakdown.insert(0, column, key_values[column])
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Run the cross-subset evaluation and write a CSV report."""
    args = parse_args()
    rows = []
    prediction_frames = []

    for subset in args.subsets:
        data = load_subset(cast(SubsetName, subset), args.data_dir)
        for use_regime_features in _regime_options(subset, cast(RegimeMode, args.regime_mode)):
            for model_name, model_factory in _model_specs(args.with_xgboost):
                details = evaluate_regressor_predictions(
                    data,
                    model_name=model_name,
                    model_factory=model_factory,
                    max_rul=args.max_rul,
                    use_regime_features=use_regime_features,
                    n_regimes=args.n_regimes,
                )
                result = details.result
                rows.append(result.as_dict())
                predictions = prediction_rows(
                    subset=result.subset,
                    model=result.model_name,
                    seed=None,
                    unit_ids=details.unit_ids,
                    end_cycles=details.end_cycles,
                    y_true=details.y_true,
                    y_pred=details.y_pred,
                )
                predictions["use_regime_features"] = result.use_regime_features
                if details.operating_regimes is not None:
                    predictions["operating_regime"] = details.operating_regimes
                prediction_frames.append(predictions)

    results = (
        pd.DataFrame(rows)
        .sort_values(["subset", "use_regime_features", "model"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_out = args.predictions_out or args.out.with_name(f"{args.out.stem}_predictions.csv")
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)
    target_cap_diagnostics = summarize_target_cap_diagnostics(predictions, max_rul=args.max_rul)
    target_cap_diagnostics_out = args.target_cap_diagnostics_out or args.out.with_name(
        f"{args.out.stem}_target_cap_diagnostics.csv"
    )
    target_cap_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    target_cap_diagnostics.to_csv(target_cap_diagnostics_out, index=False)
    s_score_diagnostics = summarize_s_score_diagnostics(predictions)
    s_score_diagnostics_out = args.s_score_diagnostics_out or args.out.with_name(
        f"{args.out.stem}_s_score_diagnostics.csv"
    )
    s_score_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    s_score_diagnostics.to_csv(s_score_diagnostics_out, index=False)
    regime_diagnostics = None
    if "operating_regime" in predictions.columns and predictions["operating_regime"].notna().any():
        regime_diagnostics = summarize_operating_regime_diagnostics(predictions)
        regime_diagnostics_out = args.regime_diagnostics_out or args.out.with_name(
            f"{args.out.stem}_regime_diagnostics.csv"
        )
        regime_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        regime_diagnostics.to_csv(regime_diagnostics_out, index=False)
    print(results.round({"rmse": 2, "s_score": 2}).to_string(index=False))
    print(f"\nSaved results to {args.out}")
    print(f"Saved predictions to {predictions_out}")
    print(f"Saved target-cap diagnostics to {target_cap_diagnostics_out}")
    print(f"Saved S-score diagnostics to {s_score_diagnostics_out}")
    if regime_diagnostics is not None:
        print(f"Saved operating-regime diagnostics to {regime_diagnostics_out}")


if __name__ == "__main__":
    main()
