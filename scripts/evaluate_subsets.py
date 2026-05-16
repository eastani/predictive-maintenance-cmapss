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
from pdm.evaluation import Regressor, evaluate_regressor
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


def main() -> None:
    """Run the cross-subset evaluation and write a CSV report."""
    args = parse_args()
    rows = []

    for subset in args.subsets:
        data = load_subset(cast(SubsetName, subset), args.data_dir)
        for use_regime_features in _regime_options(subset, cast(RegimeMode, args.regime_mode)):
            for model_name, model_factory in _model_specs(args.with_xgboost):
                result = evaluate_regressor(
                    data,
                    model_name=model_name,
                    model_factory=model_factory,
                    max_rul=args.max_rul,
                    use_regime_features=use_regime_features,
                    n_regimes=args.n_regimes,
                )
                rows.append(result.as_dict())

    results = (
        pd.DataFrame(rows)
        .sort_values(["subset", "use_regime_features", "model"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(results.round({"rmse": 2, "s_score": 2}).to_string(index=False))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
