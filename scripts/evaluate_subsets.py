"""Evaluate baseline models across CMAPSS subsets.

Examples:
    uv run python scripts/evaluate_subsets.py --data-dir data/raw \
        --out reports/cross_subset.csv
    uv run python scripts/evaluate_subsets.py --data-dir data/raw \
        --subsets FD002 FD004 --with-xgboost
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

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
    parser.add_argument("--metadata-out", type=Path, default=None)
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


def _metadata_output_path(results_out: Path, metadata_out: Path | None) -> Path:
    """Resolve the benchmark metadata output path."""
    return metadata_out or results_out.with_name(f"{results_out.stem}_metadata.json")


def _dependency_version(package_name: str) -> str | None:
    """Return an installed dependency version when it is importable by metadata."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash, or None outside a git checkout."""
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return None
    return commit


def _json_ready(value: Any) -> Any:
    """Convert argparse values into JSON-serializable metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _model_metadata(include_xgboost: bool) -> list[dict[str, Any]]:
    """Describe the configured benchmark models without fitting them."""
    models: list[dict[str, Any]] = [
        {
            "name": "ridge",
            "estimator": "sklearn.linear_model.Ridge",
            "parameters": {"alpha": 1.0},
        }
    ]
    if include_xgboost:
        models.append(
            {
                "name": "xgboost",
                "estimator": "xgboost.XGBRegressor",
                "parameters": {
                    "n_estimators": 500,
                    "max_depth": 3,
                    "learning_rate": 0.03,
                },
            }
        )
    return models


def build_run_metadata(
    args: argparse.Namespace,
    *,
    command: list[str],
    output_files: dict[str, Path],
    started_at: str,
    finished_at: str,
    n_result_rows: int,
    n_prediction_rows: int,
    git_commit: str | None,
) -> dict[str, Any]:
    """Build reproducibility metadata for a cross-subset benchmark run."""
    return {
        "schema_version": 1,
        "script": "scripts/evaluate_subsets.py",
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "git": {"commit": git_commit},
        "inputs": {
            "data_dir": str(args.data_dir),
            "subsets": list(args.subsets),
        },
        "configuration": {
            key: _json_ready(value)
            for key, value in vars(args).items()
            if key not in {"out", "metadata_out"}
        },
        "target_convention": {
            "training_max_rul": args.max_rul,
            "headline_metrics": "raw_test_rul",
            "diagnostics": "raw_and_capped_test_rul",
        },
        "models": _model_metadata(args.with_xgboost),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": _dependency_version("scikit-learn"),
            "xgboost": _dependency_version("xgboost") if args.with_xgboost else None,
        },
        "outputs": {name: str(path) for name, path in output_files.items()},
        "counts": {
            "result_rows": n_result_rows,
            "prediction_rows": n_prediction_rows,
        },
    }


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
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
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
    output_files = {
        "results": args.out,
        "predictions": predictions_out,
        "target_cap_diagnostics": target_cap_diagnostics_out,
        "s_score_diagnostics": s_score_diagnostics_out,
    }
    if regime_diagnostics is not None:
        output_files["regime_diagnostics"] = regime_diagnostics_out
    metadata_out = _metadata_output_path(args.out, args.metadata_out)
    output_files["metadata"] = metadata_out
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    metadata = build_run_metadata(
        args,
        command=sys.argv,
        output_files=output_files,
        started_at=started_at,
        finished_at=finished_at,
        n_result_rows=len(results),
        n_prediction_rows=len(predictions),
        git_commit=_git_commit(),
    )
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(results.round({"rmse": 2, "s_score": 2}).to_string(index=False))
    print(f"\nSaved results to {args.out}")
    print(f"Saved predictions to {predictions_out}")
    print(f"Saved target-cap diagnostics to {target_cap_diagnostics_out}")
    print(f"Saved S-score diagnostics to {s_score_diagnostics_out}")
    if regime_diagnostics is not None:
        print(f"Saved operating-regime diagnostics to {regime_diagnostics_out}")
    print(f"Saved run metadata to {metadata_out}")


if __name__ == "__main__":
    main()
