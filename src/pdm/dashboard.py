"""Dashboard helpers for CMAPSS benchmark results.

The pure pandas helpers in this module are intentionally importable without
Dash installed. The web app itself is built lazily by :func:`create_dash_app`
so the core package and CI test environment do not need dashboard dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BENCHMARK_ROWS: tuple[dict[str, str | int | float | bool], ...] = (
    {
        "subset": "FD001",
        "model": "ridge",
        "rmse": 18.266572450079927,
        "s_score": 592.5992938083934,
        "n_train_samples": 20631,
        "n_test_units": 100,
        "n_features": 105,
        "use_regime_features": False,
    },
    {
        "subset": "FD001",
        "model": "xgboost",
        "rmse": 18.230478513582547,
        "s_score": 814.841774889357,
        "n_train_samples": 20631,
        "n_test_units": 100,
        "n_features": 105,
        "use_regime_features": False,
    },
    {
        "subset": "FD002",
        "model": "ridge",
        "rmse": 29.722072888690104,
        "s_score": 15282.532348360342,
        "n_train_samples": 53759,
        "n_test_units": 259,
        "n_features": 294,
        "use_regime_features": True,
    },
    {
        "subset": "FD002",
        "model": "xgboost",
        "rmse": 28.21415963246592,
        "s_score": 11269.466512708423,
        "n_train_samples": 53759,
        "n_test_units": 259,
        "n_features": 294,
        "use_regime_features": True,
    },
    {
        "subset": "FD003",
        "model": "ridge",
        "rmse": 19.165895175209425,
        "s_score": 720.0106492080727,
        "n_train_samples": 24720,
        "n_test_units": 100,
        "n_features": 112,
        "use_regime_features": False,
    },
    {
        "subset": "FD003",
        "model": "xgboost",
        "rmse": 18.724528615223452,
        "s_score": 1412.18727353794,
        "n_train_samples": 24720,
        "n_test_units": 100,
        "n_features": 112,
        "use_regime_features": False,
    },
    {
        "subset": "FD004",
        "model": "ridge",
        "rmse": 30.678632489421357,
        "s_score": 6946.847347696393,
        "n_train_samples": 61249,
        "n_test_units": 248,
        "n_features": 294,
        "use_regime_features": True,
    },
    {
        "subset": "FD004",
        "model": "xgboost",
        "rmse": 28.919442838490536,
        "s_score": 5912.410685250083,
        "n_train_samples": 61249,
        "n_test_units": 248,
        "n_features": 294,
        "use_regime_features": True,
    },
)

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "subset",
    "model",
    "rmse",
    "s_score",
    "n_train_samples",
    "n_test_units",
    "n_features",
    "use_regime_features",
)


def load_benchmark_results(path: str | Path | None = None) -> pd.DataFrame:
    """Load benchmark results from CSV or bundled measured defaults."""
    if path is None:
        df = pd.DataFrame(DEFAULT_BENCHMARK_ROWS)
    else:
        df = pd.read_csv(path)

    missing = [column for column in BENCHMARK_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Benchmark results are missing columns: {missing}")

    return df.loc[:, BENCHMARK_COLUMNS].sort_values(["subset", "model"]).reset_index(drop=True)


def model_delta_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize XGBoost minus Ridge deltas per subset."""
    required_models = {"ridge", "xgboost"}
    rows: list[dict[str, str | float | bool]] = []

    for subset, group in results.groupby("subset", sort=True):
        available = set(group["model"])
        if not required_models.issubset(available):
            continue

        indexed = group.set_index("model")
        rmse_delta = float(indexed.loc["xgboost", "rmse"] - indexed.loc["ridge", "rmse"])
        s_score_delta = float(indexed.loc["xgboost", "s_score"] - indexed.loc["ridge", "s_score"])
        rows.append(
            {
                "subset": str(subset),
                "rmse_delta": rmse_delta,
                "s_score_delta": s_score_delta,
                "xgboost_better_rmse": rmse_delta < 0,
                "xgboost_better_s_score": s_score_delta < 0,
            }
        )

    return pd.DataFrame(rows)


def create_dash_app(results_path: str | Path | None = None) -> Any:
    """Create a Plotly Dash dashboard for benchmark inspection."""
    try:
        import plotly.express as px
        from dash import Dash, dash_table, dcc, html
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Dashboard dependencies are required. Install with: uv sync --extra viz"
        ) from exc

    results = load_benchmark_results(results_path)
    deltas = model_delta_summary(results)

    app = Dash(__name__, title="CMAPSS Benchmark Dashboard")
    app.layout = html.Main(
        [
            html.H1("CMAPSS Predictive Maintenance Benchmarks"),
            html.P(
                "Compare Ridge and XGBoost across FD001-FD004. Lower RMSE and S-score are better; "
                "S-score penalizes late predictions more heavily."
            ),
            html.Section(
                [
                    html.H2("Headline Metrics"),
                    dcc.Graph(
                        figure=px.bar(
                            results,
                            x="subset",
                            y="rmse",
                            color="model",
                            barmode="group",
                            title="RMSE by subset and model",
                        )
                    ),
                    dcc.Graph(
                        figure=px.bar(
                            results,
                            x="subset",
                            y="s_score",
                            color="model",
                            barmode="group",
                            title="CMAPSS S-score by subset and model",
                        )
                    ),
                ]
            ),
            html.Section(
                [
                    html.H2("XGBoost minus Ridge"),
                    dcc.Graph(
                        figure=px.bar(
                            deltas,
                            x="subset",
                            y=["rmse_delta", "s_score_delta"],
                            barmode="group",
                            title="Negative deltas mean XGBoost improves over Ridge",
                        )
                    ),
                ]
            ),
            html.Section(
                [
                    html.H2("Raw Results"),
                    dash_table.DataTable(
                        data=results.round({"rmse": 2, "s_score": 2}).to_dict("records"),
                        columns=[{"name": column, "id": column} for column in results.columns],
                        page_size=8,
                        sort_action="native",
                    ),
                ]
            ),
        ]
    )
    return app
