"""Train a small FD001 baseline artifact for the inference API.

Run after downloading CMAPSS:

    uv run python scripts/train_fd001_artifact.py --data-dir data/raw \
        --out artifacts/fd001-ridge.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdm.data import SENSOR_COLUMNS, load_subset
from pdm.features import add_rolling_features, clip_rul, drop_constant_sensors
from pdm.models import build_baseline_regressor, make_xy
from pdm.serving import ModelArtifact, save_model_artifact


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/fd001-ridge.joblib"))
    parser.add_argument("--max-rul", type=int, default=125)
    return parser.parse_args()


def main() -> None:
    """Train and persist a Ridge baseline model artifact."""
    args = parse_args()
    data = load_subset("FD001", args.data_dir)
    train, dropped = drop_constant_sensors(data.train)
    train = train.assign(RUL=clip_rul(train["RUL"], max_rul=args.max_rul))

    sensor_columns = [column for column in SENSOR_COLUMNS if column not in dropped]
    train = add_rolling_features(train, windows=(5, 10, 20), columns=sensor_columns)
    feature_columns = tuple(
        column for column in train.columns if column.startswith("sensor_") and column != "RUL"
    )

    x, y = make_xy(train, feature_columns)
    model = build_baseline_regressor(alpha=1.0)
    model.fit(x, y)

    artifact = ModelArtifact(
        model=model,
        feature_columns=feature_columns,
        model_version="fd001-ridge-baseline",
        metadata={
            "subset": data.subset,
            "max_rul": args.max_rul,
            "dropped_constant_sensors": dropped,
            "windows": [5, 10, 20],
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model_artifact(artifact, args.out)
    print(f"Saved model artifact to {args.out}")


if __name__ == "__main__":
    main()
