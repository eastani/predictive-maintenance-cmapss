# Operations

## Serving Boundary

The API serves engineered feature vectors against a persisted model artifact.
This keeps ingestion and feature engineering separate from inference serving.

```bash
uv run python scripts/train_fd001_artifact.py \
  --data-dir data/raw \
  --out artifacts/fd001-ridge.joblib

PDM_MODEL_PATH=artifacts/fd001-ridge.joblib \
  uv run uvicorn pdm.api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict-rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 1, "cycle": 120, "features": {"sensor_02_mean_5": 0.1}}'
```

Real requests must provide every feature column stored in the artifact. Missing
values return a 400 response instead of being silently filled.

## Container

```bash
docker build -t predictive-maintenance-cmapss .

docker run --rm -p 8000:8000 \
  -e PDM_MODEL_PATH=/models/fd001-ridge.joblib \
  -v "$PWD/artifacts:/models:ro" \
  predictive-maintenance-cmapss
```

## Dashboard

The dashboard is intentionally a benchmark-inspection tool, not a simulated
live operations screen.

```bash
uv sync --extra viz
uv run python scripts/run_dashboard.py
```

To inspect freshly generated results:

```bash
uv run python scripts/run_dashboard.py --results reports/cross_subset_results.csv
```

## Production Gaps

The repository has a serving boundary, but it is not a full production PdM
system. Missing pieces include:

- Online feature computation from historian or OPC-UA data.
- Drift monitoring and alert calibration.
- Asset metadata and maintenance work-order integration.
- Model registry and staged deployment workflow.
- Backtesting against operational costs, not only benchmark scores.
