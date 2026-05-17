## Summary

- 

## Why

-

## Validation

- [ ] `uv run ruff check src tests scripts`
- [ ] `uv run ruff format --check src tests scripts`
- [ ] `uv run mypy src/pdm/__init__.py src/pdm/api.py src/pdm/dashboard.py src/pdm/data.py src/pdm/deep.py src/pdm/diagnostics.py src/pdm/evaluation.py src/pdm/features.py src/pdm/models.py src/pdm/sequences.py src/pdm/serving.py`
- [ ] `uv run pytest`
- [ ] `uv run mkdocs build --strict`

## Benchmark Impact

If this changes benchmark behavior, include:

- exact command:
- subset(s):
- raw metrics:
- capped-125 metrics, if applicable:
- S-score early/late contribution, if applicable:
- caveats:

If this does not change benchmark behavior, write: `No benchmark behavior change.`

## Claim Discipline

- [ ] The PR does not claim model superiority from a single seed.
- [ ] The PR does not rely on RMSE alone when S-score is relevant.
- [ ] The PR documents raw vs capped target convention when target labels are part of the result.
