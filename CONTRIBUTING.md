# Contributing

Thanks for considering a contribution. This project is a predictive
maintenance reference pipeline, so the bar for changes is not only "does the
metric improve?" but also "can the result be inspected and reproduced?"

## Project Principles

- Keep benchmark claims evidence-backed.
- Report RMSE and CMAPSS S-score together.
- For FD002 and FD004, account for operating regimes.
- For target-convention-sensitive experiments, report raw and capped-125
  metrics side by side.
- Prefer small pull requests with one clear question or one clear improvement.

## Good First Contributions

Good first issues should be narrow, reproducible, and low-risk:

- Improve documentation clarity without changing benchmark claims.
- Add focused tests for existing helper functions.
- Add diagnostic plots or tables from already measured CSV outputs.
- Improve error messages, CLI help text, or type annotations.
- Add small examples that make an existing workflow easier to reproduce.

Avoid using a first contribution to introduce a new model family. New models
need a clear experiment plan and benchmark comparison.

## Experiment Contributions

Before opening an experiment PR, state the hypothesis explicitly:

1. What failure mode are you testing?
2. Which subset is affected?
3. Which baseline are you comparing against?
4. Which metrics will decide whether the change helped?
5. What result would make you reject the change?

Use the [Experiment Plan](docs/experiment-plan.md) as the default standard for
LSTM and benchmark changes.

## Local Setup

```bash
uv sync --extra dev --extra boost --extra viz --extra deep --extra docs
```

Download the CMAPSS data if your change needs real benchmark runs:

```bash
bash scripts/download_data.sh
```

## Validation

For most code changes, run:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src/pdm/__init__.py src/pdm/api.py src/pdm/dashboard.py src/pdm/data.py src/pdm/deep.py src/pdm/diagnostics.py src/pdm/evaluation.py src/pdm/features.py src/pdm/models.py src/pdm/sequences.py src/pdm/serving.py
uv run pytest
```

For documentation changes, also run:

```bash
uv run mkdocs build --strict
```

For diagnostic chart changes, regenerate assets and run the asset test:

```bash
uv run python scripts/generate_diagnostic_assets.py
uv run pytest tests/test_generate_diagnostic_assets.py
```

## Pull Request Expectations

A good PR description includes:

- What changed.
- Why it changed.
- What evidence or validation supports the change.
- Any benchmark numbers that changed.
- Any caveats or claims that should not be made from the result.

If a PR changes benchmark results, include the exact command used to generate
the result and explain whether the comparison uses raw labels, capped-125
labels, or both.

## What To Avoid

- Best-seed reporting without repeated-seed summaries.
- Claiming production readiness from offline benchmark scores.
- Optimizing only raw RMSE while ignoring S-score direction.
- Mixing regime-aware and non-regime-aware results in one aggregate.
- Adding large generated artifacts outside `docs/assets` without a clear reason.
