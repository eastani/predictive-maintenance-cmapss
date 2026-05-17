# Experiment Plan

This project is not trying to prove that a single model family always wins.
The next experiments are designed to test specific failure hypotheses and to
avoid confusing benchmark artifacts with real model improvement.

## Current Evidence

The strongest current conclusion is not "XGBoost wins" or "LSTM loses." The
evidence is more specific:

| Observation | Interpretation |
| ----------- | -------------- |
| XGBoost improves RMSE on all four subsets. | Extra model capacity helps point accuracy, but not always asymmetric operational cost. |
| XGBoost worsens S-score on FD001 and FD003. | Lower RMSE can still create costly late predictions. |
| Regime-aware features help FD002 and FD004. | Multiple operating conditions need explicit preprocessing, not one global sensor scale. |
| FD002 high-RUL errors shrink under capped-125 scoring. | Raw-label scores partly measure target-convention mismatch. |
| FD002 S-score is mostly early-prediction cost. | The dominant issue is over-conservative high-RUL underprediction, not near-failure optimism. |
| FD004 has a larger late-prediction share. | The safety risk profile differs by subset and cannot be inferred from RMSE alone. |

## Hypotheses To Test

### 1. Target Convention Hypothesis

**Question:** Are poor high-RUL predictions caused mainly by the model, or by
training on capped labels while evaluating on raw labels?

**Test:** Report raw and capped-125 metrics for every new model run.

**Decision rule:** A model is not considered better unless it improves the
relevant metric under both the raw and capped conventions, or unless the
documentation explicitly explains why one convention is the operating target.

**Do not do:** Do not tune a model only against raw RMSE and then explain away
the result after the fact.

### 2. Sequence Model Hypothesis

**Question:** Can the LSTM close the gap on FD002/FD004 when it sees denser
training windows and trains for longer?

**Test:** Run repeated-seed LSTM experiments with a controlled grid:

| Variable | First values |
| -------- | ------------ |
| Subsets | FD002, FD004 |
| Sequence length | 30, 50 |
| Stride | 5, 10 |
| Epochs | 5, 10 |
| Seeds | At least 3 per configuration |

**Decision rule:** Report mean and standard deviation, not a best seed. Compare
against regime-aware Ridge and XGBoost under raw, capped-125, RUL-band, and
S-score contribution diagnostics.

**Do not do:** Do not claim a sequence-model advantage from one lucky seed or
from capped-only evaluation.

### 3. Late-Risk Hypothesis

**Question:** Does a lower-RMSE model increase unsafe late predictions?

**Test:** Track early/late counts, S-score contribution share, and maximum late
error for every benchmark run.

**Decision rule:** If a model improves RMSE but increases late S-score share,
it needs explicit operational justification before being treated as superior.

**Do not do:** Do not use RMSE as the deployment selection metric by itself.

### 4. Regime Robustness Hypothesis

**Question:** Is the average score hiding poor behavior in one operating
regime?

**Test:** Keep per-regime residual diagnostics for FD002 and FD004.

**Decision rule:** A model is more credible when it improves the worst regime,
not only the average. If the best-regime score improves while the worst-regime
score degrades, document the tradeoff.

**Do not do:** Do not report only aggregate FD002/FD004 metrics for
multi-condition experiments.

## Operational Risks

The benchmark is a public simulation, but the same failure modes map to real
predictive-maintenance systems:

| Risk | Why it matters | Current mitigation |
| ---- | -------------- | ------------------ |
| Target convention mismatch | A model can look bad because the evaluation label convention differs from training. | Raw and capped-125 diagnostics are reported together. |
| Late prediction risk | Overestimating remaining life can delay maintenance and miss failures. | S-score and late-contribution diagnostics are tracked. |
| Over-conservative alerts | Underestimating RUL can waste maintenance capacity and reduce trust. | Early S-score share and high-RUL bands are tracked. |
| Regime-specific weakness | One operating condition can be poorly served by a model with good average performance. | Per-regime residual diagnostics are reported. |
| Experiment variance | Neural models can look strong or weak depending on seed and window density. | Repeated-seed summaries are required for LSTM claims. |
| Offline-only validation | Historical benchmark scores do not prove production readiness. | Serving boundary exists, but drift checks and alert policy are still future work. |

## Next Practical Experiments

1. **FD002 LSTM density test**
   - Run `stride=5`, `epochs=5`, `seeds=42 43 44`.
   - Compare against the existing `stride=10`, `epochs=3` result.
   - Stop early if variance stays high and RMSE remains above XGBoost under
     both raw and capped-125 scoring.

2. **FD004 LSTM pilot**
   - Start with the same CPU-friendly setting used for FD002.
   - Use it as a risk scan, not a performance claim.
   - Pay particular attention to late S-score contribution because FD004 shows
     more late-risk than FD002.

3. **Diagnostic plot refresh**
   - Regenerate static diagnostic assets whenever benchmark tables change.
   - Keep the charts in sync with the documented tables.

## What Would Count As Progress

Useful progress is not only a lower RMSE. A stronger result would show at least
one of the following:

- Better RMSE without increasing late S-score share.
- Better worst-regime RMSE on FD002 or FD004.
- Lower high-RUL compression under both raw and capped target conventions.
- Lower variance across repeated LSTM seeds.
- A clearer operational policy for choosing between an early-biased and
  late-biased model.

## What Is Out Of Scope For The Next Step

- Large hyperparameter searches without a diagnostic question.
- Adding more model families before the LSTM hypotheses are tested.
- Claiming production readiness without drift detection, alert thresholds, and
  model versioning.
- Optimizing only for leaderboard-style raw RMSE.
