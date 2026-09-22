# Model Selection Report

Stage 1 (smoke test) and Stage 2 (validation-only screening, 3 seeds × 4 testbeds, TRAIN+VAL only — TEST never touched) are complete. Raw results: `results/screening/screening_results.csv`; aggregated: `results/screening/screening_summary.csv`, `results/screening/screening_overall.csv`.

## Overall screening result (mean across all 4 testbeds, all seeds)

| Candidate | macro-F1 | balanced acc. | minority recall | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| **C3 PIRK** | **0.9977** | **0.9974** | **0.9951** | **0.0003** | **0.9993** | **1.0000** |
| C2 STGAT | 0.6899 | 0.6754 | 0.3812 | 0.0305 | 0.3223 | 0.7157 |
| C1 RLS-CUSUM | 0.6471 | 0.6379 | 0.3183 | 0.0425 | 0.2830 | 0.7414 |
| C4 TCN-CP | 0.5424 | 0.5383 | 0.1320 | 0.0555 | 0.2215 | 0.5560 |

**Per-testbed, PIRK is consistently at or near ceiling** (macro-F1 0.996–0.999 on every one of case14/39/57/118), with no seed variance (it is a closed-form fit, not randomly initialized — "seed" is not a meaningful axis of variation for this candidate). The other three candidates cluster well below it (0.48–0.77) with real seed-to-seed variance for the neural candidates (C2, C4).

## Decision: Candidate 3 (PIRK) selected, with one required extension before freezing

**PIRK wins decisively and consistently — this is not a close call requiring Stage 3's 2–3-candidate tuning shootout.** Per master-prompt §10, the final candidate must not be picked merely for beating weak baselines, so this result was checked for a trivial-task artifact before being accepted:

- **Why PIRK wins, mechanistically:** our replay attacks (per `attacks.py`/`build_dataset.py`) replace the *entire* measurement vector at time `t` with a genuinely real historical vector from a different time — internally consistent, but generally inconsistent with the smooth local temporal evolution that real load data produces (`DATA_AUDIT.md`'s "plain vs. encoded" note flagged this as a real, expected effect of using real load data, not a bug). PIRK's joint one-step linear predictor + full-covariance Mahalanobis residual is precisely the right statistic to catch exactly this kind of joint temporal discontinuity — a textbook match between mechanism and attack model, not an accident of a leaky pipeline. Cross-checked for leakage: `PIRKSimplified.fit` only uses TRAIN-split scenario-internal consecutive pairs, is scored on VAL, and window construction was independently audited in `DATA_AUDIT.md`.
- **Why C2/C4 underperform here:** both are trained as generic supervised classifiers under a fixed, modest epoch budget (25 epochs, appropriate for *screening*, not final performance) with no architectural access to an explicit joint-covariance statistic — STGAT's spatial attention and TCN-CP's temporal convolution have to *learn* the joint-consistency structure from scratch and, at screening budget, only partially do so (val AUC was still visibly improving with more epochs in pre-screening diagnostics). This is a legitimate screening finding (candidates that need substantially more training budget than a closed-form competitor are a real practical weakness for this problem, not a sign the comparison is invalid) — recorded here rather than discarded.
- **Why C1 also underperforms:** RLS-CUSUM's per-sensor-independent online adaptation, by design, discards exactly the cross-sensor joint structure that turns out to be the dominant detection signal in this multi-bus setting — a direct, informative confirmation that weakness **W5** (source paper's total lack of spatial structure) is the single most consequential axis in this extension, exactly as hypothesized in `MODEL_CANDIDATES.md`'s ranking.

**Required extension before freezing:** the *screening* version of PIRK (`src/candidates/pirk.py`) is a purely data-driven joint-Gaussian one-step predictor — it does not yet use the actual power-system topology, despite being named for physics-informed design in `MODEL_CANDIDATES.md`. Freezing that version as the final model would be scientifically thin (a reviewer could reasonably call it "Hotelling's T² with a linear predictor," a decades-old technique with no clear new contribution) and would not exploit the one piece of side information genuinely unique to this domain: the DC power-flow topology matrix is available from the testbed (unlike the source paper's genuinely unknown `A,B,C`). Before freezing, `pirk.py` is extended to a genuine two-mechanism hybrid:

1. **Temporal branch** (already validated above): joint one-step linear predictor + Mahalanobis residual.
2. **Spatial branch** (new): a classical weighted-least-squares power-system state-estimation residual using the true DC power-flow topology (`Bbus`/`Bf`, extracted from PandaPower's internal DC solve) — the standard bad-data-detection statistic from the power-systems literature, here repurposed and *fused* with the temporal branch rather than used alone, which to our knowledge is not how it's combined with replay-attack detection specifically (see `LITERATURE_NOTE.md`).
3. **Calibrated fusion decision layer**: the two residual statistics are combined into one chi-squared-calibrated statistic with an analytically stated target FPR, replacing the source paper's max-of-sample threshold (fixes **W1**).

This keeps the "2–4 complementary mechanisms" design principle (master-prompt §7) and gives the final model an honest, checkable novelty claim beyond the already-decisive screening number. Given PIRK's screening performance is already near-ceiling, the spatial branch is not expected to move macro-F1 much further (a ceiling effect) — its purpose is interpretability, formal calibration, and (to be tested explicitly in the robustness stage) resilience to topology changes and sensor dropout that a purely data-driven joint model has no principled way to handle.

## Candidates 1, 2, 4 — disposition

Not discarded: retained as three of the mandatory 10+ benchmarks in the final comparison (`BENCHMARK_WTL.csv` stage), since they represent legitimate, differently-motivated approaches, not straw men. C2 (STGAT) and C4 (TCN-CP) will be given the same (or greater) training budget as in the final benchmark run as a fairness check, though screening already suggests neither is likely to threaten PIRK's result.

## Addendum: physics extension result and freeze

`src/candidates/pirk_final.py` (`PISTRD`) implements the two-branch design above. Verified (empirically, against `net.res_line.p_from_mw`) that PandaPower's internal `Bf` DC power-flow matrix, restricted to its first `n_line` rows, exactly reconstructs our measured line flows from bus voltage angles — giving a genuine, checkable physics observation model for the spatial branch (`src/candidates/physics_utils.py`).

**Validation-only results (F1-optimal threshold on the fused score, per-testbed):**

| Testbed | macro-F1 | minority recall | FPR | PR-AUC |
|---|---|---|---|---|
| case14 | 0.9837 | 1.0000 | 0.0033 | 0.9836 |
| case39 | 0.9937 | 0.9883 | 0.0009 | 0.9983 |
| case57 | 0.9936 | 0.9767 | 0.0000 | 1.0000 |
| case118 | 0.9984 | 0.9940 | 0.0000 | 1.0000 |

Slightly below the temporal-only screening version (0.996–0.999) — an expected, small, and accepted trade-off: the spatial branch contributes essentially nothing to detecting *this* dataset's whole-vector replay attacks (confirmed analytically: a whole-vector replay is a genuinely valid historical physical snapshot, so it is spatially self-consistent by construction and invisible to any single-instant physics check) but is retained for the robustness experiments targeting partial/subset replay attacks, where spatial self-consistency is expected to genuinely break.

**Hyperparameter sensitivity check** (ridge `alpha` ∈ {0.1, 1, 10}, `cov_reg` ∈ {1e-4, 1e-3, 1e-2}, case14 + case118): validation macro-F1 varies by ≤0.006 across the entire grid — the result is not a fragile, cherry-picked artifact.

**`FINAL_MODEL_CONFIG.yaml` is now frozen.** Per master-prompt §11, no further architecture change will be made based on TEST performance from this point forward.

## Next steps (per execution order)

1. ~~Implement the physics-informed spatial branch and fuse with the validated temporal branch.~~ done
2. ~~Stage 3 validation-only hyperparameter check.~~ done
3. ~~Freeze `FINAL_MODEL_CONFIG.yaml`.~~ done
4. Build out the full 10+ benchmark suite and run the frozen final model + all benchmarks against the untouched TEST split.
