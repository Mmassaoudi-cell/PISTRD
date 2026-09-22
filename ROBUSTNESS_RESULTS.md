# Robustness: Partial (Subset-Sensor) Replay Attacks

Motivation: the main dataset's attack model replays the *entire* measurement vector at once. `DITHER_NECESSITY_FINDING.md` established that in that whole-vector setting, the spatial (physics) branch adds essentially nothing beyond the temporal branch — a whole-vector replay is a genuinely valid historical snapshot, so it is spatially self-consistent by construction. This experiment (`results/robustness/partial_replay_results.csv`, `src/robustness/partial_replay.py`) tests a harder, more stealthy variant: only a random **fraction** of sensors (100%, 50%, 30%, 15%) are replayed during the attack window, the rest remain live — exactly the scenario the spatial branch was motivated for. Evaluated using the model's **already-frozen** VAL-calibrated thresholds (not re-tuned on this harder data), so this is a genuine generalization test, not an oracle re-fit.

## Headline finding: the temporal branch is far more robust to partial attacks than expected — the spatial branch's incremental value is real but inconsistent, not decisive

| Testbed | Fraction | Full macro-F1 | Temporal-only macro-F1 | Spatial-only macro-F1 | Temporal-only minority recall |
|---|---|---|---|---|---|
| case14 | 1.00 | 0.842 | 0.903 | 0.891 | 1.000 |
| case14 | 0.50 | 0.850 | 0.850 | 0.963 | 1.000 |
| case14 | 0.30 | 0.874 | 0.874 | 0.962 | 1.000 |
| case14 | 0.15 | 0.883 | 0.883 | 0.920 | 0.986 |
| case118 | 1.00 | 0.834 | 0.912 | 0.881 | 1.000 |
| case118 | 0.50 | 0.915 | 0.915 | 0.958 | 1.000 |
| case118 | 0.30 | 0.940 | 0.940 | 0.943 | 1.000 |
| case118 | 0.15 | 0.861 | 0.861 | 0.983 | 1.000 |

(case39/case57 show the same pattern; full table in the CSV.)

Three honest observations, none of which support an oversold "physics branch is essential" narrative:

1. **The temporal (joint-Gaussian one-step Mahalanobis) branch alone stays at or near minority-recall = 1.0 down to 15% of sensors attacked, on every testbed.** The learned inverse-covariance weighting in the Mahalanobis statistic apparently amplifies even a small subset of inconsistent dimensions through their learned correlations with the rest — the mechanism is considerably more robust to attack sparsity than the original design rationale assumed.
2. **The spatial branch's contribution is inconsistent, not monotonic.** It sometimes exceeds temporal-only substantially (case14 frac=0.5: 0.963 vs. 0.850) and sometimes underperforms it (case118 frac=1.0: 0.881 vs. 0.912) — there is no clean "spatial branch wins as attacks get sparser" trend across testbeds.
3. **The OR-fused "Full" score is sometimes worse than temporal-only alone** (e.g., case118 frac=1.0: 0.834 vs. 0.912) — OR-fusion trades some precision for coverage, and when the spatial branch's own false-alarm behavior doesn't generalize as cleanly to a new attack variant as the temporal branch's does, the fusion can cost more than it buys.

## Revised, honest positioning

The manuscript should **not** claim the physics/spatial branch as a decisive, load-bearing contribution — the evidence here doesn't support that. It should instead report:
- The **temporal joint-covariance mechanism is the primary, effective contribution**, validated as robust across whole-vector attacks (main results), partial attacks down to 15% of sensors (this experiment), and across all 4 testbed scales.
- The **spatial/physics branch is retained as a secondary, complementary, interpretable addition** with a real but inconsistent effect — reported honestly as such, with the fused OR-decision's precision/coverage trade-off shown explicitly rather than asserted as a strict improvement.
- This is a legitimate, if less dramatic, scientific outcome: an ablation and a robustness experiment were run specifically to test a design hypothesis (spatial branch matters most for partial attacks), and the result was "partially confirmed, and not decisive" rather than a clean win — reported as such per master-prompt §4/§28 rather than reframed after the fact to sound more decisive than the data supports.
