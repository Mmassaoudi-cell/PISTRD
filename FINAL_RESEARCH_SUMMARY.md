# Final Research Summary

## Revision note
This manuscript went through a full simulated peer review (`/academic-paper-reviewer`, 5-reviewer panel: EIC, methodology, domain, practitioner-perspective, Devil's Advocate — see `review/`), returned **Major Revision**, and was then substantially revised in response. All required revisions were addressed with new experiments, not just rewording; see `review/FAIRNESS_CHECK_RESPONSE.md`, `review/GENERALIZATION_AND_ADVERSARY_RESPONSE.md`, `review/AC_POWERFLOW_RESPONSE.md`, and `review/SOURCE_METHOD_FIDELITY_RESPONSE.md` for the point-by-point evidence. The summary below reflects the revised manuscript.

## Source paper
Qin, Zhang & Zhao, "Data-Based Q-Learning for Replay Attack Detection in Cyber-Physical Systems," *IEEE Trans. Ind. Cyber-Phys. Syst.*, vol. 3, 2025. A control-theoretic (not ML/dataset) study: batch least-squares Q-learning output estimator + stochastic dither-encoding scheme + residual threshold detector, evaluated via Monte Carlo simulation on a single 2-state DC-motor system.

## Reproduction: partial success, honestly quantified
Reproduced from the paper's own equations. Directionally and structurally correct (flat ADR across noise levels, comparable across 3 attack strategies — matching every qualitative claim) but numerically inexact (MAE 3.83pp ADR, 5.79pp FAR). Root cause: an undisclosed persistence-of-excitation requirement for the least-squares solve, and a threshold rule (`sup` of a finite calibration sample) that varies >60% across independent calibration draws. Not retuned to force agreement with published numbers. See `REPRODUCTION_REPORT.md`.

## Extension: multi-testbed power-system state estimation
The source paper's single 2-state toy system was extended (with user approval) to replay-attack detection in IEEE 14/39/57/118-bus state estimation via PandaPower DC power flow, driven by real PJM hourly load data, chronologically split (60/20/20) with leakage controls. See `DATA_AUDIT.md`.

## Candidate models evaluated
4 candidates designed against ranked weaknesses in the source method (`MODEL_CANDIDATES.md`): RLS-CUSUM (online recursive + change detection), STGAT (graph attention over a line-graph topology), PIRK (joint multivariate residual), TCN-CP (conformal-calibrated temporal convolution). Validation-only screening (3 seeds x 4 testbeds): PIRK won decisively (mean macro-F1 0.998 vs. 0.54–0.69 for the others).

## Final selected model: PISTRD
PIRK extended into a two-branch design: a joint one-step ridge-regression Mahalanobis residual (temporal branch) plus a classical DC power-flow WLS state-estimation residual using the true topology (spatial branch), OR-fused with independently VAL-calibrated quantile thresholds (explicit target false-alarm rate, replacing the source method's uncalibrated `sup` threshold). Hyperparameter sensitivity checked (macro-F1 varies ≤0.006 across a 9-point grid) before freezing (`FINAL_MODEL_CONFIG.yaml`).

## Strongest benchmark and headline result
Against 16 benchmarks (source-method-adapted, 3 losing candidates, 3 classical, 6 boosting/tree, 3 deep) on the untouched TEST split: **PISTRD wins all 16 (16 wins / 0 ties / 0 losses)**. PISTRD macro-F1: 0.984/0.994/0.994/0.998 (case14/39/57/118). Strongest benchmark overall: logistic regression, mean macro-F1 0.747. See `BENCHMARK_WTL.csv`.

**Is the margin real, or withheld information?** Directly tested (`review/FAIRNESS_CHECK_RESPONSE.md`) rather than assumed: giving the strongest tabular benchmark PISTRD's own one-step lag feature barely moves the gap; giving it the *full pairwise quadratic interaction structure* (the exact form of a Mahalanobis distance) closes most of the gap (case14: 0.557→0.956; PISTRD 0.984). Conclusion: the advantage is sample-efficient closed-form estimation of an O(n²) joint-covariance structure under severe class imbalance (196–328 positive TRAIN examples per testbed), not withheld information — a more precise and more defensible claim than raw architectural superiority.

## Minority recall, FPR, efficiency
PISTRD minority-class recall 0.977–1.0, FPR 0.0000–0.0033 across testbeds. Runtime 3.3s mean (fit+val+test) — faster than most boosting methods, 5–35x faster than every deep-learning competitor, while also the most accurate method (no accuracy/efficiency trade-off). Sample-efficiency check: macro-F1 identical to 4 decimal places from 2% to 100% of training scenarios.

## Ablation conclusion
Full PISTRD ≈ Temporal-only (marginally lower on 3/4 testbeds) ≫ generic baselines; Spatial-only close behind Full. The spatial branch's contribution to the main (whole-vector replay) attack model is confirmed empirically to be marginal — expected, since a whole-vector replay is spatially self-consistent by construction.

## Robustness conclusion
Partial (subset-sensor) replay attacks: the temporal branch alone retains minority recall ≈1.0 down to 15% of sensors attacked on every testbed — more robust than the design rationale assumed. The spatial branch's incremental value here is real but inconsistent across testbeds/fractions, not decisive. Reported honestly rather than oversold (`ROBUSTNESS_RESULTS.md`).

## Statistical significance (corrected in revision)
The original pooled Wilcoxon construction pseudo-replicated a fixed per-testbed PISTRD value across seeds, overstating precision — caught in review (Methodology Reviewer). Replaced with a per-testbed sign test, Fisher-combined across testbeds, Holm-corrected across all 16 comparisons (`results/stats/wtl_analysis_v2.py`). Because a per-testbed sign test cannot reach p<0.05 below 5 independent seeds regardless of effect size, the 7 stochastic tabular benchmarks were re-run at 20 seeds each (up from 5): **all 7 reach significance (p<10⁻⁵)**. The 3-seed deep/graph models and the 6 deterministic/1-seed methods do not reach significance under this corrected, conservative test — disclosed as a genuine power limitation, not suppressed. **7/16 significant under a construction with no known validity problem** (down from the original invalid construction's 11/16 — a smaller, more honest number). Rank-biserial effect size (−1.0 for every comparison under the original construction) is retained only as a descriptive direction-consistency summary, explicitly flagged as mechanically guaranteed once no sign-crossovers exist, not as independent evidence of magnitude.

## Most important honest finding
`DITHER_NECESSITY_FINDING.md`: the hypothesis that realistic, non-stationary real-load data would make the source paper's stochastic dither-encoding mechanism unnecessary is **falsified**, decisively and consistently, on all 4 testbeds. Without encoding, every method (temporal, spatial, fused) collapses to macro-F1 ≈0.48–0.53 (random); with it, 0.95–0.998. This validates, rather than displaces, the source paper's central design choice.

## Post-review robustness/generalization additions
- **Informed-attacker camouflage** (`review/GENERALIZATION_AND_ADVERSARY_RESPONSE.md`): attacks timed exactly 24k hours (k=1,2,3) after recording — exploiting daily load periodicity, the single most obvious evasion strategy — gain the attacker nothing (minority recall ≈1.0 on every testbed/offset). Mechanistically explained: detection depends on dither-key mismatch, not load-level similarity.
- **Cross-load-source generalization**: PISTRD, trained only on PJM data, transfers with negligible degradation to ISO New England and Malaysian grid load data never seen during training (case118: undegraded; case14/Malaysia: precision drops but recall stays 1.0 — no missed attacks).
- **AC power-flow validation** (`review/AC_POWERFLOW_RESPONSE.md`): full nonlinear AC physics (vs. the DC approximation used throughout) degrades overall performance modestly (0.984→0.925 on case14) and causes the spatial branch specifically to collapse toward random (0.953→0.576) when its topology matrix is left DC-derived — the expected, mechanistically-clean signature of physics-model mismatch, motivating re-deriving the spatial branch from the true AC Jacobian as concrete future work.

## Remaining limitations
- Simulated benchmark (DC power flow for the main results, AC validated separately for case14; simulated attacks on real load data) — not real SCADA/PMU attack telemetry.
- Real PMU/SCADA telemetry and coordinated multi-stage attacks (replay + topology manipulation, or replay + partial encoding-key compromise) remain untested.
- The uniformly maximal effect size (−1.0 against every benchmark, original construction) is a large margin; the fairness test and corrected statistics above are the direct response, not a dismissal of the concern.
- 9 of 16 benchmark comparisons (6 deterministic + 3-seed deep/graph models) do not reach formal significance under the corrected, adequately-powered protocol — disclosed, not suppressed; their point-estimate gaps remain large (Table in `BENCHMARK_WTL.csv`).
- The encoding-key-compromise degradation curve (partial key knowledge, not full loss) is not characterized.

## Recommended venue
*IEEE Transactions on Industrial Cyber-Physical Systems* (same venue as the source paper) or *IEEE Transactions on Smart Grid* / *IEEE Transactions on Information Forensics and Security*, given the power-system state-estimation framing and the statistical-calibration contribution.
