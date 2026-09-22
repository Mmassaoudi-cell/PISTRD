# Source Method Weakness Analysis

Basis: `SOURCE_PAPER_AUDIT.md` + hands-on reproduction in `src/reproduction/` (see `REPRODUCTION_REPORT.md` for the numeric comparison). Weaknesses are ranked by expected impact on a follow-on hybrid design, not by how "interesting" they are.

---

## Ranked weaknesses

### 1. [HIGH] Threshold is a single order statistic (`sup` of a small no-attack sample), not a calibrated quantile — fragile and undocumented
`g_th = sup_{g_k ∈ L1} g_k` (Eq. 23) sets the operating threshold to the single largest value of the detection statistic observed during a finite attack-free calibration window. This is:
- **High-variance by construction.** Our reproduction shows the reproduced threshold swings by >2× depending on which finite calibration sample is used (median 0.048–0.071 across calibration-seed batches at `σ_e=0.01`, vs. the published 0.045) — i.e., the paper's own reported number is itself just one draw from a distribution the paper never characterizes.
- **No FAR/ADR trade-off control.** A max-based threshold gives no principled way to trade false alarms against detection rate (no ROC curve, no target FAR). Any operator who needs FAR < 0.1% instead of ~1–2% has no lever to pull except re-running calibration and hoping for a larger sample max.
- **Silently non-stationary.** If the no-attack statistic's distribution has heavier tails than Gaussian (plausible, since `g_k` is a sum of squared residuals), a longer calibration run will keep pushing `g_th` up indefinitely — the paper never shows this doesn't happen.

**Impact on design:** any successor method should replace the max-of-sample threshold with a statistically grounded one (e.g., an extreme-value/quantile-based threshold with a stated target FAR, or a learned/adaptive threshold with calibrated confidence).

### 2. [HIGH] Identifiability of `H_y` implicitly requires persistence-of-excitation that the paper never states
The batch least-squares solve of `Ψᵀ H̄_y = Γ` (Eq. 17) needs `ΨΨᵀ` full rank. Under the paper's own closed-loop LQR operation with no stated process/measurement noise on the state (only the downstream "comprehensive noise" `e_k` is quantified), the true state trajectory converges toward the origin within a few samples (dominant closed-loop eigenvalue ≈0.98 but with strong control authority from `B`), starving the regression of persistently-exciting data over most of the `L_n=400`-sample learning window.

In reproduction, omitting an explicit exploration/probing signal on `u_k` during learning made `H_y` numerically unidentifiable; a small probing perturbation had to be added (Assumption #7 in the audit) to get a well-posed regression at all — and the reproduced ADR/FAR are visibly sensitive to the probing amplitude used (see `REPRODUCTION_REPORT.md`). **The paper claims a "model-free" pipeline but never discloses or analyzes the excitation requirement that makes the pipeline work at all** — this is a real gap between the theoretical algorithm (Algorithm 1) and what is actually needed to run it, and it is not mentioned anywhere in Section III–V or Remark 4's complexity discussion.

**Impact on design:** a successor method should either (a) explicitly characterize/inject the excitation needed for identifiability, or (b) avoid needing an explicit persistently-exciting calibration phase altogether (e.g., an online/recursive estimator that keeps adapting rather than freezing `H_y` after one offline batch fit).

### 3. [MEDIUM-HIGH] Single fixed `H_y`, no online adaptation — stale under any drift
`H_y` is learned **once** from `L_n` samples and held constant for the rest of operation (Algorithm 1, lines 1–3 vs. 4–14). If the plant, controller gains, or noise statistics drift even slightly after the calibration window (very plausible in real industrial CPS — wear, retuning, load changes), the residual `r_k` picks up model-mismatch energy indistinguishable from an attack, silently degrading both ADR and FAR without any mechanism to detect or correct for it. The 3-noise-level sweep in Table II is the only robustness evidence offered, and it re-learns `H_y` fresh at each noise level rather than testing drift *after* learning.

### 4. [MEDIUM] No ablation of the three coupled mechanisms (encoding, estimator, thresholding)
The method bundles three independent ideas — (a) Q-learning output estimation, (b) stochastic dither encoding to unmask replay, (c) max-based residual thresholding — but the paper never isolates their individual contributions. Table III compares against *entirely different* competing pipelines ([19], [27]), not against "ours minus encoding" or "ours minus Q-learning estimator (e.g., replaced by a simple linear predictor)". It is therefore impossible to tell from the paper alone whether the reported gains come from the Q-learning estimator being more accurate, from the encoding scheme, or from the (very lenient, max-based) thresholding choice.

### 5. [MEDIUM] Weak/narrow benchmark set, and the DC-motor testbed is over-simple
Only 2 baselines ([19], [27]), both from the same narrow CPS replay-detection sub-literature, both evaluated on the *same single 2-state, 1-output, deterministic-in-truth* toy plant that the authors did not even design themselves (borrowed from [19]). No classical statistical baseline (CUSUM, chi-squared residual test, Kalman-innovation test), no ML/DL baseline, no second test system of different scale, no nonlinear system, no multi-sensor system. A method that only has ever been shown to work on one 2-state SISO(effectively) LTI plant offers very limited evidence of generality — this is the single most consequential limitation for anyone trying to build on this work.

### 6. [MEDIUM] Metrics are minimal and time-window semantics are unclear
Only ADR and FAR (point estimates, mean over 1000 MC reps) are reported — no detection latency (time-to-detect after attack onset, despite the theoretical analysis explicitly distinguishing an early "partially-replaced" phase `(k0,k0+N-1)` from a later "fully-replaced" phase, Eqs. 28–30, which is exactly a detection-latency argument the empirical section never measures), no variance/CI on ADR/FAR across the 1000 reps, no ROC/PR curve, no confusion-matrix-style breakdown by attack strategy severity.

### 7. [MEDIUM] Complexity claim (Remark 4) is asserted, never measured
`O(n^2.807)`–`O(n^3)` for the Eq. 17 matrix inversion is stated but no wall-clock time, FLOPs, or scaling curve (e.g., ADR/FAR/runtime vs. state dimension `n`) is given anywhere. For a paper explicitly motivated by wanting a lightweight, deployable industrial detector, there is zero empirical efficiency evidence.

### 8. [LOW-MEDIUM] Restriction to linear, Gaussian, single-plant setting
The entire method (LQR, ARE, quadratic Q-function, Gaussian dither) is provably exact only for LTI systems with Gaussian noise. Real industrial CPS frequently have nonlinear dynamics, actuator saturation, and non-Gaussian disturbances (the paper's Remark 3 asserts robustness to "low-quality data" but this is not tested — the only noise sweep is 3 Gaussian variance levels of the same noise model).

### 9. [LOW] Encoding scheme adds a synchronized-random-seed dependency that is a soft security assumption, unexamined
Remark 2 assumes "the attacker cannot access the encoding and decoding information" and that synchronization is maintained via a shared seed. No discussion of what happens under seed desynchronization, replay of the *encoded* channel itself, or an adversary with partial side-channel knowledge of the dither process. This is a legitimate but completely unexplored threat-model boundary.

---

## Predictive / architectural / computational / scientific — summary matrix

| Category | Key issue | Severity |
|---|---|---|
| Predictive | Threshold fragility (max-of-sample) inflates FAR variance, no latency measure | High |
| Architectural | No online adaptation of `H_y`; single fixed batch fit | Med-High |
| Architectural | Implicit, undisclosed persistence-of-excitation requirement | High |
| Computational | No runtime/latency/FLOPs evidence despite a complexity remark | Medium |
| Scientific | 2 narrow baselines, 1 toy testbed, no ablation, no statistics beyond MC mean | High (compounded) |

---

## What this implies for the hybrid-candidate stage

The weaknesses above point toward improvements along three largely independent axes, which will shape the 3–5 candidate designs in `MODEL_CANDIDATES.md`:

1. **Estimation axis** — replace/augment the fixed batch Q-learning estimator with something that (a) does not need a hand-tuned excitation phase, and/or (b) adapts online instead of freezing after `L_n` samples.
2. **Decision axis** — replace the max-of-sample threshold with a statistically calibrated / adaptive detector (targets a stated FAR, quantifies uncertainty, and ideally reports detection latency, not just a point ADR/FAR).
3. **Evidence axis** — regardless of architecture, close the scientific gaps: multi-testbed evaluation (per the agreed scope, extending to IEEE bus-system state estimation under replay attacks via PandaPower), a real ablation of the three coupled mechanisms, a proper (10+) benchmark suite, and statistical testing across seeds — independent of which candidate architecture wins.
