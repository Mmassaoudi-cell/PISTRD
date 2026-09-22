# Source Paper Audit

**Paper:** Junshuai Qin, Zhengdao Zhang, Huarong Zhao, "Data-Based Q-Learning for Replay Attack Detection in Cyber-Physical Systems," *IEEE Transactions on Industrial Cyber-Physical Systems*, vol. 3, pp. 165–172, 2025. DOI 10.1109/TICPS.2025.3532881.

Audited in full (all 8 pages, including references and author bios). This document is the ground truth used to build `SOURCE_METHOD_REPRODUCTION`. Any number later reported as "reproduced" is only valid if it was actually produced by code in `src/reproduction/`.

---

## 1. Target problem

Detect **replay attacks** on the sensor-measurement channel of a networked cyber-physical system (CPS) modeled as a discrete-time LTI system, **without knowing the system matrices A, B, C**. A replay attacker records a window of past sensor output and later re-transmits it in place of the live measurement, which can hide other attacks and destabilize closed-loop control.

## 2. Claimed novelty (paper's own framing)

1. First application of **Q-learning to model-free output estimation** (not control) — learns a constant matrix `H_y` from normal operating data, then estimates `ŷ_k` in real time from a sliding window of historical input/output data, with no system model.
2. A **residual-based detector** with a data-derived threshold (max residual-energy observed during normal operation), claimed simpler than prior output-based/residual-based detectors and not requiring a model.
3. A **stochastic measurement encoding/decoding scheme** (dither signal `d_k ~ N(0, Σ_d)`) that converts a *hidden* replay attack into a *detectable additive disturbance* in the residual, without degrading control performance when no attack is present (unlike watermarking schemes that always perturb the actuator).
4. A **theoretical detectability proof** (Theorem 1) giving a sufficient condition on `Σ_d` for guaranteed detection, covering both the "not-yet-fully-replaced" and "fully-replaced" phases of an attack window.

## 3. System / test system

Single simulated 2-state networked DC motor system, sourced from Li & Shen, "A data-driven attack detection approach for DC servo motor systems based on mixed optimization strategy," *IEEE Trans. Ind. Informat.*, 2020 (ref [19] in the paper). Discretized with zero-order hold, sampling time `t = 0.1s`:

```
A = [[-0.0022, -0.0030],
     [ 1.0625,  0.9808]]
B = [[0.9516, 0     ],
     [6.1303, 4.5728]]
C = [0, 0.170667]           # 1×2 row -> scalar output y_k as printed in the paper
```

Process/measurement noise: `w_k ~ N(0, Σ_w)`, `v_k ~ N(0, Σ_v)` (values not given numerically — see §9 Missing). Initial state `x0 = [0.1, -0.1]`.

No other test system, dataset, or real-world data is used anywhere in the paper. There is **no train/val/test split, no labeled dataset, and no cross-validation** — the entire empirical section is a single-testbed Monte-Carlo simulation study.

## 4. Preprocessing / feature engineering

None in the ML sense. The only "feature construction" is the stacking of historical input/output windows into regression vectors `ȳ_{k-1,k-N}`, `ū_{k-1,k-N}` and their Kronecker-product expansion `z̄_k = z_k ⊗ z_k` used to linearize the quadratic Q-function into a linear-in-parameters least-squares problem (Eqs. 7–17).

## 5. Model / architecture

Not a neural network. It is a **batch (offline) least-squares Q-learning / approximate dynamic programming (ADP) scheme**, closely following Rizvi & Lin's output-feedback Q-learning LQR method (ref [26]), repurposed here for *output estimation* rather than control:

- Q-function parameterized as a quadratic form in `[ȳ, ū, y_k]` with unknown kernel matrix `H_y` (Eq. 13).
- `H_y` vectorized (`H̄_y`) and solved via least squares from the Bellman recursion `Ψᵀ H̄_y = Γ` (Eqs. 15–17), using `L ≥ l(l+1)/2` samples where `l = (p+1)N + mN`.
- Iterated until `‖H̄_y^j − H̄_y^{j-1}‖ < ε`.
- Output estimate `ŷ_k` obtained in closed form from `H_y` submatrices (Eq. 18) — no gradient descent, no neural network, no experience replay.

## 6. Attack model / detection mechanism

- Encoding: `y_k^c = ỹ_k + d_k`; decoding: `y_k^d = y_k^c − d_k = ỹ_k` (Eq. 19), synchronized via shared random seed (Remark 2).
- Residual: `r_k = y_k^d − ŷ_k` (Eq. 21).
- Detection statistic: sliding-window average residual energy `g_k = (1/f) Σ_{i=k-f+1}^{k} r_i^T r_i` (Eq. 22), window length `f`.
- Threshold: `g_th = sup g_k` over the attack-free period (Eq. 23) — i.e., **the threshold is literally the max detection statistic observed during the no-attack calibration run**, not a statistically derived quantile.
- Binary decision rule: `g_k > g_th ⇒ H1 (attacked)` else `H0` (Eq. 24).

## 7. Theoretical result

Theorem 1: sufficient condition for detectability, `Σ_d > max(|r'_k^T Δŷ_k| − ½‖Δŷ_k‖²)`, derived separately for the partially-replaced window (`k ∈ (k0, k0+N-1)`) and fully-replaced window (`k ∈ (k0+N, k0+N1-1)`) phases of an attack. Full proof given (Eqs. 25–30).

## 8. Optimization method / training procedure

- "Training" = one offline least-squares solve (or minimum-norm pseudoinverse per Remark 1) over `L_n = 400` samples of closed-loop LQR-stabilized normal operation, iterated to convergence (`ε = 0.1`).
- No stochastic gradient descent, no epochs in the deep-learning sense, no mini-batching.
- Complexity stated by the authors: **O(n^2.807)–O(n^3)** dominated by the matrix inversion in Eq. 17 (Remark 4).

## 9. Hyperparameters (as given in Section V)

| Symbol | Value | Meaning |
|---|---|---|
| `x0` | `[0.1, −0.1]` | initial state |
| `Q_L = R_L` | `0.5·I ∈ R^{2×2}` | LQR controller weighting |
| `S_y` | `0.2·I ∈ R^{2×2}` | output-cost weighting (**dimension inconsistent with scalar y_k** — see §11) |
| `R` | `0.1·I ∈ R^{2×2}` | control-cost weighting in Q-function |
| `L_n` | 400 | samples of normal data used to learn `H_y` |
| `L_t` | 1000 | total simulation length (samples) |
| `f` | 3 | detection window length |
| `ε` | 0.1 | convergence tolerance for `H̄_y` iteration |
| `e_k` covariance `Σ_e` | 0.01, 0.02, 0.03 (3 cases) | "comprehensive noise" injected into measured output |
| `Σ_d` | 0.1, 0.2, 0.3 (paired with the `Σ_e` cases per Theorem 1) | encoding-noise covariance |
| Monte Carlo repetitions | 1000 | per attack strategy, per noise level |

## 10. Attack scenarios (Section V)

Attacker records true output from **30.1 s–50 s** (300s at 10Hz... actually `t=0.1s` sampling ⇒ this is samples ~301–500) and replays it under 3 strategies:

- **Strategy 1 (continuous):** replay recorded 30.1–50s data continuously during 60.1–80s.
- **Strategy 2 (periodic):** replay recorded 30.1–40s data during 60.1–70s and 80.1–90s (10s attack period).
- **Strategy 3 (aperiodic):** replay recorded 42.1–50s data during 53.1–61s, 65.1–73s, 85.1–93s (12s and 20s gaps).

## 11. Metrics

- **ADR** (Attack Detection Rate, %) and **FAR** (False Alarm Rate, %), each averaged over 1000 Monte-Carlo repetitions (Table II, Table III).
- No precision/recall/F1, no ROC/PR-AUC, no confusion matrix, no runtime/latency numbers reported despite the complexity remark.

## 12. Baselines compared (Table III, Figs. 2–4)

1. **Our method** (this paper).
2. **[19] "data-based"** — Li & Shen 2020 mixed-optimization data-driven detector (same DC-motor system; this is the *source of the testbed itself*).
3. **[27] "model-based"** — Ye, Zhang, Guo 2019 stochastic coding detection scheme (requires the true system model).

Only 2 baselines total, both drawn directly from the CPS replay-detection literature — no ML/DL baselines of any kind.

## 13. Reported results (Published, to be checked against reproduction)

**Table II — own method sensitivity across noise levels** (1000 MC reps/strategy):

| Strategy | Σ_e=0.01, Σ_d=0.1 ADR/FAR | Σ_e=0.02, Σ_d=0.2 ADR/FAR | Σ_e=0.03, Σ_d=0.3 ADR/FAR |
|---|---|---|---|
| 1 | 90.00% / 0.25% | 90.50% / 0.25% | 91.50% / 0.5% |
| 2 | 90.50% / 1.25% | 89.50% / 1.75% | 90.50% / 1.50% |
| 3 | 89.58% / 1.94% | 91.67% / 1.67% | 90.83% / 1.67% |

**Table III — comparison at Σ_e=0.01, Σ_d=0.1, f=3** (1000 MC reps/strategy):

| Method | Strat.1 ADR/FAR | Strat.2 ADR/FAR | Strat.3 ADR/FAR |
|---|---|---|---|
| Ours | 90.00% / 0.25% | 90.50% / 1.25% | 89.58% / 1.94% |
| [19] | 83.50% / 1.16% | 82.70% / 1.71% | 82.33% / 2.51% |
| [27] | 89.57% / 1.27% | 89.27% / 1.61% | 89.32% / 2.18% |

Also: detection threshold `g_th = 0.045` at `Σ_e=0.01` (Fig. 1a), rising to ~0.08 and ~0.12 at the higher noise levels per text (thresholds for Fig. 1b–d not separately tabulated beyond these three text-stated values).

## 14. Figures

- Fig. 1(a)–(d): `g_k` vs. threshold under no-attack and the 3 attack strategies (single realization).
- Fig. 2, 3, 4: three-way comparison (ours vs. [19] vs. [27]) of detection curves for strategies 1, 2, 3 respectively.

## 15. Hardware / software environment

**Not stated anywhere in the paper.** No language, library, solver, or hardware is named.

## 16. Ablations / robustness studies

- The 3-noise-level sweep (Table II) is the only sensitivity analysis, and it is not framed as an ablation (no component of the method is removed).
- The 3 attack-strategy sweep is the closest thing to a robustness study (timing/pattern robustness only).
- **No** ablation of the Q-learning estimator vs. the encoding scheme vs. the detector individually. No noise-model misspecification test, no non-Gaussian noise test, no sensor dropout, no parameter-uncertainty sweep on `f`, `L_n`, `ε`, or the LQR weights.

## 17. Statistical testing

None. Only means over 1000 Monte-Carlo repetitions are reported — no variance, no confidence intervals, no significance test against the two baselines.

## 18. Limitations (author-stated)

Only one sentence in the Conclusion: "Future research will focus on extending the proposed detection scheme to more complex systems." No explicit limitations section; scalability, nonlinear systems, multi-sensor/multi-agent CPS, and non-Gaussian noise are implicitly out of scope.

---

## Classification: Fully / Partially / Missing

### Fully specified (directly reproducible)
- System matrices `A, B, C`, sampling time, initial state (Eq. 31, §V text).
- Q-learning algorithm structure and recursion (Eqs. 2–18, Algorithm 1).
- Encoding/decoding scheme and residual/detection-statistic definitions (Eqs. 19–24).
- `L_n=400`, `L_t=1000`, `f=3`, `ε=0.1`, the 3 `(Σ_e, Σ_d)` pairs, 1000 MC repetitions, all 3 attack strategies' timing.
- `Q_L=R_L=0.5I`, `R=0.1I` for the Q-function.

### Partially specified (implementation assumptions required — documented here, not silently invented)
1. **`S_y = 0.2·I ∈ R^{2×2}`** is dimensionally inconsistent with the stated scalar output `y_k = [0, 0.170667]x_k` (`p=1`). *Assumption:* treat `S_y` as a `1×1` scalar `0.2` (drop to the compatible dimension) since `C` as printed gives `p=1`; note this as a likely OCR/notation artifact in the source paper (the original Li & Shen 2020 DC-motor model has 2 outputs — position and velocity — so the source paper may have silently dropped one output row when adapting it). This is documented, not silently resolved, and both variants (`p=1` scalar-output and `p=2` two-output, restoring the likely original `C`) will be implemented and compared in the reproduction so the sensitivity of ADR/FAR to this choice is visible rather than hidden.
2. **Noise covariances `Σ_w`, `Σ_v`** (process/measurement noise of the base system model, Eq. 1) are never given numerically — only the *comprehensive* noise `e_k` added at the output-encoding stage (Eq. 19) is quantified (0.01/0.02/0.03). *Assumption:* set `Σ_w = Σ_v = 0` (i.e., treat Eq. 1's `w_k, v_k` as folded into the single stated `e_k` term used for encoding) since the paper's own Fig. 1/Table II text only ever manipulates `e_k`. This is the simplest and most literal reading; it is stated explicitly in `FINAL_MODEL_CONFIG`-equivalent reproduction settings.
3. **LQR gain**: the paper says the system runs "under stable closed-loop control with an LQR controller" using `Q_L=R_L=0.5I` but never states whether this is a discrete-time LQR on `(A,B,Q_L,R_L)` computed via the standard DARE. *Assumption:* standard discrete-time LQR via `scipy`/`control`-equivalent DARE solve on the true `(A,B)` (available to the simulator, though not to the detection algorithm) is used to generate the closed-loop trajectories used both for `H_y` learning and for simulation.
4. **`N` (historical window length)**: defined as "an upper bound on the observability index of the system" but no explicit value is given for the DC-motor testbed. *Assumption:* since `n=2` and `(A,C)` is observable with `p=1`, the observability index is at most `n=2`; `N=2` is used, documented as an assumption.
5. **Detection-window placement / edge handling** for `g_k` at the start of the detection phase (`k=L_n+1,...`) is not fully specified (e.g., whether the first `f-1` samples of the detection phase use a partial window or wait for `f` full samples). *Assumption:* zero/normal-padding is not used; `g_k` is only evaluated once `f` full residual samples are available, i.e., detection begins at `k = L_n + f`.
6. Exact numeric values of `g_th` for the two higher-noise cases (Table II's `Σ_e=0.02/0.03` rows) are stated once in prose ("0.045, 0.08, and 0.12 respectively") but not shown per-panel in Fig. 1 beyond the first case — reproduction recomputes them per Eq. 23 rather than hardcoding the prose values.

### Missing (cannot be inferred reliably — will not be silently invented)
- Software/hardware/environment used by the authors.
- Any runtime, latency, or memory measurement (despite Remark 4 discussing complexity qualitatively).
- The baseline [19] and [27] method implementations are not given in enough algorithmic detail in *this* paper to reproduce exactly (they are described only in 1–2 summary sentences); reproducing them faithfully would require pulling in the [19]/[27] papers directly. Given they are cited only as external comparators and this paper's own contribution is what must be reproduced first, `SOURCE_METHOD_REPRODUCTION` will implement **our-method only** to full fidelity, and will implement principled, clearly-labeled re-implementations of [19]/[27] only if/when needed for the benchmark stage — never presented as byte-for-byte reproductions of those two papers.
- Random seed(s) used for the paper's own Monte-Carlo runs (irreproducible exactly; seed sensitivity will instead be reported as a distribution over many seeds in this reproduction).

---

## Positioning note for later stages

This paper is a **control-theoretic / detection-theoretic** contribution (ADP-style Q-learning for estimation + stochastic dithering for attack unmasking + threshold detector), evaluated on a **single toy 2-state LTI system** via Monte Carlo simulation — not a machine-learning classification study. Per the scope agreed with the user, the follow-on hybrid-model work will preserve this paper as the primary reproduction/benchmark anchor, and will **extend the empirical surface to multi-testbed power-system state estimation (IEEE bus systems via PandaPower) under replay attacks**, generating labeled residual/feature data so that a proper 10+-benchmark ML/DL comparison (macro-F1, PR-AUC, etc.) becomes scientifically meaningful, while keeping the DC-motor reproduction as the faithful, primary anchor to the source paper.
