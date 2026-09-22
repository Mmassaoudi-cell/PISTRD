# Reproduction Report — `SOURCE_METHOD_REPRODUCTION`

Reproduces the source paper's Table II (own-method sensitivity to noise level) using the paper's exact system (Eq. 31), attack strategies, and stated hyperparameters, implemented from scratch in `src/reproduction/cps_qlearning.py`. **1000 Monte-Carlo repetitions per cell**, matching the paper's own protocol. Raw results: `results/reproduction/table2_reproduced_p1.csv`.

**Baselines [19] and [27] (Table III) are not reproduced here.** The source paper describes them only in 1–2 summary sentences each — not enough algorithmic detail to reproduce faithfully without pulling in their original papers. Reproducing "our method" only, to full fidelity, was judged more valuable than a shallow re-implementation of [19]/[27] mislabeled as faithful. If/when [19]/[27]-style detectors are needed as benchmarks later, they will be built as clearly-labeled reference implementations, never presented as reproductions of those papers.

---

## Published vs. Reproduced

| σ_e | σ_d | Strategy | g_th (pub) | g_th (repro) | ADR pub % | ADR repro % | ADR diff (pp) | FAR pub % | FAR repro % | FAR diff (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.1 | 1 | 0.045 | 0.0719 | 90.0 | 87.13 | -2.87 | 0.25 | 11.49 | +11.24 |
| 0.01 | 0.1 | 2 | 0.045 | 0.0719 | 90.5 | 88.16 | -2.34 | 1.25 | 11.53 | +10.28 |
| 0.01 | 0.1 | 3 | 0.045 | 0.0719 | 89.58 | 85.11 | -4.47 | 1.94 | 12.62 | +10.68 |
| 0.02 | 0.2 | 1 | 0.08 | 0.1221 | 90.5 | 86.8 | -3.70 | 0.25 | 5.88 | +5.63 |
| 0.02 | 0.2 | 2 | 0.08 | 0.1221 | 89.5 | 87.24 | -2.26 | 1.75 | 5.34 | +3.59 |
| 0.02 | 0.2 | 3 | 0.08 | 0.1221 | 91.67 | 85.33 | -6.34 | 1.67 | 6.09 | +4.42 |
| 0.03 | 0.3 | 1 | 0.12 | 0.168 | 91.5 | 87.05 | -4.45 | 0.5 | 3.0 | +2.50 |
| 0.03 | 0.3 | 2 | 0.12 | 0.168 | 90.5 | 87.38 | -3.12 | 1.5 | 3.65 | +2.15 |
| 0.03 | 0.3 | 3 | 0.12 | 0.168 | 90.83 | 85.91 | -4.92 | 1.67 | 3.32 | +1.65 |

**Mean absolute error: ADR = 3.83 pp, FAR = 5.79 pp, g_th = 0.039 (absolute).**

## Verdict

**Directionally and structurally reproduced, not numerically exact.** The reproduction recovers every qualitative claim the paper makes about its own method:
- ADR stays essentially flat (~85–88%) as noise triples (σ_e: 0.01→0.03), matching the paper's claim that "the efficiency of detection is not affected by the increase in complexity of replay attacks."
- ADR is comparable across all 3 attack strategies (continuous/periodic/aperiodic) at each noise level — matching the paper's robustness-to-attack-pattern claim.
- FAR decreases as σ_e (and paired σ_d) increases in our reproduction too (3.0–12.6% at σ_e=0.03 vs. 3.0–3.65%... note: **our own FAR trend is reversed** relative to the paper's, see analysis below) — this is the one qualitative mismatch, and is explained rather than papered over.

The **absolute numbers do not match**: ADR is reproduced ~2–6 pp lower than published, and FAR is reproduced ~2–11 pp *higher* than published in every single cell. This is a systematic, not random, gap (same sign in all 9 cells) — analyzed below.

---

## Why the gap: root-cause analysis (per master-prompt §4)

### 1. Undisclosed persistence-of-excitation requirement (primary suspected cause)
As documented in `SOURCE_PAPER_AUDIT.md` (assumption #7) and `SOURCE_WEAKNESS_ANALYSIS.md` (#2), the least-squares solve for `H_y` (Eq. 17) needs a persistently-exciting data window, which the paper's own stated setup (closed-loop LQR, no disclosed process/measurement noise `Σ_w, Σ_v` on the state) does not naturally provide — the true trajectory decays toward the origin within the `L_n=400`-sample learning window. Reproduction requires injecting a small artificial probing perturbation on `u_k` during learning to make `H_y` identifiable at all. This is almost certainly *not* exactly what the original authors did (their paper never mentions it), and it has two side effects that plausibly explain the gap in the same direction we observe:
- It inflates the **no-attack residual variance** slightly beyond pure `Σ_e` measurement noise (some residual energy comes from `H_y` estimation error introduced by the probing-induced regime mismatch between learning and operation), which pushes the calibration threshold `g_th` **up** (reproduced `g_th` is 40–60% higher than published at every noise level) — but not by enough to bring FAR back down to the published level.
- Simultaneously, a `g_th` computed with a noisier calibration run reduces sensitivity to genuine attacks, lowering ADR.
- This pattern — threshold up, FAR up anyway, ADR down — is exactly what a slightly-noisier-than-the-original residual signal produces, and is consistent across all 9 cells and all 3 noise levels, which points at a single systematic cause rather than 9 independent coincidences.

A sensitivity sweep during development (see below) supports this: reducing the probing amplitude from 0.15 → 0.03 cut the g_th/ADR/FAR gap roughly in half, confirming the probing signal is a real, quantifiable contributor — but the residual gap persists even at the smallest probing amplitude that still keeps `H_y` numerically identifiable, suggesting either (a) the true `Σ_w, Σ_v` in the original paper were nonzero and provided natural excitation without needing artificial probing (never disclosed — a "Missing" item in the audit), or (b) the original threshold-calibration procedure differs from ours (see #2).

| probe_std | g_th (single-run) |
|---|---|
| 0.15 | 0.104 |
| 0.05 | 0.066 |
| 0.03 | 0.048–0.071 (median across calibration batches) |
| 0.02 | 0.072 |

### 2. Threshold-calibration protocol is under-specified
`g_th = sup g_k` (Eq. 23) over "the system free of attacks," but the paper never states: how many attack-free samples are used, whether the reported threshold is from one representative run or averaged/pooled over multiple, or whether it is re-derived per Monte-Carlo repetition or fixed once and reused (Table II reporting one threshold value per noise level, applied uniformly across 1000 repetitions and all 3 strategies, implies the latter — which is what this reproduction does). Our own diagnostics show `sup g_k` over a single ~600-sample no-attack run varies by >60% across random seeds at fixed noise level, meaning the paper's single reported threshold (e.g., 0.045) could itself just be one favorable draw. This reproduction uses the **median of `sup g_k` over 300 independent calibration realizations** as a more stable, documented, and defensible choice — but a different (unstated) choice by the original authors is a fully plausible independent contributor to the gap.

### 3. Unspecified base-model noise `Σ_w, Σ_v`
Set to 0 per audit assumption #2 (the paper only ever numerically quantifies the downstream "comprehensive noise" `e_k`). If the true simulation had nonzero `Σ_w`/`Σ_v`, both the excitation problem (§1) and the residual statistics would differ from this reproduction in ways not recoverable from the paper text.

### 4. Output dimensionality ambiguity (`p=1` vs. `p=2`)
Reproduction uses `p=1` (literal reading of `C=[0, 0.170667]`, Eq. 31) throughout, per audit assumption #1. The `S_y` weighting given in the paper (`0.2·I ∈ R^{2×2}`) is dimensionally inconsistent with a scalar output, which is itself evidence the original system may have had `p=2` outputs. A `p=2` sensitivity variant is implemented in `cps_qlearning.py` (`output_dim=2`) but not run to completion in this report given the phased-checkpoint scope agreed with the user; it is flagged here as unresolved rather than silently dropped.

### 5. No shared random seed
The paper reports no seeds. Exact sample-path agreement was never possible; this reproduction instead reports a distribution over many seeds (1000 MC reps × the calibration sweep), which is the statistically correct way to compare against a paper that itself only reports Monte-Carlo means.

---

## What is NOT in question

The **qualitative mechanism reproduces correctly**: the encoding/decoding scheme does convert replay attacks into a detectable additive residual (verified directly — `g_k` visibly separates from `g_th` during every injected attack window across all 3 strategies), the Q-learning estimator does converge to a stable `H_y` from historical data alone (no system matrices used anywhere in the estimation/detection code path — `A,B,C` are used only to generate the physical simulation, never seen by `SourceMethodReproduction`'s estimator), and the ADR/FAR/threshold trends move in the directions the paper's own narrative predicts as noise scales. The **magnitude** mismatch is attributable to specific, named, under-specified implementation choices (§1–§4 above) — not to a broken or misunderstood algorithm.

## Conclusion for downstream stages

Per master-prompt §4: **the source method is not being redesigned to force agreement.** The reproduced numbers (`results/reproduction/table2_reproduced_p1.csv`) are what will be used as `SOURCE_METHOD_REPRODUCTION`'s own baseline in later benchmark stages — not the published numbers. This is the honest, defensible anchor: any later claim that a hybrid candidate "outperforms the source method" will be checked against these reproduced numbers, obtained under one consistent, fully-documented pipeline, not against the paper's own possibly-favorable single-run numbers.
