# Data Audit — Multi-Testbed Power-System Replay-Attack Extension

Per the scope agreed with the user: the source paper's single 2-state DC-motor Monte-Carlo study (fully reproduced separately, see `REPRODUCTION_REPORT.md`) is extended to **multi-testbed power-system state estimation under replay attacks**, so that a genuine 10+-benchmark ML/DL comparison (macro-F1, etc.) is scientifically meaningful. This document records every design decision in that extension.

## Testbeds

IEEE 14 / 39 / 57 / 118-bus systems, loaded via `pandapower.networks` (`case14`, `case39`, `case57`, `case118`) — the exact 4 systems the master prompt recommends for power-system studies. Topology (bus-branch graph, including transformers) is extracted for the graph-based candidate (`src/pipeline/testbeds.py`).

## Ground-truth dynamics: real load data, not synthetic random walk

Rather than inventing a synthetic load process, the "state" trajectory (bus voltage angles) is driven by **real hourly load data**: `C:\Users\MMASSAOUDI\Desktop\Data\Load Data\PJM_Load_hourly.csv` (32,896 hourly readings, 1998–2002, mean 29,766 MW). A contiguous 3,000-hour slice (~125 days) is used, normalized to a load multiplier (`value/mean`, range ≈[0.59, 1.81]) and applied uniformly to every load bus in the testbed at each hour; DC power flow (`pandapower.rundcpp`) is solved at each of the 3,000 steps to get ground-truth bus voltage angles and line power flows.

This is a **deliberate scientific improvement over the source paper's testbed**, not just a bigger dataset: the source paper's DC-motor system is driven by a *deterministic*, LQR-stabilized trajectory decaying to the origin, which turned out to lack the persistence-of-excitation needed for reliable estimation (see `REPRODUCTION_REPORT.md` §1) — a real weakness (`SOURCE_WEAKNESS_ANALYSIS.md` #2) the paper never discloses. Real hourly load data has natural, non-trivial temporal structure (daily/weekly cycles, genuine variance) that removes this excitation problem entirely and, as a side effect, means a replayed historical window is *not* merely "old data that looks the same" — it can be statistically distinguishable purely from its temporal context (e.g., replaying a low-load overnight window during a midday peak). Whether pure ML/DL residual models can already exploit this cue **without** the source paper's stochastic-dither encoding trick is an explicit research question carried into the benchmark stage (§ "plain vs. encoded streams" below), not assumed either way.

## Measurement model

- **State analogue:** bus voltage angles (`va_degree`) from DC power flow.
- **Measurement analogue (`y_k` in the source paper):** line active-power flows (`p_from_mw`), one per transmission line/transformer — a higher-dimensional, physically redundant measurement set typical of real SCADA/state-estimation systems (unlike the source paper's single scalar output).

Since DC power flow is linear, the true trajectory depends only on the load-multiplier series (not on any later noise/attack draw), so it is computed **once per testbed and cached** (`results/data/<testbed>_true_traj.npz`), then reused across all attack scenarios and noise realizations — the same computational pattern used in `SOURCE_METHOD_REPRODUCTION` for the DC-motor system.

## Replay-attack scenario generation

The source paper's 3 fixed strategies (continuous / periodic / aperiodic) are **generalized, not discarded**: `src/pipeline/attacks.py` draws randomized scenarios of the same 3 structural types (1, 2, or 3 replayed sub-intervals) with randomized duration (8/16/24/36-hour) and randomized placement, so the dataset contains many distinct attack instances rather than 3 fixed ones repeated identically. Each scenario independently draws:
- a **measurement-noise** realization `e_k ~ N(0, σ_e²)` (per-line, scaled to a fraction of that line's own signal variance — see below),
- a **dither** realization `d_k ~ N(0, σ_d²)` for the source-method-style encoding scheme,
- and its own record/attack window placement.

## Plain vs. encoded received streams — an explicit research question

Two parallel "received" measurement streams are generated and stored side by side:
- **`X_plain`**: naive replay with no defense — the receiver simply gets the historical value verbatim during an attack window. This is what a generic ML/DL anomaly detector (most of the literature) would see.
- **`X_encoded`**: the source paper's stochastic dither encoding/decoding scheme (Eq. 19–20) applied — the specific mechanism the source paper claims as a core contribution.

Both are kept through the whole pipeline so the benchmark stage can directly test whether the source paper's encoding trick still adds value once (a) the data has realistic temporal structure and (b) modern spatio-temporal models are used — rather than assuming the source paper's justification (needed to unmask replay against a naive residual check) still holds in this richer setting.

## Noise/dither calibration — fit on TRAIN only

Per-line noise and dither standard deviations are set as a fraction of that line's **own signal standard deviation computed from the TRAIN split only** (`NOISE_FRAC=0.02`, `DITHER_FRAC=0.06`, satisfying `σ_d > σ_e` in the spirit of the source paper's Theorem 1) and then reused unchanged for VAL/TEST noise generation — no statistic is ever computed from val/test data. This mirrors the master prompt's leakage-control requirement (§13) applied to a quantity (sensor precision) that must be fixed before val/test are touched.

## Split policy — chronological, not random-row

Per master-prompt §13 ("prefer temporal split... instead of random row splitting"), the 3,000-hour trajectory is split **chronologically**, contiguous and non-overlapping:

| Split | Hours | Fraction |
|---|---|---|
| train | 0–1800 | 60% |
| val | 1800–2400 | 20% |
| test | 2400–3000 | 20% |

Every attack scenario for a given split is generated using only that split's local time range — no scenario's record or replay window ever crosses a split boundary, so no information about a later time period leaks into an earlier split (or vice versa) through the attack-injection process itself. Recorded in `DATA_SPLIT_MANIFEST.csv` (per testbed × split: window counts, positive rate, scenario count, window length, sensor count, noise/dither fractions).

## Windowing and labels

- Window length `W=12` (12-hour context — spans the shortest attack duration used, 8 hours, and gives useful pre/post context for the shortest onset transition without inflating dimensionality on the largest testbed, case118 with 173 lines).
- Label is **point-wise**: a window ending at time `t` is labeled attacked iff `t` itself falls inside a replay interval (same semantics as the ADR/FAR point-wise detection convention used in `SOURCE_METHOD_REPRODUCTION`).
- Stride 1 (every valid `t` produces one window) — this is the standard sliding-window construction for time-series anomaly detection; it does inflate the effective sample count with highly overlapping windows, which is why the **split is temporal at the scenario/hour level**, not at the window level, so no near-duplicate window ever appears on both sides of a split.

## Class imbalance (expected, not corrected at generation time)

Observed positive rates run ~2–7% across testbeds/splits (see `DATA_SPLIT_MANIFEST.csv`) — genuinely imbalanced, consistent with real attack-detection settings and with the master prompt's mandate to use macro-F1 (not accuracy) as the primary metric. No oversampling/undersampling is applied at the data-generation stage; any class-balancing is a modeling choice to be made per-benchmark, fit on TRAIN only, at the benchmark stage.

## Output artifacts

- `results/data/<testbed>_true_traj.npz` — cached ground-truth trajectory (per testbed).
- `results/data/<testbed>_<split>.npz` — `X_plain`, `X_encoded` (N, W, n_line) float32 raw sequence windows, `y` (N,) labels, `edges` (topology) — for sequence/graph-aware models (STGAT, TCN-CP, LSTM/GRU/Transformer benchmarks).
- `results/data/<testbed>_<split>_stats.csv` — per-window aggregated statistical features (mean/std/min/max/last per sensor, for both plain and encoded streams) — for classical ML benchmarks (LogReg/SVM/kNN/RF/XGBoost/etc.) that need tabular input.
- `DATA_SPLIT_MANIFEST.csv` — the authoritative record of every split's size, positive rate, and generation parameters.

## What is *not* claimed

This is a **simulated benchmark**, not real attack telemetry — exactly like the source paper's own DC-motor study. The contribution is a controlled, physically grounded (DC power-flow, real topology, real load time series), leakage-audited, multi-scale (14→118 bus) testbed suite that is strictly harder and more diverse than the source paper's single toy system, not a claim of having captured real-world SCADA attack data.
