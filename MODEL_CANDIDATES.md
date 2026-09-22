# Hybrid Candidate Models

Derived directly from the ranked weaknesses in `SOURCE_WEAKNESS_ANALYSIS.md`. Each candidate exists to fix specific, named weaknesses — not because its components are fashionable. Per the agreed scope, the empirical surface is extended from the source paper's single 2-state DC-motor system to **multi-testbed power-system state estimation under replay attacks** (IEEE 14/39/57/118-bus, via PandaPower, with real hourly load time series from `C:\Users\MMASSAOUDI\Desktop\Data\Load Data\PJM_Load_hourly.csv` driving realistic temporal load variation — see `DATA_AUDIT.md`). This unlocks a genuine topology/spatial axis the source paper's SISO toy testbed never had access to, which is why a graph-structured candidate is scientifically justified here and would not have been for the original testbed alone.

Weaknesses being targeted (numbering matches `SOURCE_WEAKNESS_ANALYSIS.md`):
- **W1** fragile max-of-sample threshold, no statistical calibration, no FAR/ADR trade-off control
- **W2** undisclosed persistence-of-excitation requirement for the batch least-squares solve
- **W3** single fixed `H_y`, no online adaptation to drift
- **W5** single toy testbed, no spatial/topology structure exploited
- **W6** no detection latency, no uncertainty quantification
- **W7** no measured efficiency evidence

---

## Candidate 1 — RLS-CUSUM: Recursive Least-Squares Q-Estimator with Sequential Change Detection

**Fixes:** W2 (excitation-hungry batch solve), W3 (frozen `H_y`), W1 (ad-hoc threshold), W6 (no latency guarantee).

**Architecture:** Replace the one-shot batch least-squares solve of `Ψᵀ H̄_y = Γ` (source Eq. 17) with a recursive least-squares (RLS) update of `H̄_y` that keeps adapting every sample instead of freezing after `L_n`. Replace the max-of-sample threshold `g_th = sup g_k` with a CUSUM (cumulative sum) sequential-change statistic on the residual energy.

**Core formulation:**
```
H̄_y,k = H̄_y,k-1 + K_k (Γ_k − ψ_kᵀ H̄_y,k-1),   K_k = P_{k-1} ψ_k / (λ + ψ_kᵀ P_{k-1} ψ_k)
P_k = (1/λ)(P_{k-1} − K_k ψ_kᵀ P_{k-1})                          [forgetting factor λ ∈ (0,1]]
S_k = max(0, S_{k-1} + r_kᵀr_k − ν),   flag if S_k > h            [CUSUM, Wald-style]
```
`ν` (reference/slack) and `h` (threshold) are set via Wald's average-run-length bounds given a target FAR, instead of an empirical sample max.

**Why it should outperform:** directly removes the two mechanisms diagnosed as most responsible for the reproduction gap (undisclosed PE requirement compounding with a frozen estimator, plus a threshold with no FAR guarantee). CUSUM also naturally reports detection latency (steps until `S_k > h`), which the source paper's Theorem 1 discusses theoretically but never measures.

**Efficiency:** O(l²) per-step RLS update replaces repeated O(l³) batch re-solves; no calibration-only phase needed since it's always adapting — removes the entire `L_n`-sample "frozen calibration window" concept.

**Expected weakness:** RLS forgetting factor `λ` trades off adaptivity vs. estimation variance; poor tuning can cause drift or instability. Still fundamentally a linear/quadratic estimator — inherits the source method's Gaussian/LTI assumptions and has no spatial/topology awareness for multi-bus systems.

**Required ablation:** RLS-only (keep old max threshold) vs. CUSUM-only (keep batch `H_y`) vs. combined, to isolate which mechanism drives any gain.

**Estimated cost:** lowest of all candidates — comparable to the source method, marginally higher due to per-step updates instead of one batch solve.

---

## Candidate 2 — STGAT: Spatio-Temporal Graph Attention Residual Detector

**Fixes:** W5 (no spatial structure — the single biggest gap once the testbed becomes multi-bus), W6 (no uncertainty), partially W3 (learned model can be periodically re-fit/fine-tuned rather than frozen forever).

**Architecture:** A Graph Attention Network (GAT) over the bus/line admittance graph, one layer aggregating spatial context across electrically-connected buses, feeding a small per-node temporal encoder (causal 1D-conv, small receptive field) over a sliding measurement window; output is a per-sensor expected-measurement vector. Residual = observed − predicted, aggregated via **learned** attention weights (not a uniform average) into a single detection statistic.

**Core formulation:**
```
h_i^(0) = MLP(window of measurements at bus i)
h_i^(t+1) = Σ_{j ∈ N(i) ∪ {i}} α_ij W h_j^(t),   α_ij = softmax_j(LeakyReLU(aᵀ[Wh_i ‖ Wh_j]))
ẑ_i = readout(h_i^(L))
g_k = Σ_i β_i (z_i,k − ẑ_i,k)²,   β = learned attention weights (not uniform 1/f window average)
```

**Why it should outperform:** the source method treats each measurement independently in time with zero cross-sensor structure — physically wrong for a networked system where a DC/AC power-flow model implies `z = Hx`, i.e., measurements are *not* independent across buses. A replay attack on a subset of sensors should be exposable by its inconsistency with electrically-connected neighbors even when the temporal residual alone is ambiguous. This is the candidate most likely to win on multi-bus systems specifically, and is untestable on the source paper's own 2-state SISO testbed — direct evidence that extending the empirical surface was necessary to expose this axis at all.

**Efficiency:** graphs are small (14–118 nodes); GAT forward pass is cheap; but this candidate is supervised (needs labeled attack/no-attack training data) and requires a training loop unlike the closed-form source method.

**Expected weakness:** needs a topology graph (available in PandaPower testbeds but not always in the field), and needs labeled training data — a real departure from the source method's data-driven-but-unsupervised character. Sensitive to topology changes/reconfiguration unless retrained or made topology-invariant.

**Required ablation:** spatial (GAT) branch alone vs. temporal branch alone vs. combined; learned attention weights vs. uniform window averaging (to prove the attention mechanism itself adds value, not just "more parameters").

**Estimated cost:** moderate — small graph neural net, GPU-optional (feasible on CPU given testbed sizes), heavier than Candidate 1, lighter than Candidate 4.

---

## Candidate 3 — PIRK: Physics-Informed Residual Kalman Filter with Learned Innovation Whitening

**Fixes:** W6 (no calibrated uncertainty), W8 (untested Gaussian/linear assumption), W3 (online via KF recursion).

**Architecture:** A Kalman filter built on the (partially) known DC power-flow measurement Jacobian `H` (available for a power-system testbed, unlike the source paper's genuinely-unknown `A,B,C` — a legitimate, disclosed departure from "fully model-free," argued explicitly in the manuscript as physics availability being a property of the *new* domain, not a weakening of rigor), augmented with a small learned correction network that whitens the innovation sequence so that a chi-squared/Mahalanobis hypothesis test is properly calibrated even when real noise deviates from the assumed Gaussian model.

**Core formulation:**
```
x̂_k|k-1 = A x̂_k-1|k-1,   P_k|k-1 = A P_k-1|k-1 Aᵀ + Q
ν_k = z_k − H x̂_k|k-1                    [innovation / residual]
Σ̂_k = f_θ(ν_{k-w:k-1})  (small learned network correcting the nominal S_k = H P_k|k-1 Hᵀ + R)
T_k = ν_kᵀ Σ̂_k^{-1} ν_k  ~ χ²(p) under H0,  flag if T_k > χ²_{1-α}(p)
```

**Why it should outperform:** gives a *statistically calibrated* threshold with a target false-alarm rate `α` set analytically (chi-squared quantile), rather than the source method's empirical sample max — directly answers W1 and W6 with a formal guarantee under the (learned-corrected) noise model, and reports genuine uncertainty (`Σ̂_k`) rather than a point statistic.

**Efficiency:** KF recursion is O(n²) per step (cheap for these bus-system sizes), the correction network is small; no repeated batch matrix inversion.

**Expected weakness:** requires a topology/physics model (`H`), which reintroduces some of the "needs the system model" limitation the source paper explicitly tried to avoid — the manuscript will need to argue this trade-off honestly (physics-informed vs. fully model-free) rather than hide it. Less naturally suited to the DC-motor reproduction testbed (would need `A,B,C` known — available to the simulator but not the "model-free" spirit of the original paper).

**Required ablation:** physics-only KF (no learned whitening, use nominal `S_k`) vs. learned-whitening-only (replace `H` with a learned linear map) vs. combined.

**Estimated cost:** low-moderate — comparable to Candidate 1, small added cost for the correction network.

---

## Candidate 4 — TCN-CP: Temporal Convolutional Autoencoder with Conformal Anomaly Calibration

**Fixes:** W1 (threshold calibration — with a *distribution-free, finite-sample* guarantee, the strongest fix among all four candidates), W7 (efficiency — TCNs are fully parallel, no recurrence).

**Architecture:** A causal temporal convolutional (TCN) encoder-decoder reconstructs each measurement window; reconstruction error is the anomaly score. Threshold is set via **split conformal prediction**: a held-out calibration set of attack-free scores gives a threshold with a provable finite-sample FAR bound, regardless of the true noise distribution — directly replacing the source method's undocumented, high-variance `sup`-based threshold (quantified as varying >60% across calibration draws in `REPRODUCTION_REPORT.md`) with a method that has a proof.

**Core formulation:**
```
ẑ_{k-w:k} = Decoder(Encoder(z_{k-w:k}))
s_k = ‖z_{k-w:k} − ẑ_{k-w:k}‖²                          [nonconformity score]
threshold q̂ = the ⌈(1-α)(n_cal+1)⌉-th order statistic of {s_i}_{i=1}^{n_cal}   [conformal, distribution-free]
flag if s_k > q̂                                          [guarantees P(false alarm) ≤ α, finite-sample]
```

**Why it should outperform:** solves W1 with a formal, assumption-light guarantee (conformal calibration needs only exchangeability of calibration/test data under H0, not Gaussianity) — strictly stronger than Candidate 3's chi-squared calibration (which still assumes a parametric noise model) and strictly stronger than the source method's `sup`-of-a-sample approach.

**Efficiency:** TCN training/inference is fully parallelizable (no recurrence like an LSTM/GRU would need); conformal calibration is O(n_cal log n_cal) once, O(1) per test point thereafter.

**Expected weakness:** reconstruction-based scoring can under-detect replay attacks specifically *because* replayed data is genuinely real historical data that "looks normal" in isolation — the TCN's receptive field must span the attack-onset discontinuity (boundary between live and replayed data) to catch it, meaning window length is a sensitive hyperparameter and very short attack bursts near the window edge could be missed. This is a real, attack-model-specific risk that must be tested explicitly (robustness experiment: ADR vs. attack duration relative to window length).

**Required ablation:** TCN-AE + empirical threshold (no conformal) vs. linear-AE + conformal vs. full TCN-CP, to separate the encoder architecture's contribution from the calibration mechanism's contribution.

**Estimated cost:** highest among the four — full training loop, largest parameter count, but still small in absolute terms (bus-system windows are low-dimensional compared to typical DL workloads).

---

## Considered but not advanced: cross-testbed meta-learning adapter

A hypernetwork/meta-learning scheme that trains jointly across the 4 bus systems so a new testbed needs only a few-shot calibration was considered (would target W5's generalization gap most directly). **Deprioritized**, not deleted: it adds a second axis of complexity (meta-training loop) on top of whichever spatial/temporal/calibration mechanism wins screening, with uncertain payoff given the 4-testbed scope already agreed, and risks diluting screening signal by conflating "does the architecture work" with "does the meta-learning wrapper work." If Candidate 2 or 4 clears screening comfortably, a lightweight cross-testbed fine-tuning experiment (not a full meta-learning architecture) will be run instead as a generalization experiment, not a 5th architecture.

---

## Ranking (predictive potential / novelty / efficiency / data fit / feasibility / Transactions-level contribution)

| Rank | Candidate | Predictive potential | Novelty | Efficiency | Data fit | Feasibility | Contribution level |
|---|---|---|---|---|---|---|---|
| 1 | **STGAT (C2)** | High on multi-bus (only candidate using topology) | High — spatial axis literally unavailable to the source paper | Moderate | Best — purpose-built for multi-bus | Moderate (needs labels, training loop) | Strong — new mechanism + new empirical surface |
| 2 | **TCN-CP (C4)** | Moderate-high, calibration is provably strong | High — conformal calibration rare in this literature (to be confirmed in the literature-verification step) | High | Good | High | Strong — rigorous calibration contribution |
| 3 | **RLS-CUSUM (C1)** | Moderate — fixes real gaps but stays linear/quadratic | Moderate — well-established individual techniques (RLS, CUSUM), novelty is in combining them to fix this specific paper's gaps | Highest | Good, closest to source method | Highest | Adequate as a strong *baseline upgrade*, weaker as the flagship contribution |
| 4 | **PIRK (C3)** | Moderate, but trades away "model-free" framing | Moderate | High | Good if topology known | Moderate | Adequate, but the model-free retreat is a harder sell for positioning against the source paper |

**Working hypothesis entering screening:** C2 (STGAT) and C4 (TCN-CP) are the strongest flagship candidates — one wins on exploiting the new spatial axis, the other wins on rigorous calibration — and C1 (RLS-CUSUM) is retained primarily as a strong, cheap upgrade path/benchmark in case C2/C4 underperform on validation. This is a hypothesis, not a decision: Section 9's validation-only screening (train/validation data only) will determine the actual ranking, and the final model may combine the winning backbone with the winning calibration layer only if screening evidence supports it.
