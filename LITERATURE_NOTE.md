# Literature / Novelty Verification Note (preliminary — Step 23 will finalize)

Quick web-search check (Aug 2026) against the two mechanisms most central to `MODEL_CANDIDATES.md` C2 and C4, done before final-model selection so the manuscript's novelty framing isn't decided in a vacuum.

## Graph attention networks for grid attack detection — NOT novel in isolation
Active, recent literature already exists:
- "A Hierarchical Spatio-Temporal Graph Attention Network for False Data Injection Attack Detection in Smart Grids" (HST-GAT, *Processes*, Feb 2026)
- "A graph attention network-based framework for dynamic topology change-aware FDIA detection in smart grids" (AST-TGT, Sept 2025)
- "Detection of Coordinated False Data Injection Attacks... Using Graph Attention Network" (2026)
- "Graph attention and Kolmogorov–Arnold network based smart grids intrusion detection" (*Sci. Reports*, 2025)

**Implication:** Candidate 2 (STGAT) cannot be positioned as "first GNN for grid attack detection" — that claim would be false. GNNs for **FDIA** detection in smart grids are an established, fairly crowded sub-area.

## Conformal calibration for CPS anomaly detection — also emerging, not unprecedented
- "Conformal machine learning for reliable anomaly detection in industrial cyber-physical systems" (*Reliability Engineering & System Safety*, Feb 2026)
- "CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent Conformal Risk Control" (2026)
- "Conformance-Driven Anomaly Detection for Cyber-Physical Systems" (CPS-IoT Week 2025)
- "Out-of-Distribution Detection in Dependent Data for Cyber-Physical Systems with Conformal Guarantees" (2024)

**Implication:** Candidate 4 (TCN-CP)'s calibration mechanism is not unprecedented either — conformal calibration for CPS/streaming anomaly detection is an active 2024-2026 direction.

## Where the actual gap sits
Both search sets are dominated by **FDIA** (false-data-injection) detection — a much larger, more ML-native literature. Searches turned up essentially nothing specifically combining GNN or conformal calibration with **replay-attack** detection specifically. The replay-attack sub-literature (which the source paper itself surveys in its intro: watermarking [5]-[7], coding/encoding [8]-[11], entropy/subspace/data-driven [12]-[19]) has stayed comparatively closer to classical control/detection theory and has not absorbed these ML tools the way FDIA detection has. This is the honest, narrow gap to claim:

> "To our knowledge, existing replay-attack-detection work for cyber-physical/power systems has not incorporated topology-aware graph learning or distribution-free calibrated thresholding — tools already established in the adjacent FDIA-detection literature but not yet applied to the replay-attack sub-problem, which has a structurally different attacker model (replayed data is genuinely real historical telemetry, not synthetically fabricated values) and a different baseline literature (watermarking/encoding-based rather than pure ML classification)."

This is the honest framing to carry into the manuscript's Introduction/Related-Work — **not** "first GNN for grid security" or "first conformal CPS detector," both of which a reviewer could immediately falsify. Final wording depends on which candidate wins screening; this note is input to that final positioning, not the final claim itself. A full pass (IEEE Xplore / more targeted queries once the final model is fixed) is still required at Step 23 before the manuscript is written.
