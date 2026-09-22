# Finding: The Source Paper's Dither-Encoding Mechanism Is Necessary, Not Redundant

`DATA_AUDIT.md` flagged an open question when the multi-bus extension was designed: since real PJM load data gives the testbeds genuine non-stationary temporal structure (unlike the source paper's deterministic LQR-decaying toy trajectory), would a plain residual model already detect whole-vector replay attacks from that non-stationarity alone, making the source paper's stochastic dither-encoding/decoding scheme (Eq. 19–20) redundant in this richer setting?

**Answer: No — decisively and consistently across all 4 testbeds.** `results/ablation/dither_necessity_results.csv`:

| Testbed | Full (plain) | Full (encoded) | Temporal-only (plain) | Temporal-only (encoded) | Spatial-only (plain) | Spatial-only (encoded) |
|---|---|---|---|---|---|---|
| case14 | 0.526 | 0.984 | 0.517 | 0.988 | 0.497 | 0.953 |
| case39 | 0.484 | 0.994 | 0.488 | 0.995 | 0.490 | 0.982 |
| case57 | 0.494 | 0.994 | 0.501 | 0.995 | 0.490 | 0.990 |
| case118 | 0.485 | 0.998 | 0.502 | 0.998 | 0.487 | 0.998 |

Without dither encoding, **every variant collapses to macro-F1 ≈ 0.48–0.53 — statistically indistinguishable from a majority-class/random detector** — on every testbed, regardless of using the joint temporal predictor, the physics-informed spatial residual, or their fusion. With dither encoding, every variant jumps to 0.95–0.999.

## Why: the original hypothesis was wrong about the actual attack timing

The hypothesis in `DATA_AUDIT.md` assumed replaying old load data would create an obvious mismatch with the current time's load level (e.g., "replaying an overnight low-load window during a midday peak"). In practice, `attacks.py` places each replay only 5–30 hours after its own recorded window — well within the timescale over which real hourly load is highly autocorrelated (similar time-of-day effects, gradual daily trends). A one-step (or even joint-covariance) residual model sees a plausible, smoothly-continued value either way and cannot tell "genuinely current" from "recently recorded" data apart on typical load dynamics alone. The **spatial** branch fails for a different, cleaner reason (argued analytically in `MODEL_SELECTION_REPORT.md`): a whole-vector replay is, by construction, a fully valid historical physical snapshot, so it is spatially self-consistent and invisible to any single-instant physics check *regardless of encoding*.

What actually carries the entire detection signal is the dither-mismatch term itself (`d(ridx) − d(k)`, variance `2Σ_d`) that only appears in the residual when the current decode key doesn't match the key used at record time — exactly the mechanism the source paper introduced for its toy system, and confirmed here to be doing essentially all of the work at 4 much larger, more realistic scales.

## Implication for positioning (revises the framing in `DATA_AUDIT.md`)

This is a **validating**, not undermining, finding for the source paper: its core innovation (stochastic measurement encoding to unmask replay) is shown here to generalize and remain essential far beyond its original 2-state toy testbed — realistic load non-stationarity does *not* make it obsolete. The manuscript's contribution is therefore correctly scoped as: keep the encoding mechanism (it is load-bearing and not something we are replacing), and fix the *other* diagnosed weaknesses around it — the fragile max-of-sample threshold (W1), the lack of joint/spatial modeling (W5), and the lack of an online/adaptive option (W3) — rather than a claim of having removed the need for encoding altogether.

This also means: any future benchmark or ablation number computed on the "plain" stream should be read as a synthetic best case for testing threshold/robustness null behavior, not as evidence about real-world achievable performance without encoding.
