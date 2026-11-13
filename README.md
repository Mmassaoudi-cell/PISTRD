# Calibrated Joint-Residual Replay Attack Detection in Power Systems

Implementation of the Physics-Informed Spatio-Temporal Residual Detector (PISTRD), a calibrated joint-residual method for replay-attack detection in networked power systems.

PISTRD combines two residual statistics computed on a dither-encoded measurement stream. The temporal branch fits a joint one-step ridge predictor over all sensors and scores its residual with the full residual covariance (Mahalanobis distance). The spatial branch computes a regularized DC state-estimation residual from the network topology. Both branches are fitted on the training split, calibrated on attack-free validation samples, and fused at test time with frozen thresholds. Training is closed form, so the model needs no gradient descent and no random initialization.

## Contents

- `pistrd.py`: the complete model (temporal branch, spatial branch, validation calibration, and fused decision) and the DC observation matrix of the IEEE test networks.
- `LICENSE`: MIT License.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/Mmassaoudi-cell/PISTRD.git
cd PISTRD
pip install numpy scikit-learn pandapower
```

## Usage

A split is a list of chronological sequences, one array of shape `(T, m)` per scenario, where `m` is the number of measured line flows in MW. Validation labels are 0 for normal samples and 1 for replayed samples.

```python
from pistrd import PISTRD, line_observation_matrix

H = line_observation_matrix("case14")          # also "case39", "case57", "case118"

model = PISTRD(H, alpha=0.05, ridge_alpha=1.0, cov_reg=1e-3, spatial_reg=1e-2)
model.fit(train_sequences)                     # training split only
model.calibrate(val_sequences, val_labels)     # validation split only

out = model.score(test_sequences)
out["temporal_score"]                          # Mahalanobis one-step residual
out["spatial_score"]                           # DC state-estimation residual
out["score"]                                   # fused score, alarm when > 1
out["flag"]                                    # 0/1 alarm
```

The default arguments are the frozen configuration used in the paper. The total false-alarm target `alpha` is split equally between the two branches.

## Key result

On the held-out test split of the IEEE 14-, 39-, 57-, and 118-bus systems, driven by real hourly load profiles, PISTRD reaches a macro-F1 of 0.984 to 0.998, with a mean of 0.992.

## Citation

If you use this code, please cite:

```bibtex
@article{massaoudi_pistrd,
  author = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  title  = {Calibrated Joint-Residual Replay Attack Detection in Power Systems},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.

## Contact

Mohamed Massaoudi, mohamed.massaoudi@tamu.edu
