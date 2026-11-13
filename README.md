# Calibrated Joint-Residual Replay Attack Detection in Power Systems

Source code of the Physics-Informed Spatio-Temporal Residual Detector (PISTRD), a calibrated joint-residual method for replay-attack detection in networked power systems.

PISTRD combines two residual statistics computed on a dither-encoded measurement stream. The temporal branch fits a joint one-step ridge predictor over all sensors and scores its residual with the full residual covariance (Mahalanobis distance). The spatial branch computes a regularized DC state-estimation residual from the network topology. Both branches are fitted on the training split, calibrated on the validation split, and fused at test time with frozen thresholds. Training is closed form, so the model needs no gradient descent and no random initialization.

## Repository structure

```
configs/pistrd_final.yaml   frozen configuration of the final model
requirements.txt            Python dependencies
src/pipeline/               IEEE 14/39/57/118-bus simulation, replay-attack generation, dataset builder
src/candidates/             PISTRD (pirk_final.py) and the screened candidates (RLS-CUSUM, STGAT, PIRK, TCN-CP)
src/eval/                   shared evaluation harness (thresholds selected on validation only)
src/screening/              validation-only candidate screening
src/benchmarks/             classical, boosting, sequence, and graph benchmarks; lag-feature check
src/ablation/               branch ablation and dither-necessity experiments
src/robustness/             partial (subset-sensor) replay and timing-camouflage experiments
src/generalization/         cross-load-source and AC power-flow validation
src/stats/                  sign test, Fisher combination, Holm correction, win/tie/loss analysis
src/figures/                figure and table generation from the result files
src/reproduction/           reproduction of the data-based Q-learning detector of Qin, Zhang, and Zhao (DC-motor testbed)
```

## Installation

Python 3.11 or later is required.

```bash
git clone https://github.com/Mmassaoudi-cell/PISTRD.git
cd PISTRD
pip install -r requirements.txt
```

## Data

The load data are public and are not redistributed here. Create a `data/` folder in the repository root, or set the `PISTRD_LOAD_DIR` environment variable to the folder that holds the files.

- `PJM_Load_hourly.csv` (required): hourly PJM load, comma-separated, with the columns `Datetime` and `PJM_Load_MW`. It is available from PJM Data Miner (https://dataminer2.pjm.com) and in the Kaggle dataset "Hourly Energy Consumption" (https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).
- `isone_load.csv` and `malaysia_load.csv` (only for `src/generalization/cross_load_source.py`): hourly load from ISO New England (https://www.iso-ne.com/isoexpress) and from the Malaysian grid, semicolon-separated, with the hourly demand in a column named `load`.

The IEEE test networks are loaded from pandapower, so no network files are needed.

## Usage

Run every command from the repository root. All outputs are written to `results/`, `figures/`, and `tables/`, which are created on demand.

```bash
# 1. Build the labeled multi-testbed dataset (IEEE 14/39/57/118-bus)
python src/pipeline/build_dataset.py

# 2. Validation-only candidate screening
python src/screening/run_screening.py
python src/screening/aggregate_screening.py

# 3. Final model and the full benchmark suite on the test split
python src/benchmarks/run_benchmarks.py
python src/benchmarks/rerun_tabular_more_seeds.py
python src/benchmarks/fairness_check.py

# 4. Ablation and robustness
python src/ablation/run_ablation.py
python src/ablation/dither_necessity.py
python src/robustness/partial_replay.py
python src/robustness/camouflaged_replay.py

# 5. Generalization
python src/generalization/cross_load_source.py
python src/generalization/ac_power_flow_validation.py

# 6. Statistics, figures, and tables
python src/stats/wtl_analysis.py
python src/stats/wtl_analysis_v2.py
python src/stats/make_stats_table.py
python src/figures/make_figures.py
python src/figures/make_schematics.py
python src/figures/make_tables.py

# Optional: reproduce the reference Q-learning detector on its DC-motor testbed
python src/reproduction/run_reproduction.py --n_mc 1000 --n_calib 300
python src/reproduction/make_report_table.py
```

Step 1 must be run first, because every later step reads `results/data/`. The benchmark scripts append to their result files, so delete `results/benchmarks/` before a fresh run.

To use the detector on its own:

```python
import sys
sys.path += ["src/candidates", "src/pipeline"]
from pirk_final import PISTRD

model = PISTRD("case14").fit("results/data/case14_train_stats.csv", n_line=15)
model.calibrate("results/data/case14_val_stats.csv")
scores = model.score_split("results/data/case14_test_stats.csv")  # temporal_score, spatial_score, score, flag
```

## Reproducibility notes

- Splits are chronological (60/20/20) with no shuffling. Noise and dither scales, scalers, and model parameters are fitted on the training split only, and every decision threshold is selected on the validation split only.
- PISTRD is deterministic given the data. The stochastic benchmarks use fixed seeds (20 for the tabular models, 3 for the sequence and graph models).
- Attack scenarios, measurement noise, and dither are drawn from generators seeded with a stable CRC32 hash, so a rebuilt dataset is identical across runs and machines. The scenario draws are independent realizations of the protocol described in the paper, so metrics obtained from a fresh build can differ slightly from the published values.

## Key result

On the held-out test split of the IEEE 14-, 39-, 57-, and 118-bus systems, PISTRD reaches a macro-F1 of 0.984 to 0.998 (mean 0.992, against 0.747 for the strongest benchmark) and has the highest mean macro-F1 of all 17 evaluated methods. One fit-validation-test run takes about 3.3 s, which is roughly 5 to 36 times faster than the evaluated deep-learning models.

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

The reference method reproduced in `src/reproduction/` is described in Qin, Zhang, and Zhao, "Data-Based Q-Learning for Replay Attack Detection in Cyber-Physical Systems," IEEE Transactions on Industrial Cyber-Physical Systems, 2025.

## License

Released under the MIT License. See `LICENSE`.

## Contact

Mohamed Massaoudi, mohamed.massaoudi@tamu.edu
