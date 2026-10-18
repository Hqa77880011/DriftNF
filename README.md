# DriftNF

Implementation of **Risk-calibrated pre-commit auditing of unlabeled neuro-fuzzy updates under concept drift.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Use one entity per NPZ with chronological `train: (T_train,D)` and `test: (T_test,D)`. Initialization data must be normal. A separate evaluation NPZ contains `label: (T_test,)`; optional `affected_sensor: (T_test,D)` and `fault_mask: (T_test,)` enable explanation and update-safety metrics. Public sources are [SWaT/WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/), [SMAP/MSL](https://github.com/khundman/telemanom) and [SMD](https://github.com/NetManAIOps/OmniAnomaly). Each entity is fitted independently. The preparer uses a chronological 80/20 normal split, training-only median/MAD scaling, constant-channel removal and causal filling.

```bash
python -m driftnf prepare --input data/entity.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m driftnf train --config config.json --output runs/full
python -m driftnf run --config config.json --output runs/full
```

`config.json` records the method settings and seed `7`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m driftnf evaluate --input data/labels.npz --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `frozen`, `direct`, `raw-means`, `no-anchor`, `no-fault`, `no-recurrence`, `descriptor-only`, `occupancy-only`, `no-counterfactual`, `dense`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m driftnf prepare --input data/entity.npz --config config.json --output runs/frozen --variant frozen
python -m driftnf train --config config.json --output runs/frozen --variant frozen
python -m driftnf run --config config.json --output runs/frozen --variant frozen
python -m driftnf evaluate --input data/labels.npz --config config.json --output runs/frozen --variant frozen
```

## Outputs

All main outputs are written below `runs/full/`.

`predictions.csv` contains scores, alarms, top sensors, rule/signature counts, explanation diagnostics and audit-inclusive latency. `audit.json` records evidence, candidate scores, bounds, slack and vetoes. `metrics.csv` reports Affiliation F1, VUS-PR, FAR, deletion fidelity, ranking stability and P95 cost. Supplied intervention annotations enable top-3 recall, anomaly retention, action correctness, veto precision and certificate ECE. Evaluation labels do not enter the monitor.

```bash
python -m driftnf plot --config config.json --output runs/full
```
