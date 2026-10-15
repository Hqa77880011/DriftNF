# DriftNF

Implementation of **Risk-calibrated pre-commit auditing of unlabeled neuro-fuzzy updates under concept drift.**

## Install

Use Python 3.11 in an isolated environment. From the repository root:

```bash
python -m pip install -e .
```

## Data

Use one entity per NPZ with chronological `train: (T_train,D)` and `test: (T_test,D)`. Initialization data must be normal. A separate evaluation NPZ contains `label: (T_test,)`; optional `affected_sensor: (T_test,D)` and `fault_mask: (T_test,)` enable explanation and update-safety metrics. Public sources are [SWaT/WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/), [SMAP/MSL](https://github.com/khundman/telemanom) and [SMD](https://github.com/NetManAIOps/OmniAnomaly). Each entity is fitted independently. The preparer uses a chronological 80/20 normal split, training-only median/MAD scaling, constant-channel removal and causal filling.

## Workflow

```bash
python -m driftnf prepare --input data/entity.npz
python -m driftnf train
python -m driftnf run
python -m driftnf evaluate --input data/labels.npz
python -m driftnf plot
```

Main commands accept `--config config.json` and `--output runs/default`. Use a separate output directory for each seed and condition. Its `settings.json` records the resolved configuration.

## Configuration

`window`, `buffer` and `persistence` set causal context and drift persistence. `max_rules` and `regimes` bound retained state. `maximum_contamination` and `sample_risk` control candidate admission. `alpha` and `audit_scale` determine one-sided clipped audit bounds. `fit_margin`, `anchor_damage` and `fault_margin` are fixed certificate thresholds; `online_rate` controls tentative parameter interpolation. Common training fields `epochs`, `batch_size`, `learning_rate`, `width` and `seed` control optimization, model size and randomness where present. Inspect `config.json` for defaults.

## Outputs

`predictions.csv` contains scores, alarms, top sensors, rule/signature counts, explanation diagnostics and audit-inclusive latency. `audit.json` records evidence, candidate scores, bounds, slack and vetoes. `metrics.csv` reports Affiliation F1, VUS-PR, FAR, deletion fidelity, ranking stability and P95 cost. Supplied intervention annotations enable top-3 recall, anomaly retention, action correctness, veto precision and certificate ECE. Evaluation labels do not enter the monitor.

## Method and comparisons

The causal TCN has three left-padded residual blocks. Gaussian rules use straight-through sparse antecedents and local next-descriptor predictors. Online updates fit weighted ridge candidates, including regime recall, before auditing recent fit, anchor damage and pseudo-fault separation. A veto preserves both deployed parameters and the regime ledger. Affiliation and VUS use [Affiliation](https://github.com/ahstat/affiliation-metrics-py) and [TimeEval](https://timeeval.readthedocs.io/en/latest/api/timeeval.metrics.html).

Controls: `full`, `frozen`, `direct`, `raw-means`, `no-anchor`, `no-fault`, `no-recurrence`, `descriptor-only`, `occupancy-only`, `no-counterfactual`, `dense`. Select a control with `--variant` and keep the same data and seed. Changes to trainable objectives or architectures require a separate training run.

The `driftnf.compare` command averages repeated rows within independent units, then reports means, bootstrap intervals, paired Wilcoxon comparisons, rank-biserial effects and Holm adjustment per metric. For example:

```bash
python -m driftnf.compare --inputs "runs/*/metrics.csv" --reference full --metrics affiliation_f1 far --unit seed
```

