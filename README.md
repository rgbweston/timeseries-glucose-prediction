# Timeseries Glucose Prediction

This repository contains a small synthetic version of the **OhioT1DM** dataset along with example scripts for exploratory data analysis and a toy LSTM model.

## Contents

- `Synthetic_OhioT1DM‑like_Dataset_Generator.py` – script that generates the synthetic dataset.
- `synthetic_ohiot1dm/` – pre-generated CSV files (`<id>_train.csv`, `<id>_test.csv`) plus `metadata.csv` describing the patients.
- `eda_synthetic_t1dm.py` – performs a simple exploratory data analysis and saves plots under `eda_synthetic/`.
- `train_lstm_t1dm.py` – a small PyTorch example of training an LSTM on a synthetic sine-wave time series.

## Requirements

The scripts rely on standard Python scientific packages:

- `pandas` and `numpy`
- `matplotlib`
- `torch` and `scikit-learn` (for the LSTM demo)

Install them with pip if needed:

```bash
pip install pandas numpy matplotlib torch scikit-learn
```

## Generating the dataset

Run the generator to create the CSV files in `synthetic_ohiot1dm/`:

```bash
python Synthetic_OhioT1DM‑like_Dataset_Generator.py
```

This will produce one train and one test file per patient plus a `metadata.csv` summary. A snippet of the CSV format is shown below:

```csv
timestamp,glucose_level,finger_stick,basal,temp_basal,bolus,meal,sleep,sleep_quality,work,work_intensity,stressors,hypo_event,illness,exercise,exercise_intensity,basis_heart_rate,basis_gsr,basis_skin_temperature,acceleration,basis_steps,basis_air_temperature
2025-01-01 00:00:00,115.0,,0.97,0.0,0.0,0.0,1,1.0,0,0,0,0,0,0,0,78.0454603754775,1.0918536979859716,90.74982913483234,0.12568247179103736,,
2025-01-01 00:05:00,108.6,116.44326550087061,0.95,0.0,0.0,0.0,1,1.0,0,0,0,0,0,0,0,77.5970016352206,0.7745261008515978,89.682374688316,0.2755479498902593,,
```

Metadata example:

```csv
id,gender,age_range,pump,band,cohort,train_records,test_records
540,male,20-40,630G,Empatica,2020,11947,2884
544,male,40-60,530G,Empatica,2020,10623,2704
```

## Exploratory data analysis

To generate basic distribution plots run:

```bash
python eda_synthetic_t1dm.py
```

The figures will be written to the `eda_synthetic/` directory.

## LSTM example

`train_lstm_t1dm.py` demonstrates a simple LSTM for sequence forecasting on a toy sine-wave dataset. It is intended as a starting point for building more advanced glucose prediction models.

Run it directly:

```bash
python train_lstm_t1dm.py
```

This will print training loss, evaluate on a held-out set and show a plot of predicted vs true values.

## Repository structure

```
.
├── Synthetic_OhioT1DM‑like_Dataset_Generator.py
├── eda_synthetic_t1dm.py
├── train_lstm_t1dm.py
├── synthetic_ohiot1dm/
│   ├── 540_train.csv
│   ├── ...
│   ├── metadata.csv
└── eda_synthetic/
    ├── glucose_distribution.png
    └── ...
```

Feel free to modify the scripts to suit your experiments or extend the dataset.
