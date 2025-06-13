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
