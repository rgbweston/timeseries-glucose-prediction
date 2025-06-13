"""
Exploratory Data Analysis for Synthetic OhioT1DM Dataset
========================================================
Generates summary statistics and key visualizations (~6 plots) for the synthetic
CSV-based dataset modeling the OhioT1DM challenge.

Requirements:
- pandas
- numpy
- glob
- matplotlib

Run:
    python eda_synthetic_t1dm.py
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Output directory for plots
# -----------------------------
EDA_OUT = 'eda_synthetic'
os.makedirs(EDA_OUT, exist_ok=True)

# -----------------------------
# 1. Load Metadata
# -----------------------------
meta_path = 'synthetic_ohiot1dm/metadata.csv'
meta = pd.read_csv(meta_path)
print("=== Dataset Description ===")
print(f"Number of patients: {meta.shape[0]}")
print("Patient demographics and record counts:")
print(meta)
print("\n")

# -----------------------------
# 2. Load CSV Data into DataFrame
# -----------------------------
csv_files = glob.glob('synthetic_ohiot1dm/*_train.csv') + glob.glob('synthetic_ohiot1dm/*_test.csv')
all_dfs = []
for path in csv_files:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    patient_id = os.path.basename(path).split('_')[0]
    df['patient_id'] = int(patient_id)
    df['set'] = 'train' if '_train.csv' in path else 'test'
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

print("=== Combined DataFrame ===")
print(df.info())
print(df.describe())
print("Missing values per column:")
print(df.isnull().sum())

# -----------------------------
# 3. Feature Overview
# -----------------------------
print("\n=== Feature List ===")
for col in df.columns:
    print(f"- {col}")

# -----------------------------
# 4. Visualizations (~6 plots)
# -----------------------------

# 4.1 CGM Glucose Distribution
plt.figure(figsize=(8,4))
plt.hist(df['glucose_level'].dropna(), bins=50)
plt.title('CGM Glucose Distribution')
plt.xlabel('Glucose (mg/dL)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{EDA_OUT}/glucose_distribution.png")
plt.show()

# 4.2 Time-Series Glucose for First Patient (train set)
pid0 = meta['id'].iloc[0]
df0 = df[(df['patient_id']==pid0) & (df['set']=='train')].sort_values('timestamp')
plt.figure(figsize=(10,4))
plt.plot(df0['timestamp'], df0['glucose_level'])
plt.title(f'Glucose Time Series for Patient {pid0}')
plt.xlabel('Timestamp')
plt.ylabel('Glucose (mg/dL)')
plt.tight_layout()
plt.savefig(f"{EDA_OUT}/glucose_timeseries_patient_{pid0}.png")
plt.show()

# 4.3 Basal Insulin Rate Distribution
plt.figure(figsize=(8,4))
plt.hist(df['basal'].dropna(), bins=40)
plt.title('Basal Insulin Rate Distribution')
plt.xlabel('Basal Rate (U/hr)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{EDA_OUT}/basal_rate_distribution.png")
plt.show()

# 4.4 Bolus Insulin Distribution
plt.figure(figsize=(8,4))
plt.hist(df['bolus'].dropna(), bins=40)
plt.title('Bolus Insulin Distribution')
plt.xlabel('Bolus Dose (Units)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{EDA_OUT}/bolus_distribution.png")
plt.show()

# 4.5 Meal Carbohydrate Intake Histogram (>0)
meals = df['meal'].dropna()
plt.figure(figsize=(8,4))
plt.hist(meals[meals>0], bins=30)
plt.title('Meal Carbohydrate Intake (>0 only)')
plt.xlabel('Carbs (g)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f"{EDA_OUT}/meal_carbs_histogram.png")
plt.show()

# 4.6 Heart Rate Distribution (Empatica or Basis)
if 'basis_heart_rate' in df.columns:
    plt.figure(figsize=(8,4))
    plt.hist(df['basis_heart_rate'].dropna(), bins=40)
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{EDA_OUT}/heart_rate_distribution.png")
    plt.show()

print("\nEDA complete. Plots saved as PNG files in 'eda_synthetic' folder.")
