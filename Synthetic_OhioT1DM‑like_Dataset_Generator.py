"""
Synthetic OhioT1DM‑like Dataset Generator
=========================================
Generates a fully‑featured synthetic replica of the OhioT1DM dataset that
conforms to:
 • Patient roster & demographics (Table 1 in Marling & Bunescu 2020)
 • Record counts & cohort splits (Table 2)
 • Variable list & temporal resolution (5‑minute CGM; 1‑ or 5‑minute sensor
   channels depending on band type)
 • Train/test separation into CSV files (one train + one test file per patient),
   plus a metadata.csv summary

The output folder structure mirrors the real dataset for easy drop‑in use.

Usage
-----
$ python synthetic_ohiot1dm_generator.py  # writes ./synthetic_ohiot1dm/

Requirements: pandas, numpy
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- CONFIG ------------------------------------ #
ROOT_OUT   = Path("synthetic_ohiot1dm")  # where CSV files & metadata are written
START_DATE = datetime(2025, 1, 1, 0, 0)    # anchor date for timestamps
FREQ       = "5min"                     # data frequency for CGM & sensors
SEED       = 42
np.random.seed(SEED)
random.seed(SEED)

# Patient metadata straight from Table 1 & Table 2
PATIENTS = [
    {"id":540,"gender":"male","age_range":"20-40","pump":"630G","band":"Empatica","cohort":2020,"train":11947,"test":2884},
    {"id":544,"gender":"male","age_range":"40-60","pump":"530G","band":"Empatica","cohort":2020,"train":10623,"test":2704},
    {"id":552,"gender":"male","age_range":"20-40","pump":"630G","band":"Empatica","cohort":2020,"train":9080,"test":2352},
    {"id":567,"gender":"female","age_range":"20-40","pump":"630G","band":"Empatica","cohort":2020,"train":10858,"test":2377},
    {"id":584,"gender":"male","age_range":"40-60","pump":"530G","band":"Empatica","cohort":2020,"train":12150,"test":2653},
    {"id":596,"gender":"male","age_range":"60-80","pump":"530G","band":"Empatica","cohort":2020,"train":10877,"test":2731},
    {"id":559,"gender":"female","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":10796,"test":2514},
    {"id":563,"gender":"male","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":12124,"test":2570},
    {"id":570,"gender":"male","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":10982,"test":2745},
    {"id":575,"gender":"female","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":11866,"test":2590},
    {"id":588,"gender":"female","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":12640,"test":2791},
    {"id":591,"gender":"female","age_range":"40-60","pump":"530G","band":"Basis","cohort":2018,"train":10847,"test":2760},
]

# --------------------------- DATA GENERATION fns --------------------------- #

def generate_glucose(n:int)->np.ndarray:
    """Simulate CGM glucose with circadian + noise."""
    x = np.linspace(0,2*np.pi,n)
    base = 110 + 15 * np.sin(x)
    noise = np.random.normal(0,10,n)
    return np.clip(base + noise,55,300)


def random_events(n:int, prob:float, low:float, high:float)->np.ndarray:
    """Sparse events like meals, bolus."""
    arr = np.zeros(n)
    mask = np.random.rand(n)<prob
    arr[mask] = np.random.uniform(low,high,mask.sum())
    return arr


def band_sensors(n:int, band:str)->dict[str,np.ndarray]:
    """Sensor band readings."""
    hr = np.random.normal(75,8,n)
    gsr= np.random.normal(0.8,0.2,n)
    temp=np.random.normal(90,1.5,n)
    acc= np.random.normal(0.3,0.1,n) if band=='Empatica' else np.full(n,np.nan)
    steps=np.random.poisson(5,n)        if band=='Basis'    else np.full(n,np.nan)
    air =np.random.normal(70,3,n)       if band=='Basis'    else np.full(n,np.nan)
    return {
        'basis_heart_rate':hr,
        'basis_gsr':gsr,
        'basis_skin_temperature':temp,
        'acceleration':acc,
        'basis_steps':steps,
        'basis_air_temperature':air,
    }

# ------------------------ GENERATION & OUTPUT ------------------------------ #

def generate_patient(patient:dict)->pd.DataFrame:
    total = patient['train']+patient['test']
    times = pd.date_range(START_DATE,periods=total,freq=FREQ)
    glus = generate_glucose(total)
    meals= random_events(total,0.013,30,70)
    bolus= meals/10
    basal=1.0 + np.random.normal(0,0.05,total)
    tempb=np.zeros(total)
    finger=np.where(np.random.rand(total)<0.02,glus+np.random.normal(0,5,total),np.nan)

    sleep = ((times.hour<6)|(times.hour>=22)).astype(int)
    sleep_q = np.where(sleep,np.random.randint(1,4,total),np.nan)
    work  = ((times.dayofweek<5)&(times.hour>=8)&(times.hour<17)).astype(int)
    work_int=np.where(work,np.random.randint(3,7,total),0)
    stress=np.random.rand(total)<0.002
    hypo  =(glus<70)
    ill   =np.random.rand(total)<0.0005
    ex_min=np.where(np.random.rand(total)<0.003,np.random.randint(10,60,total),0)
    ex_int=np.where(ex_min,np.random.randint(4,9,total),0)

    sensors=band_sensors(total,patient['band'])

    df=pd.DataFrame({
        'timestamp':times,
        'glucose_level':np.round(glus,1),
        'finger_stick':finger,
        'basal':np.round(basal,2),
        'temp_basal':tempb,
        'bolus':bolus,
        'meal':meals,
        'sleep':sleep,
        'sleep_quality':sleep_q,
        'work':work,
        'work_intensity':work_int,
        'stressors':stress.astype(int),
        'hypo_event':hypo.astype(int),
        'illness':ill.astype(int),
        'exercise':ex_min,
        'exercise_intensity':ex_int,
        **sensors
    })
    return df


def main():
    ROOT_OUT.mkdir(exist_ok=True)
    meta=[]
    for p in PATIENTS:
        df=generate_patient(p)
        train=df.iloc[:p['train']]
        test =df.iloc[p['train']:p['train']+p['test']]
        train.to_csv(ROOT_OUT/f"{p['id']}_train.csv",index=False)
        test .to_csv(ROOT_OUT/f"{p['id']}_test.csv" ,index=False)
        meta.append({
            'id':p['id'],'gender':p['gender'],'age_range':p['age_range'],
            'pump':p['pump'],'band':p['band'],'cohort':p['cohort'],
            'train_records':len(train),'test_records':len(test)
        })
        print(f"Patient {p['id']} OK (train={len(train)}, test={len(test)})")
    pd.DataFrame(meta).to_csv(ROOT_OUT/'metadata.csv',index=False)
    print(f"\nMetadata + CSV files written to {ROOT_OUT.resolve()}")

if __name__=='__main__':
    main()
