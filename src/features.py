"""Shared data loading and feature preparation for the NIJ Recidivism Challenge data.

The data describe 25,835 people released from Georgia prisons to parole supervision (2013-2015),
with whether they were re-arrested in years 1, 2 and 3 after release.

Rule used throughout: only information known AT RELEASE is used. The dataset also has
measurements taken during supervision (drug tests, violations, program attendance, employment).
Those are left out, because they are recorded after the outcome window starts, and people
re-arrested early have less time to accumulate them. Using them would leak the answer into the inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nij_recidivism_full.csv.gz"
OUTCOME = "Recidivism_Arrest_Year1"

AGE_MID = {"18-22": 20, "23-27": 25, "28-32": 30, "33-37": 35, "38-42": 40, "43-47": 45, "48 or older": 50}
PRISON_YEARS = {"Less than 1 year": 0.5, "1-2 years": 1.5, "Greater than 2 to 3 years": 2.5, "More than 3 years": 4.0}
EDUCATION = {"Less than HS diploma": 0, "High School Diploma": 1, "At least some college": 2}
SUP_LEVEL = {"Standard": 0, "Specialized": 1, "High": 2}

PRIOR_COUNTS = [
    "Dependents", "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd", "Prior_Arrest_Episodes_Violent",
    "Prior_Arrest_Episodes_Property", "Prior_Arrest_Episodes_Drug", "Prior_Arrest_Episodes_PPViolationCharges",
    "Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd", "Prior_Conviction_Episodes_Prop",
    "Prior_Conviction_Episodes_Drug",
]
PRIOR_FLAGS = [
    "Prior_Arrest_Episodes_DVCharges", "Prior_Arrest_Episodes_GunCharges", "Prior_Conviction_Episodes_Viol",
    "Prior_Conviction_Episodes_PPViolationCharges", "Prior_Conviction_Episodes_DomesticViolenceCharges",
    "Prior_Conviction_Episodes_GunCharges", "Prior_Revocations_Parole", "Prior_Revocations_Probation",
]
CONDITIONS = ["Condition_MH_SA", "Condition_Cog_Ed", "Condition_Other"]


def _count(v) -> float:
    """'10 or more' -> 10, '3' -> 3."""
    if pd.isna(v):
        return np.nan
    m = re.match(r"\s*(\d+)", str(v))
    return float(m.group(1)) if m else np.nan


def _flag(v) -> float:
    if pd.isna(v):
        return np.nan
    return float(str(v).strip().lower() in {"true", "1"})


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def build_features(df: pd.DataFrame, include_demographics: bool = False,
                   include_conditions: bool = True) -> pd.DataFrame:
    """Numeric feature matrix built only from information known at release."""
    X = pd.DataFrame(index=df.index)
    X["age_at_release"] = df["Age_at_Release"].map(AGE_MID)
    X["prison_years"] = df["Prison_Years"].map(PRISON_YEARS)
    X["education"] = df["Education_Level"].map(EDUCATION)
    X["supervision_level"] = df["Supervision_Level_First"].map(SUP_LEVEL)
    X["supervision_level_missing"] = X["supervision_level"].isna().astype(float)
    X["supervision_level"] = X["supervision_level"].fillna(1)
    X["state_risk_score"] = df["Supervision_Risk_Score_First"]
    X["state_risk_score_missing"] = X["state_risk_score"].isna().astype(float)
    X["state_risk_score"] = X["state_risk_score"].fillna(df["Supervision_Risk_Score_First"].median())
    X["gang_affiliated"] = df["Gang_Affiliated"].map(_flag)
    X["gang_affiliated_missing"] = X["gang_affiliated"].isna().astype(float)
    X["gang_affiliated"] = X["gang_affiliated"].fillna(0)
    for c in PRIOR_COUNTS:
        X[c.lower()] = df[c].map(_count)
    for c in PRIOR_FLAGS:
        X[c.lower()] = df[c].map(_flag)
    offense = df["Prison_Offense"].fillna("Missing")
    X = X.join(pd.get_dummies(offense, prefix="offense", dtype=float))
    X = X.join(pd.get_dummies(df["Residence_PUMA"].astype(str), prefix="puma", dtype=float))
    if include_conditions:
        for c in CONDITIONS:
            X[c.lower()] = df[c].map(_flag)
    if include_demographics:
        X["male"] = (df["Gender"] == "M").astype(float)
        X["black"] = (df["Race"] == "BLACK").astype(float)
    return X.astype(float)
