from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, time
import pandas as pd
import random
import re
from contextlib import asynccontextmanager

CSV_FILE = "trains_dataset_with_extras_corrected.csv"
df_trains = pd.DataFrame()

# ---------------- Helper Functions ---------------- #

def parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return None

def is_cert_valid(cert_date_str: str, ref_date: datetime) -> bool:
    cert_date = parse_date_safe(cert_date_str)
    return cert_date is not None and cert_date >= ref_date

def contains_ignore_case(text: Optional[str], pattern: str) -> bool:
    return bool(text and re.search(pattern, text, re.IGNORECASE))

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

def is_rush_hour(dt: datetime) -> bool:
    t = dt.time()
    return (time(7,0) <= t <= time(10,0)) or (time(17,0) <= t <= time(20,0))

# ---------------- Scoring Logic ---------------- #

def compute_scores_for_row(row: pd.Series) -> pd.Series:
    date_dt = parse_date_safe(row['date']) or datetime.now()

    rolling_valid = is_cert_valid(row['rolling_stock_cert_valid_till'], date_dt)
    signalling_valid = is_cert_valid(row['signalling_cert_valid_till'], date_dt)
    telecom_valid = is_cert_valid(row['telecom_cert_valid_till'], date_dt)
    all_certs_valid = rolling_valid and signalling_valid and telecom_valid

    pfs = row.get('predicted_failure_score') or 0.5
    jobcard_count = row.get('jobcard_open_count') or 0
    cleaning_required = bool(row.get('cleaning_slot_required'))
    cleaning_available = bool(row.get('cleaning_slot_available'))
    mileage_last_7days = row.get('mileage_last_7days_km') or 0
    mileage_total = row.get('mileage_total_km') or 0
    branding_hours = row.get('branding_priority_hours') or 0
    depot_status = row.get('depot_status') or ""
    status_override = row.get('status_override') or ""
    brand_sponsor = row.get('brand_sponsor')

    fit_score = 0
    if all_certs_valid: fit_score += 30
    if pfs < 0.25: fit_score += 20
    elif pfs < 0.5: fit_score += 10
    if jobcard_count == 0: fit_score += 10
    elif jobcard_count <= 2: fit_score += 5
    if (not cleaning_required) or (cleaning_required and cleaning_available): fit_score += 10
    if mileage_last_7days < 400: fit_score += 10
    elif mileage_total < 60000: fit_score += 5
    if branding_hours > 0: fit_score += 10
    if depot_status in ["Depot A", "Depot B"]: fit_score += 10

    fit_score = min(fit_score, 100)
    hold_in_override = contains_ignore_case(status_override, "hold")

    fit_flag = 1 if (fit_score >= 60 and all_certs_valid and pfs < 0.6 and not hold_in_override) else 0

    score = fit_score / 100.0
    score -= 0.5 * pfs
    score -= 0.05 * jobcard_count
    if cleaning_required and not cleaning_available: score -= 0.2
    if is_rush_hour(date_dt) and brand_sponsor and branding_hours > 0: score += 0.15
    if depot_status == "Running": score -= 0.3
    if contains_ignore_case(status_override, "force_induct"): score += 0.2
    if hold_in_override: score = 0

    return pd.Series({
        "fit_score_100": min(fit_score, 100),
        "fit_flag": fit_flag,
        "score_card": clamp(score, 0, 1)
    })

def update_all_scores():
    global df_trains
    scores_df = df_trains.apply(compute_scores_for_row, axis=1)
    df_trains.update(scores_df)

def generate_predicted_failure_scores():
    global df_trains
    df_trains['predicted_failure_score'] = df_trains['predicted_failure_score'].apply(
        lambda x: x if pd.notna(x) else random.uniform(0,1)
    )

def explain_train_induction(row: pd.Series) -> List[str]:
    reasons = []
    date_dt = parse_date_safe(row['date']) or datetime.now()
    if not is_cert_valid(row['rolling_stock_cert_valid_till'], date_dt): reasons.append("Rolling stock cert expired")
    if not is_cert_valid(row['signalling_cert_valid_till'], date_dt): reasons.append("Signalling cert expired")
    if not is_cert_valid(row['telecom_cert_valid_till'], date_dt): reasons.append("Telecom cert expired")
    if (row.get('jobcard_open_count') or 0) > 2: reasons.append("Too many open jobcards")
    if (row.get('predicted_failure_score') or 0.5) >= 0.6: reasons.append("High failure risk")
    if row.get('cleaning_slot_required') and not row.get('cleaning_slot_available'): reasons.append("Cleaning unavailable")
    if contains_ignore_case(row.get('status_override'), "hold"): reasons.append("Status override: hold")
    return reasons

# ---------------- API Models ---------------- #

class UpdateStatusRequest(BaseModel):
    train_id: str
    status: str

class AddNoteRequest(BaseModel):
    train_id: str
    note: str

class DispatchRequest(BaseModel):
    station_id: str
    current_demand: int
    threshold: int

# ---------------- Lifespan Handler ---------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df_trains
    df_trains = pd.read_csv(CSV_FILE, dtype=str)
    numeric_cols = [
        'jobcard_open_count','branding_priority_hours','mileage_last_7days_km',
        'mileage_total_km','cleaning_slot_required','cleaning_slot_available',
        'stabling_swap_cost','predicted_failure_score','last_maintenance_days_ago',
        'fit_score_100','fit_flag','score_card'
    ]
    for col in numeric_cols:
        if col in df_trains.columns:
            df_trains[col] = pd.to_numeric(df_trains[col], errors='coerce')
    df_trains.fillna({"notes":"", "status_override":"", "brand_sponsor":"", "depot_status":""}, inplace=True)
    generate_predicted_failure_scores()
    update_all_scores()
    yield
    # optional: code here runs on shutdown

app = FastAPI(title="KMRL AI-Driven Train Induction Planning & Scheduling API", lifespan=lifespan)

# ---------------- API Endpoints ---------------- #

@app.get("/trains")
def get_trains():
    return df_trains.to_dict(orient="records")

@app.get("/trains/{train_id}")
def get_train(train_id: str):
    train = df_trains[df_trains['train_id'] == train_id]
    if train.empty:
        raise HTTPException(status_code=404, detail="Train not found")
    return train.iloc[0].to_dict()

@app.post("/rank")
def rank_trains(top_n: int = 5):
    df_fit = df_trains[df_trains['fit_flag'] == 1].copy().sort_values(by='score_card', ascending=False)
    inducted = df_fit.head(top_n)
    standby = df_fit.iloc[top_n:]
    not_fit = df_trains[df_trains['fit_flag'] == 0]

    return {
        "inducted": [
            {**row.to_dict(), "explanations": explain_train_induction(row)} for _, row in inducted.iterrows()
        ],
        "standby": [
            {**row.to_dict(), "explanations": explain_train_induction(row)} for _, row in standby.iterrows()
        ],
        "not_fit": [
            {**row.to_dict(), "explanations": explain_train_induction(row)} for _, row in not_fit.iterrows()
        ]
    }

@app.post("/dispatch")
def dispatch_train(req: DispatchRequest):
    global df_trains
    if req.current_demand <= req.threshold:
        return {"message": "No dispatch needed"}
    standby = df_trains[(df_trains['fit_flag'] == 1) & (df_trains['depot_status'] != "Running")]
    if standby.empty:
        return {"message": "No standby trains available"}
    train_to_dispatch = standby.sort_values(by='score_card', ascending=False).iloc[0]
    idx = df_trains.index[df_trains['train_id'] == train_to_dispatch['train_id']].tolist()[0]
    df_trains.at[idx, 'depot_status'] = "Running"
    update_all_scores()
    return {"message": f"Dispatched train {train_to_dispatch['train_id']} due to high demand"}

@app.post("/update_status")
def update_status(req: UpdateStatusRequest):
    global df_trains
    idx = df_trains.index[df_trains['train_id'] == req.train_id].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail="Train not found")
    df_trains.at[idx[0], 'depot_status'] = req.status
    update_all_scores()
    return {"message": f"Updated train {req.train_id} to status {req.status}"}

@app.post("/add_note")
def add_note(req: AddNoteRequest):
    global df_trains
    idx = df_trains.index[df_trains['train_id'] == req.train_id].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail="Train not found")
    current_notes = df_trains.at[idx[0], 'notes'] or ""
    df_trains.at[idx[0], 'notes'] = (current_notes + "\n" + req.note).strip()
    return {"message": f"Note added to train {req.train_id}"}
