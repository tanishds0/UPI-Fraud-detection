from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "transaction_time" in out.columns:
        ts = pd.to_datetime(out["transaction_time"], errors="coerce")
        out["transaction_hour"] = ts.dt.hour.fillna(0).astype(int)
        out["transaction_dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    else:
        out["transaction_hour"] = 0
        out["transaction_dayofweek"] = 0
    return out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Avoid division by zero, keep behavior stable for new users
    prev_tx = pd.to_numeric(out.get("num_prev_transactions", 0), errors="coerce").fillna(0)
    age_days = pd.to_numeric(out.get("account_age_days", 0), errors="coerce").fillna(0)
    amount = pd.to_numeric(out.get("amount", 0), errors="coerce").fillna(0)

    out["transaction_amount_ratio"] = amount / (prev_tx + 1.0)
    out["transaction_frequency"] = prev_tx / (age_days.clip(lower=1.0))

    # Unusual location: anything other than typical home/work cities
    location = out.get("location", pd.Series(["unknown"] * len(out)))
    location = location.astype(str)
    out["unusual_location_flag"] = (~location.isin(["home_city", "work_city"])).astype(int)

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering used by both training and inference.
    Keeps raw columns too (unused columns can be dropped later by the preprocessor).
    """
    out = df.copy()
    out = add_time_features(out)
    out = add_domain_features(out)
    return out

