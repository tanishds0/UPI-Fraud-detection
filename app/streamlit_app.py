from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import engineer_features


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "fraud_model.pkl"
    return joblib.load(model_path)


def build_input_row(
    amount: float,
    transaction_type: str,
    device_type: str,
    num_prev_transactions: int,
    account_age_days: int,
) -> pd.DataFrame:
    # The model was trained on these raw columns (plus engineered features).
    # For app simplicity we assume a domestic, typical location by default.
    row = {
        "transaction_id": 0,
        "amount": float(amount),
        "transaction_time": datetime.now().isoformat(),
        "location": "home_city",
        "device_type": device_type,
        "transaction_type": transaction_type,
        "account_age_days": int(account_age_days),
        "num_prev_transactions": int(num_prev_transactions),
        "is_foreign_transaction": 0,
    }
    df = pd.DataFrame([row])
    return engineer_features(df)


def main() -> None:
    st.set_page_config(page_title="UPI Fraud Detection System", layout="centered")
    st.title("UPI Fraud Detection System")
    st.write("Enter transaction details to estimate fraud probability.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction amount", min_value=0.0, value=500.0, step=10.0)
        transaction_type = st.selectbox(
            "Transaction type",
            options=["p2p", "merchant", "bill_payment", "upi_autopay"],
            index=0,
        )
    with col2:
        device_type = st.selectbox("Device type", options=["android", "ios", "web"], index=0)
        num_prev_transactions = st.number_input(
            "Number of previous transactions", min_value=0, value=10, step=1
        )

    account_age_days = st.number_input("Account age (days)", min_value=1, value=180, step=1)

    if st.button("Check Transaction"):
        model = load_model()
        X = build_input_row(
            amount=amount,
            transaction_type=transaction_type,
            device_type=device_type,
            num_prev_transactions=num_prev_transactions,
            account_age_days=account_age_days,
        )

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[:, 1][0])
        else:
            proba = float(model.predict(X)[0])

        label = "Fraudulent Transaction" if proba >= 0.5 else "Genuine Transaction"

        st.subheader("Result")
        st.write(f"**Fraud probability:** {proba:.2%}")
        if label.startswith("Fraudulent"):
            st.error(label)
        else:
            st.success(label)


if __name__ == "__main__":
    main()

