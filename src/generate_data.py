import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_transactions(
    n_samples: int = 10000, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic UPI transactions dataset.

    Columns:
        - transaction_id
        - amount
        - transaction_time
        - location
        - device_type
        - transaction_type
        - account_age_days
        - num_prev_transactions
        - is_foreign_transaction
        - fraud
    """
    rng = np.random.default_rng(random_state)

    transaction_id = np.arange(1, n_samples + 1)

    # Transaction amounts: mix of low and high values (log-normal)
    amount = np.round(np.exp(rng.normal(loc=7.5, scale=0.8, size=n_samples)), 2)

    # Random timestamps over ~60 days
    base = np.datetime64("2025-01-01")
    days = rng.integers(0, 60, size=n_samples)
    seconds = rng.integers(0, 24 * 60 * 60, size=n_samples)
    transaction_time = base + days.astype("timedelta64[D]") + seconds.astype(
        "timedelta64[s]"
    )

    locations = ["home_city", "work_city", "other_domestic", "international"]
    location = rng.choice(locations, size=n_samples, p=[0.45, 0.25, 0.2, 0.1])

    device_types = ["android", "ios", "web"]
    device_type = rng.choice(device_types, size=n_samples, p=[0.6, 0.3, 0.1])

    tx_types = ["p2p", "merchant", "bill_payment", "upi_autopay"]
    transaction_type = rng.choice(tx_types, size=n_samples, p=[0.5, 0.3, 0.15, 0.05])

    account_age_days = rng.integers(1, 365 * 5, size=n_samples)

    # Number of previous transactions (heavier for older accounts)
    base_freq = rng.poisson(lam=2, size=n_samples)
    age_factor = np.clip(account_age_days / 365.0, 0.1, None)
    num_prev_transactions = np.maximum(
        (base_freq * age_factor * 10).astype(int), 0
    )

    # Foreign transaction flag
    is_foreign_transaction = (location == "international").astype(int)

    # Fraud probability: higher for foreign, high amount, new accounts, unusual device/type
    prob_fraud = (
        0.02
        + 0.25 * (is_foreign_transaction == 1)
        + 0.15 * (amount > np.percentile(amount, 90))
        + 0.1 * (account_age_days < 30)
        + 0.05 * (transaction_type == "merchant")
        + 0.05 * (device_type == "web")
    )
    prob_fraud = np.clip(prob_fraud, 0, 0.95)
    fraud = rng.binomial(1, prob_fraud)

    df = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "amount": amount,
            "transaction_time": transaction_time.astype("datetime64[ns]"),
            "location": location,
            "device_type": device_type,
            "transaction_type": transaction_type,
            "account_age_days": account_age_days,
            "num_prev_transactions": num_prev_transactions,
            "is_foreign_transaction": is_foreign_transaction,
            "fraud": fraud,
        }
    )

    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transactions()
    output_path = data_dir / "transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset with {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

