import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "hour",
    "new_device",
    "tx_last_hour",
    "amount_to_balance",
    "recipient_is_new",
]


def generate_transactions(n=20000, fraud_rate=0.03, seed=42):
    """Create synthetic mobile-money-style transactions.

    The data is made up. Fraud rows are drawn from different distributions than
    normal rows (larger amounts, night-time, new devices, bursts of activity),
    with enough overlap that the problem is not trivial.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    def pick(fraud_values, normal_values):
        return np.where(is_fraud, fraud_values, normal_values)

    amount = pick(rng.lognormal(8.0, 1.0, n), rng.lognormal(6.5, 0.9, n))
    hour = np.round(pick(rng.normal(2, 3, n), rng.normal(14, 4, n))).astype(int) % 24
    new_device = pick(rng.random(n) < 0.55, rng.random(n) < 0.05).astype(int)
    tx_last_hour = pick(rng.poisson(3.0, n), rng.poisson(0.6, n))
    amount_to_balance = pick(rng.beta(6, 3, n), rng.beta(2, 8, n))
    recipient_is_new = pick(rng.random(n) < 0.8, rng.random(n) < 0.25).astype(int)

    return pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour": hour,
            "new_device": new_device,
            "tx_last_hour": tx_last_hour,
            "amount_to_balance": np.round(amount_to_balance, 4),
            "recipient_is_new": recipient_is_new,
            "is_fraud": is_fraud.astype(int),
        }
    )
