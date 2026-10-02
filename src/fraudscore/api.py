import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model import load_or_train, score


class Transaction(BaseModel):
    amount: float = Field(gt=0, description="Transaction amount in KES")
    hour: int = Field(ge=0, le=23, description="Hour of the day (0-23)")
    new_device: bool = Field(description="True if the device is new on this account")
    tx_last_hour: int = Field(ge=0, description="Transactions by this account in the last hour")
    amount_to_balance: float = Field(
        ge=0, le=1, description="Amount as a fraction of the account balance"
    )
    recipient_is_new: bool = Field(description="True if the recipient was never paid before")


class Prediction(BaseModel):
    fraud_probability: float
    risk_level: str
    flagged: bool


def create_app(bundle=None):
    """Build the API. Pass a trained bundle (used by tests), or let it load one."""
    app = FastAPI(
        title="Fraud Scoring API",
        version="1.0.0",
        description="Scores mobile-money-style transactions for fraud risk. "
        "Trained on synthetic data, so it is a demo of the pipeline, not a real fraud model.",
    )
    app.state.bundle = bundle
    lock = threading.Lock()

    def get_bundle():
        with lock:
            if app.state.bundle is None:
                app.state.bundle = load_or_train()
            return app.state.bundle

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.bundle is not None}

    @app.get("/model/metrics")
    def metrics():
        return get_bundle()["metrics"]

    @app.post("/predict", response_model=Prediction)
    def predict(transaction: Transaction):
        features = {
            "amount": transaction.amount,
            "hour": transaction.hour,
            "new_device": int(transaction.new_device),
            "tx_last_hour": transaction.tx_last_hour,
            "amount_to_balance": transaction.amount_to_balance,
            "recipient_is_new": int(transaction.recipient_is_new),
        }
        return score(get_bundle(), features)

    return app


app = create_app()
