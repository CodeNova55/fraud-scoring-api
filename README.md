# Fraud Scoring API

A small machine learning service that scores mobile-money-style transactions for fraud risk. It generates synthetic transactions, trains and evaluates a model, and serves predictions through a FastAPI API.

> **Important:** the data is synthetic (made up for this project). The metrics show that the pipeline works end to end, not how a model would perform on real transactions.

## Features
- Synthetic transaction generator (amount in KES, hour of day, new device, recent activity, amount relative to balance, new recipient)
- Random forest model trained with a stratified train/test split and class weighting for the rare fraud class
- Evaluation that goes beyond accuracy: ROC AUC, average precision, precision, recall, F1, a confusion matrix, and the "always predict not fraud" baseline
- FastAPI service with validated input, a probability, a risk band (low, medium, high), and a flag
- Interactive API docs at `/docs`
- Trains a model on first use if none has been saved yet
- 28 automated tests that need no network and run in seconds

## Getting Started

```bash
git clone https://github.com/CodeNova55/fraud-scoring-api.git
cd fraud-scoring-api
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

The code lives in `src/`, so tell Python where to find it (once per terminal):

```powershell
$env:PYTHONPATH = "src"       # Windows PowerShell
# export PYTHONPATH=src       # macOS/Linux
```

Train the model and save it with its metrics:

```bash
python -m fraudscore.train
```

Start the API and open http://127.0.0.1:8000/docs :

```bash
uvicorn fraudscore.api:app
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | score one transaction |
| `GET /model/metrics` | evaluation results from training |
| `GET /health` | service status and whether the model is loaded |

Example request:

```json
{
  "amount": 45000,
  "hour": 2,
  "new_device": true,
  "tx_last_hour": 5,
  "amount_to_balance": 0.9,
  "recipient_is_new": true
}
```

Example response (the exact numbers depend on the trained model):

```json
{
  "fraud_probability": 0.97,
  "risk_level": "high",
  "flagged": true
}
```

Invalid input (for example `hour` above 23 or a missing field) returns a `422` error.

## Reading the Results

After training, `models/metrics.json` holds the evaluation on a held-out test set. Fraud is rare, so accuracy on its own is misleading: `baseline_accuracy` is what you would get by always predicting "not fraud". Look at these instead:
- **recall**: the share of real fraud that was caught
- **precision**: the share of flagged transactions that really were fraud
- **average_precision** and **roc_auc**: how well fraud is ranked above normal transactions across all thresholds

## Running Tests

```bash
python -m pytest -v
```

## Project Structure

## Limitations
- Synthetic data only; real fraud is messier, and a real model would need real labelled data
- The decision threshold is fixed at 0.5; a real system would choose it from the cost of missed fraud versus false alarms
- The saved model is a pickle file, so only load model files you trained yourself
- No authentication, rate limiting, or monitoring
- If no saved model exists, the first request trains one, which takes a few seconds

## Git Workflow
- One feature branch per piece of work
- Feature branch -> PR into `develop`
- `develop` -> PR into `main` (after approval)
- No direct pushes to `develop` or `main`

## Roadmap
- Train on real, labelled transaction data
- Choose the threshold from business costs
- Add authentication and rate limiting
- Containerize with Docker and deploy
- Monitor for data drift
