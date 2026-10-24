"""Central config: paths and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Data files
SALES_CSV = DATA_RAW / "sales.csv"
CUSTOMERS_CSV = DATA_RAW / "customers.csv"
TICKETS_CSV = DATA_RAW / "tickets.csv"

# Forecasting
FORECAST_HORIZON_DAYS = 30
