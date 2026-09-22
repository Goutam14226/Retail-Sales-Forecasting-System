
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# --------------------------------------------------
# Production forecasting engine
# --------------------------------------------------

from forecasting.forecast import ForecastEngine


# --------------------------------------------------
# Production paths
# --------------------------------------------------

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "features_production_clean.parquet"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm_attention_production.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "feature_scaler.pkl"
)

HISTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "latest_history.parquet"
)


# --------------------------------------------------
# Feature configuration
# --------------------------------------------------

FEATURE_COLUMNS = [
    "sales",
    "sell_price",
    "price_available",
    "day_of_month",
    "week_of_year",
    "day_of_year",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "is_event_day",
    "event_count",
    "snap_active"
]


# --------------------------------------------------
# Forecast engine
# --------------------------------------------------

engine = ForecastEngine(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    feature_columns=FEATURE_COLUMNS,
    sequence_length=28,
    history_path=HISTORY_PATH
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="M5 Sales Forecasting API",
    description=(
        "API for next-day sales forecasting "
        "using LSTM with Attention."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class ForecastRequest(BaseModel):

    item_id: str = Field(
        ...,
        min_length=1,
        description="Product/item identifier"
    )

    store_id: str = Field(
        ...,
        min_length=1,
        description="Store identifier"
    )


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "M5 Sales Forecasting API is running",
        "status": "ok"
    }


# --------------------------------------------------
# Forecast endpoint
# --------------------------------------------------

@app.post("/forecast")
def forecast(request: ForecastRequest):

    try:

        result = engine.forecast_next_day(
            item_id=request.item_id,
            store_id=request.store_id
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
