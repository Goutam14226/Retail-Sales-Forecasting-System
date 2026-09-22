# Project Configuration
# Multivariate Time-Series Forecasting


# =========================
# Forecasting Configuration
# =========================

SEQUENCE_LENGTH = 28
FORECAST_HORIZON = 1

TARGET_COLUMN = "sales"

TRAIN_END_DATE = "2016-03-27"
VALIDATION_START_DATE = "2016-03-28"
VALIDATION_END_DATE = "2016-04-24"


# =========================
# Model Configuration
# =========================

BATCH_SIZE = 64
EPOCHS = 10

TRAIN_STEPS = 5000
VALIDATION_STEPS = 500


# =========================
# Feature Configuration
# =========================

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


# =========================
# Inventory Configuration
# =========================

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
Z_VALUE = 1.645

ORDERING_COST = 100
HOLDING_COST = 20

DAYS_PER_YEAR = 365
