
import pickle
import pyarrow.parquet as pq
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(
    parquet_path,
    feature_columns,
    train_end_date,
    output_path
):
    """
    Fit StandardScaler incrementally on training data only.

    Missing values:
    - sell_price -> 0
    - lag/rolling features -> 0

    The scaler is fitted only on the training period.
    """

    parquet_file = pq.ParquetFile(parquet_path)

    scaler = StandardScaler()

    total_rows = 0

    price_features = [
        "sell_price"
    ]

    history_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28"
    ]

    for rg in range(parquet_file.num_row_groups):

        df = parquet_file.read_row_group(
            rg,
            columns=["date"] + feature_columns
        ).to_pandas()

        df["date"] = pd.to_datetime(df["date"])

        df = df[
            df["date"] <= pd.Timestamp(train_end_date)
        ]

        if len(df) == 0:
            del df
            continue

        for col in price_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for col in history_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        scaler.partial_fit(df[feature_columns])

        total_rows += len(df)

        del df

        if (rg + 1) % 10 == 0:
            print(
                f"Processed row groups: "
                f"{rg + 1}/{parquet_file.num_row_groups}"
            )

    with open(output_path, "wb") as f:
        pickle.dump(scaler, f)

    print("=" * 60)
    print("SCALER FITTING COMPLETE")
    print("=" * 60)
    print("Training rows:", f"{total_rows:,}")
    print("Features:", len(feature_columns))
    print("Scaler saved to:", output_path)

    return scaler


def load_scaler(scaler_path):
    """
    Load a previously fitted scaler.
    """

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return scaler
