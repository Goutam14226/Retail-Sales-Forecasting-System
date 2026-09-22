import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq


def collect_series_history(
    parquet_path,
    series_list,
    feature_columns
):
    """
    Collect history for multiple item-store series
    using a single Parquet scan.
    """

    if not series_list:
        raise ValueError(
            "series_list cannot be empty"
        )

    requested_series = set(series_list)

    required_columns = [
        "item_id",
        "store_id",
        "date"
    ] + feature_columns

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    parts = []

    for row_group in range(
        parquet_file.num_row_groups
    ):

        df = parquet_file.read_row_group(
            row_group,
            columns=required_columns
        ).to_pandas()

        df["_series_key"] = list(
            zip(
                df["item_id"],
                df["store_id"]
            )
        )

        df = df[
            df["_series_key"].isin(
                requested_series
            )
        ].drop(
            columns="_series_key"
        )

        if len(df) > 0:
            parts.append(df)

        del df

    if not parts:
        raise ValueError(
            "None of the requested series were found."
        )

    result = pd.concat(
        parts,
        ignore_index=True
    )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = (
        result
        .sort_values(
            [
                "item_id",
                "store_id",
                "date"
            ]
        )
        .drop_duplicates(
            [
                "item_id",
                "store_id",
                "date"
            ]
        )
        .reset_index(drop=True)
    )

    return result


def batch_forecast(
    engine,
    series_list
):
    """
    Generate next-day forecasts for multiple
    item-store series using a single Parquet scan.
    """

    history_df = collect_series_history(
        parquet_path=engine.data_path,
        series_list=series_list,
        feature_columns=engine.feature_columns
    )

    results = []

    for item_id, store_id in series_list:

        series_df = history_df[
            (history_df["item_id"] == item_id) &
            (history_df["store_id"] == store_id)
        ].copy()

        if len(series_df) == 0:
            raise ValueError(
                f"Series not found: "
                f"{item_id} / {store_id}"
            )

        series_df = (
            series_df
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(series_df) < engine.sequence_length:
            raise ValueError(
                f"Not enough history for "
                f"{item_id} / {store_id}"
            )

        latest_sequence = series_df.tail(
            engine.sequence_length
        ).copy()

        last_available_date = (
            latest_sequence["date"].max()
        )

        forecast_date = (
            last_available_date +
            pd.Timedelta(days=1)
        )

        X_raw = latest_sequence[
            engine.feature_columns
        ].copy()

        if "sell_price" in X_raw.columns:
            X_raw["sell_price"] = (
                X_raw["sell_price"].fillna(0)
            )

        if X_raw.isna().sum().sum() > 0:
            raise ValueError(
                f"NaN values found for "
                f"{item_id} / {store_id}"
            )

        if np.isinf(
            X_raw.to_numpy()
        ).sum() > 0:
            raise ValueError(
                f"Inf values found for "
                f"{item_id} / {store_id}"
            )

        X_scaled = engine.scaler.transform(
            X_raw
        ).astype(np.float32)

        X_scaled = X_scaled.reshape(
            1,
            engine.sequence_length,
            len(engine.feature_columns)
        )

        prediction = engine.model(
            tf.convert_to_tensor(X_scaled),
            training=False
        ).numpy()

        forecast = float(
            prediction[0, 0]
        )

        if not np.isfinite(forecast):
            raise ValueError(
                f"Invalid forecast for "
                f"{item_id} / {store_id}"
            )

        forecast = max(
            0.0,
            forecast
        )

        results.append({
            "item_id": item_id,
            "store_id": store_id,
            "last_available_date":
                last_available_date.strftime(
                    "%Y-%m-%d"
                ),
            "forecast_date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),
            "forecast": forecast
        })

    return pd.DataFrame(results)
