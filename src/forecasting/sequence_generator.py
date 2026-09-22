
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def create_sequences(
    series_data,
    feature_columns,
    sequence_length=28,
    forecast_horizon=1
):
    """
    Create sliding-window sequences for one item-store time series.

    Returns:
        X: (num_sequences, sequence_length, num_features)
        y: (num_sequences, forecast_horizon)
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be greater than 0.")

    missing_columns = [
        column for column in feature_columns
        if column not in series_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    minimum_rows = sequence_length + forecast_horizon

    if len(series_data) < minimum_rows:
        raise ValueError(
            f"Not enough rows. Required at least {minimum_rows}, "
            f"but received {len(series_data)}."
        )

    X = []
    y = []

    total_rows = len(series_data)

    for start in range(
        total_rows - sequence_length - forecast_horizon + 1
    ):

        input_end = start + sequence_length
        target_end = input_end + forecast_horizon

        X_window = series_data.iloc[
            start:input_end
        ][feature_columns].values

        y_window = series_data.iloc[
            input_end:target_end
        ]["sales"].values

        X.append(X_window)
        y.append(y_window)

    return np.asarray(X), np.asarray(y)


def create_inference_sequence(
    series_data,
    feature_columns,
    sequence_length=28
):
    """
    Create the latest sequence for next-step inference.

    Returns:
        X: (1, sequence_length, num_features)
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")

    missing_columns = [
        column for column in feature_columns
        if column not in series_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    if len(series_data) < sequence_length:
        raise ValueError(
            f"Not enough rows. Required at least {sequence_length}, "
            f"but received {len(series_data)}."
        )

    series_data = (
        series_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    latest_data = series_data.tail(sequence_length)

    X = latest_data[feature_columns].values

    return np.expand_dims(X, axis=0)


def create_date_filtered_sequences(
    series_data,
    feature_columns,
    target_start_date,
    target_end_date,
    sequence_length=28,
    forecast_horizon=1
):
    """
    Create sequences whose target dates fall within
    the specified date range.

    Returns:
        X: Input sequences
        y: Target values
        target_dates: Target dates
    """

    series_data = (
        series_data
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    series_data["date"] = pd.to_datetime(
        series_data["date"]
    )

    target_start_date = pd.to_datetime(
        target_start_date
    )

    target_end_date = pd.to_datetime(
        target_end_date
    )

    missing_columns = [
        column for column in feature_columns
        if column not in series_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    X = []
    y = []
    target_dates = []

    for i in range(
        sequence_length,
        len(series_data) - forecast_horizon + 1
    ):

        target_date = series_data.iloc[i]["date"]

        if target_date < target_start_date:
            continue

        if target_date > target_end_date:
            break

        X_window = series_data.iloc[
            i - sequence_length:i
        ][feature_columns].values

        y_window = series_data.iloc[
            i:i + forecast_horizon
        ]["sales"].values

        X.append(X_window)
        y.append(y_window)
        target_dates.append(target_date)

    return (
        np.asarray(X),
        np.asarray(y),
        target_dates
    )


def generate_batches_final(
    parquet_path,
    feature_columns,
    sequence_length=28,
    forecast_horizon=1,
    batch_size=256,
    selected_series=None,
    target_start_date=None,
    target_end_date=None
):
    """
    Final memory-safe production sequence generator.

    Rules:
    - Processes Parquet row groups incrementally.
    - Maintains sequence_length history across row groups.
    - Generates each target date exactly once.
    - Missing sell_price is represented as 0.
    - Sequences containing invalid lag/rolling values are skipped.
    - Supports exact item-store filtering.
    """

    if target_start_date is not None:
        target_start_date = pd.Timestamp(target_start_date)

    if target_end_date is not None:
        target_end_date = pd.Timestamp(target_end_date)

    parquet_file = pq.ParquetFile(parquet_path)

    required_columns = [
        "item_id",
        "store_id",
        "date"
    ] + feature_columns

    selected_pairs = None

    if selected_series is not None:
        selected_pairs = set(selected_series)

    required_valid_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28"
    ]

    history = {}

    X_batch = []
    y_batch = []


    for row_group in range(parquet_file.num_row_groups):

        df = parquet_file.read_row_group(
            row_group,
            columns=required_columns
        ).to_pandas()

        # ------------------------------------------
        # Exact item-store filtering
        # ------------------------------------------

        if selected_pairs is not None:

            df = df[
                df.apply(
                    lambda row: (
                        row["item_id"],
                        row["store_id"]
                    ) in selected_pairs,
                    axis=1
                )
            ]

        if len(df) == 0:
            continue

        df["date"] = pd.to_datetime(df["date"])

        # ------------------------------------------
        # Process each series
        # ------------------------------------------

        for key, current_series in df.groupby(
            ["item_id", "store_id"],
            sort=False
        ):

            current_series = (
                current_series
                .sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )

            previous_length = 0

            # --------------------------------------
            # Add previous history
            # --------------------------------------

            if key in history:

                previous = history[key]

                previous_length = len(previous)

                combined = pd.concat(
                    [previous, current_series],
                    ignore_index=True
                )

            else:

                combined = current_series.copy()

            combined = (
                combined
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True)
            )

            # --------------------------------------
            # Missing price handling
            # --------------------------------------

            if "sell_price" in combined.columns:

                combined["sell_price"] = (
                    combined["sell_price"]
                    .fillna(0)
                )

            # --------------------------------------
            # Generate only NEW target dates
            # --------------------------------------

            if len(combined) >= (
                sequence_length + forecast_horizon
            ):

                max_target_index = (
                    len(combined)
                    - forecast_horizon
                )

                first_target_index = max(
                    sequence_length,
                    previous_length
                )

                for target_index in range(
                    first_target_index,
                    max_target_index + 1
                ):

                    target_date = combined.iloc[
                        target_index
                    ]["date"]

                    # ----------------------------------
                    # Target date filtering
                    # ----------------------------------

                    if (
                        target_start_date is not None
                        and target_date < target_start_date
                    ):
                        continue

                    if (
                        target_end_date is not None
                        and target_date > target_end_date
                    ):
                        continue

                    target_end = (
                        target_index
                        + forecast_horizon
                    )

                    X_window = combined.iloc[
                        target_index - sequence_length:
                        target_index
                    ][feature_columns]

                    y_window = combined.iloc[
                        target_index:
                        target_end
                    ]["sales"].values

                    # ----------------------------------
                    # Skip incomplete lag/rolling inputs
                    # ----------------------------------

                    if (
                        X_window[
                            required_valid_features
                        ]
                        .isna()
                        .any()
                        .any()
                    ):
                        continue

                    X_values = (
                        X_window
                        .values
                        .astype(np.float32)
                    )

                    y_values = (
                        y_window
                        .astype(np.float32)
                    )

                    # ----------------------------------
                    # Final numerical safety check
                    # ----------------------------------

                    if np.isnan(X_values).any():
                        continue

                    if np.isinf(X_values).any():
                        continue

                    if np.isnan(y_values).any():
                        continue

                    if np.isinf(y_values).any():
                        continue

                    # ----------------------------------
                    # Add sequence
                    # ----------------------------------

                    X_batch.append(X_values)
                    y_batch.append(y_values)

                    # ----------------------------------
                    # Yield full batch
                    # ----------------------------------

                    if len(X_batch) == batch_size:

                        yield (
                            np.asarray(
                                X_batch,
                                dtype=np.float32
                            ),
                            np.asarray(
                                y_batch,
                                dtype=np.float32
                            )
                        )

                        X_batch = []
                        y_batch = []

            # --------------------------------------
            # Keep only required history
            # --------------------------------------

            history[key] = combined.tail(
                sequence_length
            ).copy()

    # ------------------------------------------
    # Final incomplete batch
    # ------------------------------------------

    if X_batch:

        yield (
            np.asarray(
                X_batch,
                dtype=np.float32
            ),
            np.asarray(
                y_batch,
                dtype=np.float32
            )
        )

