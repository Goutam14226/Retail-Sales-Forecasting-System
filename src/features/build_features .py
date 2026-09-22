import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

from src.data.data_loader import read_parquet_row_group
from src.features.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_price_features,
    add_event_snap_features,
)


def build_features_for_row_group(
    input_path,
    row_group,
):
    """
    Build all engineered features for one Parquet row group.

    This function is intended for memory-efficient processing
    of the large feature dataset.
    """

    df = read_parquet_row_group(
        input_path,
        row_group=row_group,
    )

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_event_snap_features(df)

    return df


def get_input_metadata(input_path):
    """
    Return basic Parquet metadata.
    """

    parquet_file = pq.ParquetFile(input_path)

    return {
        "rows": parquet_file.metadata.num_rows,
        "columns": parquet_file.metadata.num_columns,
        "row_groups": parquet_file.num_row_groups,
    }


from collections import defaultdict, deque


def initialize_history():
    """
    Create an empty history state.

    Each item-store series keeps its recent observations
    required for lag, rolling, and price features.
    """

    return defaultdict(
        lambda: {
            "sales": deque(maxlen=28),
            "price": deque(maxlen=28),
        }
    )


def update_history(history, df, group_columns=("item_id", "store_id")):
    """
    Update history using the processed DataFrame.

    History is maintained separately for each item-store series.

    The latest 28 sales and prices are retained.
    """

    for _, row in df.iterrows():

        key = tuple(row[col] for col in group_columns)

        history[key]["sales"].append(row["sales"])
        history[key]["price"].append(row["sell_price"])

    return history


def add_stateful_sales_features(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Add lag and rolling sales features using historical state.

    The history contains observations from previous row groups.
    Current sales are not used to calculate the current target features.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    feature_rows = []

    for _, row in df.iterrows():

        key = tuple(row[col] for col in group_columns)

        sales_history = history[key]["sales"]

        features = {}

        # -------------------------
        # Lag features
        # -------------------------

        for lag in [1, 7, 14, 28]:

            if len(sales_history) >= lag:
                features[f"lag_{lag}"] = sales_history[-lag]
            else:
                features[f"lag_{lag}"] = float("nan")

        # -------------------------
        # Rolling features
        # -------------------------

        if len(sales_history) >= 7:
            last_7 = list(sales_history)[-7:]
            features["rolling_mean_7"] = sum(last_7) / 7
            features["rolling_std_7"] = pd.Series(last_7).std()
        else:
            features["rolling_mean_7"] = float("nan")
            features["rolling_std_7"] = float("nan")

        if len(sales_history) >= 28:
            last_28 = list(sales_history)[-28:]
            features["rolling_mean_28"] = sum(last_28) / 28
            features["rolling_std_28"] = pd.Series(last_28).std()
        else:
            features["rolling_mean_28"] = float("nan")
            features["rolling_std_28"] = float("nan")

        feature_rows.append(features)

        # IMPORTANT:
        # Update history only AFTER calculating
        # current row's features.
        sales_history.append(row["sales"])

    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    for column in feature_df.columns:
        df[column] = feature_df[column]

    return df, history


def add_stateful_price_features(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Add price features using historical state.

    The logic matches the finalized add_price_features()
    implementation:

    price_change_1:
        current price - previous observation price

    price_change_pct_1:
        (current price - previous price) / previous price

    price_relative_7:
        current price / price from 7 observations ago

    Missing prices are preserved as NaN.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    feature_rows = []

    for _, row in df.iterrows():

        key = tuple(row[col] for col in group_columns)

        price_history = history[key]["price"]

        current_price = row["sell_price"]

        # ---------------------------------
        # Previous observation price
        # ---------------------------------

        if len(price_history) >= 1:

            previous_price = price_history[-1]

            if (
                pd.notna(current_price)
                and pd.notna(previous_price)
            ):
                price_change_1 = (
                    current_price - previous_price
                )

                price_change_pct_1 = (
                    (current_price - previous_price)
                    / previous_price
                )
            else:
                price_change_1 = float("nan")
                price_change_pct_1 = float("nan")

        else:
            price_change_1 = float("nan")
            price_change_pct_1 = float("nan")


        # ---------------------------------
        # Price 7 observations ago
        # ---------------------------------

        if len(price_history) >= 7:

            price_7_days_ago = price_history[-7]

            if (
                pd.notna(current_price)
                and pd.notna(price_7_days_ago)
            ):
                price_relative_7 = (
                    current_price / price_7_days_ago
                )
            else:
                price_relative_7 = float("nan")

        else:
            price_relative_7 = float("nan")


        feature_rows.append({
            "price_change_1": price_change_1,
            "price_change_pct_1": price_change_pct_1,
            "price_relative_7": price_relative_7,
        })


        # ---------------------------------
        # Update state AFTER calculation
        # ---------------------------------

        price_history.append(current_price)


    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    for column in feature_df.columns:
        df[column] = feature_df[column]

    return df, history


def build_stateful_features_for_chunk(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Build stateful sales and price features for one chunk.

    Historical state is maintained separately for every
    item-store series.

    Current observation is added to history only AFTER
    calculating its features, preventing target leakage.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    feature_rows = []

    for _, row in df.iterrows():

        key = tuple(
            row[col]
            for col in group_columns
        )

        sales_history = history[key]["sales"]
        price_history = history[key]["price"]

        features = {}

        # =================================
        # SALES LAGS
        # =================================

        for lag in [1, 7, 14, 28]:

            if len(sales_history) >= lag:
                features[f"lag_{lag}"] = (
                    sales_history[-lag]
                )
            else:
                features[f"lag_{lag}"] = float("nan")


        # =================================
        # SALES ROLLING FEATURES
        # =================================

        if len(sales_history) >= 7:

            last_7 = list(sales_history)[-7:]

            features["rolling_mean_7"] = (
                sum(last_7) / 7
            )

            features["rolling_std_7"] = (
                pd.Series(last_7).std()
            )

        else:

            features["rolling_mean_7"] = float("nan")
            features["rolling_std_7"] = float("nan")


        if len(sales_history) >= 28:

            last_28 = list(sales_history)[-28:]

            features["rolling_mean_28"] = (
                sum(last_28) / 28
            )

            features["rolling_std_28"] = (
                pd.Series(last_28).std()
            )

        else:

            features["rolling_mean_28"] = float("nan")
            features["rolling_std_28"] = float("nan")


        # =================================
        # PRICE FEATURES
        # =================================

        current_price = row["sell_price"]


        # Previous observation price
        if len(price_history) >= 1:

            previous_price = price_history[-1]

            if (
                pd.notna(current_price)
                and pd.notna(previous_price)
            ):

                features["price_change_1"] = (
                    current_price
                    - previous_price
                )

                features["price_change_pct_1"] = (
                    (
                        current_price
                        - previous_price
                    )
                    / previous_price
                )

            else:

                features["price_change_1"] = float("nan")
                features["price_change_pct_1"] = float("nan")

        else:

            features["price_change_1"] = float("nan")
            features["price_change_pct_1"] = float("nan")


        # Price 7 observations ago
        if len(price_history) >= 7:

            price_7_days_ago = list(price_history)[-7]


            if (
                pd.notna(current_price)
                and pd.notna(price_7_days_ago)
            ):

                features["price_relative_7"] = (
                    current_price
                    / price_7_days_ago
                )

            else:

                features["price_relative_7"] = float("nan")

        else:

            features["price_relative_7"] = float("nan")


        feature_rows.append(features)


        # =================================
        # UPDATE HISTORY
        # =================================

        sales_history.append(
            row["sales"]
        )

        price_history.append(
            current_price
        )


    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    for column in feature_df.columns:
        df[column] = feature_df[column]

    return df, history


import numpy as np


def build_stateful_features_fast(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Efficient stateful feature generation for one Parquet row group.

    Historical state is maintained separately for each item-store series.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    n = len(df)

    # -----------------------------------------
    # Output arrays
    # -----------------------------------------

    lag_1 = np.full(n, np.nan)
    lag_7 = np.full(n, np.nan)
    lag_14 = np.full(n, np.nan)
    lag_28 = np.full(n, np.nan)

    rolling_mean_7 = np.full(n, np.nan)
    rolling_mean_28 = np.full(n, np.nan)
    rolling_std_7 = np.full(n, np.nan)
    rolling_std_28 = np.full(n, np.nan)

    price_change_1 = np.full(n, np.nan)
    price_change_pct_1 = np.full(n, np.nan)
    price_relative_7 = np.full(n, np.nan)

    sales = df["sales"].to_numpy(dtype=float)
    prices = df["sell_price"].to_numpy(dtype=float)

    group_values = df[list(group_columns)].to_numpy()

    start = 0

    while start < n:

        current_key = tuple(group_values[start])

        end = start + 1

        while end < n and tuple(group_values[end]) == current_key:
            end += 1

        series_sales = sales[start:end]
        series_prices = prices[start:end]

        state = history[current_key]

        sales_history = list(state["sales"])
        price_history = list(state["price"])

        history_len = len(sales_history)

        full_sales = np.concatenate([
            np.asarray(sales_history, dtype=float),
            series_sales
        ])

        full_prices = np.concatenate([
            np.asarray(price_history, dtype=float),
            series_prices
        ])

        # -----------------------------------------
        # Lag features
        # -----------------------------------------

        for i in range(len(series_sales)):

            full_index = history_len + i

            for lag, output in [
                (1, lag_1),
                (7, lag_7),
                (14, lag_14),
                (28, lag_28),
            ]:

                if full_index >= lag:
                    output[start + i] = full_sales[full_index - lag]

        # -----------------------------------------
        # Rolling features
        # -----------------------------------------

        for i in range(len(series_sales)):

            full_index = history_len + i

            if full_index >= 7:

                previous_values = full_sales[
                    full_index - 7:full_index
                ]

                rolling_mean_7[start + i] = np.mean(
                    previous_values
                )

                rolling_std_7[start + i] = np.std(
                    previous_values,
                    ddof=1
                )

            if full_index >= 28:

                previous_values = full_sales[
                    full_index - 28:full_index
                ]

                rolling_mean_28[start + i] = np.mean(
                    previous_values
                )

                rolling_std_28[start + i] = np.std(
                    previous_values,
                    ddof=1
                )

        # -----------------------------------------
        # Price features
        # -----------------------------------------

        for i in range(len(series_prices)):

            full_index = history_len + i

            current_price = full_prices[full_index]

            if full_index >= 1:

                previous_price = full_prices[full_index - 1]

                if (
                    not np.isnan(current_price)
                    and not np.isnan(previous_price)
                ):

                    price_change_1[start + i] = (
                        current_price - previous_price
                    )

                    price_change_pct_1[start + i] = (
                        (current_price - previous_price)
                        / previous_price
                    )

            if full_index >= 7:

                price_7 = full_prices[full_index - 7]

                if (
                    not np.isnan(current_price)
                    and not np.isnan(price_7)
                ):

                    price_relative_7[start + i] = (
                        current_price / price_7
                    )

        # -----------------------------------------
        # Update history
        # -----------------------------------------

        state["sales"].clear()
        state["sales"].extend(full_sales[-28:])

        state["price"].clear()
        state["price"].extend(full_prices[-28:])

        start = end

    # -----------------------------------------
    # Attach features
    # -----------------------------------------

    df["lag_1"] = lag_1
    df["lag_7"] = lag_7
    df["lag_14"] = lag_14
    df["lag_28"] = lag_28

    df["rolling_mean_7"] = rolling_mean_7
    df["rolling_mean_28"] = rolling_mean_28
    df["rolling_std_7"] = rolling_std_7
    df["rolling_std_28"] = rolling_std_28

    df["price_change_1"] = price_change_1
    df["price_change_pct_1"] = price_change_pct_1
    df["price_relative_7"] = price_relative_7

    return df, history


def build_stateful_features_vectorized(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Vectorized stateful feature generation for one Parquet row group.

    Historical state is maintained separately for each item-store series.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------------------
    # Keep original row order
    # -----------------------------------------

    original_columns = df.columns.tolist()

    # -----------------------------------------
    # Build history DataFrame
    # -----------------------------------------

    history_rows = []

    for key, state in history.items():

        sales_history = list(state["sales"])
        price_history = list(state["price"])

        max_length = max(
            len(sales_history),
            len(price_history)
        )

        for i in range(max_length):

            history_rows.append({
                group_columns[0]: key[0],
                group_columns[1]: key[1],
                "sales": (
                    sales_history[i]
                    if i < len(sales_history)
                    else np.nan
                ),
                "sell_price": (
                    price_history[i]
                    if i < len(price_history)
                    else np.nan
                ),
                "_history": True,
                "_order": -max_length + i,
            })

    # -----------------------------------------
    # Current rows
    # -----------------------------------------

    current = df.copy()

    current["_history"] = False
    current["_order"] = np.arange(len(current))

    current_columns = list(group_columns) + [
        "date",
        "sales",
        "sell_price",
        "_history",
        "_order",
    ]

    current = current[current_columns]

    # -----------------------------------------
    # Combine history + current
    # -----------------------------------------

    if history_rows:

        history_df = pd.DataFrame(history_rows)

        combined = pd.concat(
            [history_df, current],
            ignore_index=True
        )

    else:

        combined = current.copy()

    # -----------------------------------------
    # Sort by series and chronological order
    # -----------------------------------------

    combined = combined.sort_values(
        list(group_columns) + ["_order"]
    ).reset_index(drop=True)

    # -----------------------------------------
    # Grouped sales
    # -----------------------------------------

    grouped_sales = combined.groupby(
        list(group_columns),
        sort=False
    )["sales"]

    # -----------------------------------------
    # Lag features
    # -----------------------------------------

    for lag in [1, 7, 14, 28]:

        combined[f"lag_{lag}"] = (
            grouped_sales.shift(lag)
        )

    # -----------------------------------------
    # Shifted sales for rolling features
    # -----------------------------------------

    shifted_sales = grouped_sales.shift(1)

    shifted_group = [
        combined[group_columns[0]],
        combined[group_columns[1]],
    ]

    shifted_grouped = shifted_sales.groupby(
        shifted_group,
        sort=False
    )

    # -----------------------------------------
    # Rolling features
    # -----------------------------------------

    combined["rolling_mean_7"] = (
        shifted_grouped
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    combined["rolling_mean_28"] = (
        shifted_grouped
        .transform(
            lambda x: x.rolling(28).mean()
        )
    )

    combined["rolling_std_7"] = (
        shifted_grouped
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    combined["rolling_std_28"] = (
        shifted_grouped
        .transform(
            lambda x: x.rolling(28).std()
        )
    )

    # -----------------------------------------
    # Price features
    # -----------------------------------------

    grouped_price = combined.groupby(
        list(group_columns),
        sort=False
    )["sell_price"]

    previous_price = grouped_price.shift(1)
    price_7 = grouped_price.shift(7)

    combined["price_change_1"] = (
        combined["sell_price"] - previous_price
    )

    combined["price_change_pct_1"] = (
        (
            combined["sell_price"] - previous_price
        )
        / previous_price
    )

    combined["price_relative_7"] = (
        combined["sell_price"] / price_7
    )

    # -----------------------------------------
    # Preserve finalized missing-price logic
    # -----------------------------------------

    invalid_previous = (
        combined["sell_price"].isna()
        | previous_price.isna()
    )

    combined.loc[
        invalid_previous,
        [
            "price_change_1",
            "price_change_pct_1",
        ]
    ] = np.nan

    invalid_price_7 = (
        combined["sell_price"].isna()
        | price_7.isna()
    )

    combined.loc[
        invalid_price_7,
        "price_relative_7"
    ] = np.nan

    # -----------------------------------------
    # Keep only current rows
    # -----------------------------------------

    result = combined[
        combined["_history"] == False
    ].copy()

    # Current rows should retain their original order
    result = result.sort_values("_order")
    result = result.reset_index(drop=True)

    # -----------------------------------------
    # Update history
    # -----------------------------------------

    for key, group in result.groupby(
        list(group_columns),
        sort=False
    ):

        key = tuple(key)

        state = history[key]

        state["sales"].clear()
        state["sales"].extend(
            group["sales"].to_numpy()[-28:]
        )

        state["price"].clear()
        state["price"].extend(
            group["sell_price"].to_numpy()[-28:]
        )

    # -----------------------------------------
    # Restore columns
    # -----------------------------------------

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "price_change_1",
        "price_change_pct_1",
        "price_relative_7",
    ]

    for column in feature_columns:
        df[column] = result[column].to_numpy()

    return df, history


def process_and_write_features(
    input_path,
    output_path,
):
    """
    Process the input Parquet file row-group by row-group
    and write engineered features incrementally.
    """

    input_file = pq.ParquetFile(input_path)

    history = initialize_history()

    writer = None
    total_rows = 0

    try:

        for row_group in range(input_file.num_row_groups):

            print(
                f"Processing row group "
                f"{row_group + 1}/{input_file.num_row_groups}"
            )

            # Read one row group
            df = read_parquet_row_group(
                input_path,
                row_group=row_group,
            )

            # Add time features
            df = add_time_features(df)

            # Add event and SNAP features
            df = add_event_snap_features(df)

            # Add stateful lag, rolling and price features
            df, history = build_stateful_features_vectorized(
                df,
                history,
            )

            # Convert pandas DataFrame to Arrow Table
            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            # Create writer using first chunk schema
            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )

            # Write current row group
            writer.write_table(table)

            total_rows += len(df)

            print(
                f"Rows written: {total_rows:,}"
            )

            del df
            del table

    finally:

        if writer is not None:
            writer.close()

    print("\n" + "=" * 60)
    print("FEATURE BUILD COMPLETE")
    print("=" * 60)
    print(f"Total rows written: {total_rows:,}")
    print(f"Output file: {output_path}")


def add_stateful_price_features_corrected(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Calculate price features using previous AVAILABLE prices.

    Missing sell_price values are skipped when determining
    the previous available price.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    group_cols = list(group_columns)

    df = df.sort_values(
        group_cols + ["date"]
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Previous available price
    # -------------------------------------------------

    df["_previous_available_price"] = (
        df.groupby(group_cols, sort=False)["sell_price"]
        .transform(
            lambda x: x.ffill().shift(1)
        )
    )

    # -------------------------------------------------
    # Price change
    # -------------------------------------------------

    df["price_change_1"] = (
        df["sell_price"]
        - df["_previous_available_price"]
    )

    df["price_change_pct_1"] = (
        df["price_change_1"]
        / df["_previous_available_price"]
    )

    # -------------------------------------------------
    # 7th previous available price
    # -------------------------------------------------

    available = df["sell_price"].notna()

    available_df = df.loc[
        available,
        group_cols + ["sell_price"]
    ].copy()

    available_df["_available_rank"] = (
        available_df
        .groupby(group_cols, sort=False)
        .cumcount()
    )

    target_df = available_df.copy()

    target_df["_target_rank"] = (
        target_df["_available_rank"] + 7
    )

    lookup_df = available_df[
        group_cols +
        ["_available_rank", "sell_price"]
    ].copy()

    lookup_df = lookup_df.rename(
        columns={
            "_available_rank": "_target_rank",
            "sell_price": "_price_7_available_ago",
        }
    )

    df = df.merge(
        lookup_df,
        on=group_cols + ["_target_rank"],
        how="left",
        sort=False,
    )

    df["price_relative_7"] = (
        df["sell_price"]
        / df["_price_7_available_ago"]
    )

    # -------------------------------------------------
    # Remove temporary columns
    # -------------------------------------------------

    df = df.drop(
        columns=[
            "_previous_available_price",
            "_price_7_available_ago",
        ],
        errors="ignore",
    )

    return df


def add_stateful_price_features_corrected_v2(
    df,
    history,
    group_columns=("item_id", "store_id"),
):
    """
    Price features using previous AVAILABLE prices.

    Missing prices are ignored when looking for previous
    available price observations.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    group_cols = list(group_columns)

    df = df.sort_values(
        group_cols + ["date"]
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Previous available price
    # -------------------------------------------------

    df["_previous_available_price"] = (
        df.groupby(group_cols, sort=False)["sell_price"]
        .transform(
            lambda x: x.ffill().shift(1)
        )
    )

    df["price_change_1"] = (
        df["sell_price"]
        - df["_previous_available_price"]
    )

    df["price_change_pct_1"] = (
        df["price_change_1"]
        / df["_previous_available_price"]
    )

    # -------------------------------------------------
    # Rank only rows where price is available
    # -------------------------------------------------

    available = df["sell_price"].notna()

    df["_available_rank"] = pd.NA

    df.loc[available, "_available_rank"] = (
        df.loc[available]
        .groupby(group_cols, sort=False)
        .cumcount()
    )

    # -------------------------------------------------
    # Target = 7 available-price observations ago
    # -------------------------------------------------

    df["_target_rank"] = (
        df["_available_rank"] - 7
    )

    # -------------------------------------------------
    # Lookup table
    # -------------------------------------------------

    lookup = df.loc[
        available,
        group_cols + [
            "_available_rank",
            "sell_price",
        ],
    ].copy()

    lookup = lookup.rename(
        columns={
            "_available_rank": "_target_rank",
            "sell_price": "_price_7_available_ago",
        }
    )

    # -------------------------------------------------
    # Merge 7th previous available price
    # -------------------------------------------------

    df = df.merge(
        lookup,
        on=group_cols + ["_target_rank"],
        how="left",
        sort=False,
    )

    df["price_relative_7"] = (
        df["sell_price"]
        / df["_price_7_available_ago"]
    )

    # -------------------------------------------------
    # Remove temporary columns
    # -------------------------------------------------

    df = df.drop(
        columns=[
            "_previous_available_price",
            "_available_rank",
            "_target_rank",
            "_price_7_available_ago",
        ],
        errors="ignore",
    )

    return df


def add_stateful_price_features_v3(
    df,
    price_history,
    group_columns=("item_id", "store_id"),
):
    """
    Stateful price features using previous AVAILABLE prices.

    price_history stores recent available prices for each
    item-store series so row-group boundaries are handled.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    group_cols = list(group_columns)

    df = df.sort_values(
        group_cols + ["date"]
    ).reset_index(drop=True)

    # Output columns
    df["price_change_1"] = float("nan")
    df["price_change_pct_1"] = float("nan")
    df["price_relative_7"] = float("nan")

    # Process each item-store series
    for key, group in df.groupby(
        group_cols,
        sort=False
    ):

        key = tuple(key)

        if key not in price_history:
            price_history[key] = deque(maxlen=7)

        previous_prices = price_history[key]

        for idx in group.index:

            current_price = df.at[
                idx,
                "sell_price"
            ]

            # Only available prices are used
            if pd.notna(current_price):

                # Previous available price
                if len(previous_prices) >= 1:

                    previous_price = previous_prices[-1]

                    df.at[
                        idx,
                        "price_change_1"
                    ] = (
                        current_price
                        - previous_price
                    )

                    df.at[
                        idx,
                        "price_change_pct_1"
                    ] = (
                        current_price
                        - previous_price
                    ) / previous_price

                # 7th previous available price
                if len(previous_prices) >= 7:

                    price_7_ago = previous_prices[-7]

                    df.at[
                        idx,
                        "price_relative_7"
                    ] = (
                        current_price
                        / price_7_ago
                    )

                # Update state
                previous_prices.append(
                    current_price
                )

    return df, price_history
