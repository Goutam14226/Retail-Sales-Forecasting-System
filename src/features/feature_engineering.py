import pandas as pd


def add_time_features(df):
    """
    Add calendar-based time features to a DataFrame.

    Required column:
        date
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    return df


def add_lag_features(df, group_columns=("item_id", "store_id")):
    """
    Add lag-based sales features.

    Lag features are calculated separately for each item-store series.

    Required columns:
        item_id
        store_id
        date
        sales
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        list(group_columns) + ["date"]
    ).reset_index(drop=True)

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby(list(group_columns), sort=False)["sales"]
            .shift(lag)
        )

    return df


def add_rolling_features(df, group_columns=("item_id", "store_id")):
    """
    Add rolling sales statistics.

    Rolling features are calculated separately for each item-store series.

    The current day's sales is excluded to prevent target leakage.

    Required columns:
        item_id
        store_id
        date
        sales
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        list(group_columns) + ["date"]
    ).reset_index(drop=True)

    shifted_sales = (
        df.groupby(
            list(group_columns),
            sort=False
        )["sales"]
        .shift(1)
    )

    group_keys = [df[col] for col in group_columns]

    df["rolling_mean_7"] = (
        shifted_sales
        .groupby(group_keys, sort=False)
        .transform(lambda x: x.rolling(7).mean())
    )

    df["rolling_mean_28"] = (
        shifted_sales
        .groupby(group_keys, sort=False)
        .transform(lambda x: x.rolling(28).mean())
    )

    df["rolling_std_7"] = (
        shifted_sales
        .groupby(group_keys, sort=False)
        .transform(lambda x: x.rolling(7).std())
    )

    df["rolling_std_28"] = (
        shifted_sales
        .groupby(group_keys, sort=False)
        .transform(lambda x: x.rolling(28).std())
    )

    return df


def add_price_features(df, group_columns=("item_id", "store_id")):
    """
    Add price-based features.

    Price changes are calculated separately for each item-store series.

    Missing prices are not filled. Price-derived features remain NaN
    when the required price history is unavailable.

    Required columns:
        item_id
        store_id
        date
        sell_price
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        list(group_columns) + ["date"]
    ).reset_index(drop=True)

    grouped_price = (
        df.groupby(
            list(group_columns),
            sort=False
        )["sell_price"]
    )

    previous_price = grouped_price.shift(1)

    price_7_days_ago = grouped_price.shift(7)

    df["price_change_1"] = (
        df["sell_price"] - previous_price
    )

    df["price_change_pct_1"] = (
        (df["sell_price"] - previous_price)
        / previous_price
    )

    df["price_relative_7"] = (
        df["sell_price"] / price_7_days_ago
    )

    return df


def add_event_snap_features(df):
    """
    Add event and SNAP-related features.

    Features:
        is_event_day
        event_count
        snap_active

    Required columns:
        event_name_1
        event_name_2
        snap_CA
        snap_TX
        snap_WI
        state_id
    """

    df = df.copy()

    event_1 = df["event_name_1"].notna()
    event_2 = df["event_name_2"].notna()

    df["is_event_day"] = (
        event_1 | event_2
    ).astype(int)

    df["event_count"] = (
        event_1.astype(int) +
        event_2.astype(int)
    )

    df["snap_active"] = 0

    ca_mask = df["state_id"] == "CA"
    tx_mask = df["state_id"] == "TX"
    wi_mask = df["state_id"] == "WI"

    df.loc[ca_mask, "snap_active"] = (
        df.loc[ca_mask, "snap_CA"]
        .fillna(0)
        .astype(int)
    )

    df.loc[tx_mask, "snap_active"] = (
        df.loc[tx_mask, "snap_TX"]
        .fillna(0)
        .astype(int)
    )

    df.loc[wi_mask, "snap_active"] = (
        df.loc[wi_mask, "snap_WI"]
        .fillna(0)
        .astype(int)
    )

    return df
