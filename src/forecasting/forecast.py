import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf


class ForecastEngine:

    def __init__(
        self,
        data_path,
        model_path,
        scaler_path,
        feature_columns,
        sequence_length=28,
        history_path=None
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if len(feature_columns) != 19:
            raise ValueError(
                f"Expected 19 features, got {len(feature_columns)}"
            )

        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.history_path = history_path

        # Load scaler only once
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        if self.scaler.n_features_in_ != len(self.feature_columns):
            raise ValueError(
                "Scaler feature count does not match feature columns"
            )

        # Load model only once
        self.model = tf.keras.models.load_model(
            self.model_path
        )

        # Open optimized history file only once
        if self.history_path is None:
            raise ValueError(
                "history_path is required for production inference"
            )

        self.history_file = pq.ParquetFile(
            self.history_path
        )

    def forecast_next_day(self, item_id, store_id):

        required_columns = [
            "item_id",
            "store_id",
            "date"
        ] + self.feature_columns

        series_parts = []

        for row_group in range(
            self.history_file.num_row_groups
        ):

            df = self.history_file.read_row_group(
                row_group,
                columns=required_columns
            ).to_pandas()

            df = df[
                (df["item_id"] == item_id) &
                (df["store_id"] == store_id)
            ]

            if len(df) > 0:
                series_parts.append(df)

            del df

        if not series_parts:
            raise ValueError(
                f"Series not found: {item_id} / {store_id}"
            )

        series_df = pd.concat(
            series_parts,
            ignore_index=True
        )

        series_df["date"] = pd.to_datetime(
            series_df["date"]
        )

        series_df = (
            series_df
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        if len(series_df) < self.sequence_length:
            raise ValueError(
                f"Not enough history. "
                f"Required {self.sequence_length}, "
                f"found {len(series_df)}"
            )

        latest_sequence = series_df.tail(
            self.sequence_length
        ).copy()

        last_available_date = (
            latest_sequence["date"].max()
        )

        forecast_date = (
            last_available_date +
            pd.Timedelta(days=1)
        )

        X_raw = latest_sequence[
            self.feature_columns
        ].copy()

        # Missing price is represented as 0
        if "sell_price" in X_raw.columns:
            X_raw["sell_price"] = (
                X_raw["sell_price"].fillna(0)
            )

        if X_raw.isna().sum().sum() > 0:
            raise ValueError(
                "NaN values found in inference features"
            )

        if np.isinf(X_raw.to_numpy()).sum() > 0:
            raise ValueError(
                "Inf values found in inference features"
            )

        # Scale using already loaded scaler
        X_scaled = self.scaler.transform(
            X_raw
        ).astype(np.float32)

        X_scaled = X_scaled.reshape(
            1,
            self.sequence_length,
            len(self.feature_columns)
        )

        # Predict using already loaded model
        prediction = self.model(
            tf.convert_to_tensor(X_scaled),
            training=False
        ).numpy()

        prediction = float(
            prediction[0, 0]
        )

        if not np.isfinite(prediction):
            raise ValueError(
                "Model produced NaN or Inf prediction"
            )

        forecast = max(
            0.0,
            prediction
        )

        return {
            "item_id": item_id,
            "store_id": store_id,
            "last_available_date":
                last_available_date.strftime("%Y-%m-%d"),
            "forecast_date":
                forecast_date.strftime("%Y-%m-%d"),
            "forecast": forecast
        }
