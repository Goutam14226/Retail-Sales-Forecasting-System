import pickle
import threading

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ai_edge_litert.interpreter import Interpreter


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

        # Load LiteRT model only once
        self.interpreter = Interpreter(
            model_path=self.model_path
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        # Protect the interpreter during concurrent requests
        self.inference_lock = threading.Lock()

        # Open history file only once
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

        # Read only the requested item-store data
        table = pq.read_table(
            self.history_path,
            columns=required_columns,
            filters=[
                ("item_id", "==", item_id),
                ("store_id", "==", store_id),
            ],
        )

        if table.num_rows == 0:
            raise ValueError(
                f"Series not found: {item_id} / {store_id}"
            )

        series_df = table.to_pandas()

        del table

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

        # Scale using the already loaded scaler
        X_scaled = self.scaler.transform(
            X_raw
        ).astype(np.float32)

        X_scaled = X_scaled.reshape(
            1,
            self.sequence_length,
            len(self.feature_columns)
        )

        # Predict using LiteRT
        with self.inference_lock:

            self.interpreter.set_tensor(
                self.input_details[0]["index"],
                X_scaled
            )

            self.interpreter.invoke()

            prediction = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )

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
