
import tensorflow as tf


def build_lstm_attention_model(
    sequence_length,
    num_features,
    lstm_units=64,
    dense_units=32,
    learning_rate=0.001
):
    """
    Build the production LSTM + Attention forecasting model.
    """

    inputs = tf.keras.Input(
        shape=(sequence_length, num_features),
        name="input_sequence"
    )

    lstm_output = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm"
    )(inputs)

    attention_output = tf.keras.layers.Attention(
        name="attention"
    )(
        [lstm_output, lstm_output]
    )

    pooled_output = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(attention_output)

    dense_output = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="dense"
    )(pooled_output)

    output = tf.keras.layers.Dense(
        1,
        name="sales_forecast"
    )(dense_output)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="lstm_attention_forecaster"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model
