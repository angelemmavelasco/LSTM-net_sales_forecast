from dataclasses import dataclass
from typing import Optional

from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

@dataclass
class Model:
    lstm_units: int = 64
    learning_rate: float = 0.001
    model: Optional[Sequential] = None

    def build(
            self, *,
            timesteps: int,
            n_features: int,
    ) -> Sequential:
        model = Sequential(
            [
                Input(shape=(timesteps, n_features), name="time_series_input"),
                LSTM(self.lstm_units, activation="tanh", name="lstm_layer"),
                Dense(1, activation="linear", name="prediction_output"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )
        self.model = model
        return self.model

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_split: float = 0.15,
    ):
        if self.model is None:
            _, timesteps, n_features = X_train.shape
            self.build(timesteps, n_features)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False,
            verbose=1,
        )
        return history
