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
            ]
        )
        pass
