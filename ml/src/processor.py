from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class Processor:
    '''
    aux function to pre process data to train and test
    '''
    dataframe: Optional[pd.DataFrame] = None

    def create_sliding_windows(self, *,
                              data: np.ndarray,
                              target_col_idx: int = 0,
                              window_size: int = 14
                              ) -> tuple[np.ndarray, np.ndarray]:
        #feature matrix and target
        X, y = [], []

        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, target_col_idx])

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
        return X_arr, y_arr