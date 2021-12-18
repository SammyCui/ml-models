from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    generic abstract base class for ml models
    """

    @abstractmethod
    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.DataFrame],
            X_val: Union[np.array, pd.DataFrame], y_val: Union[np.array, pd.DataFrame]):
        pass

    @abstractmethod
    def predict(self, X: Union[np.array, pd.DataFrame]):
        pass


class ModelNotTrainedError(Exception):
    def __init__(self, value: str = "The model is not trained yet. Call .fit() first."):
        self.value = value

    def __str__(self):
        return repr(self.value)
