from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    generic abstract base class for ml models
    """

    @abstractmethod
    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.DataFrame]):
        pass

    @abstractmethod
    def predict(self, X: Union[np.array, pd.DataFrame]):
        pass



