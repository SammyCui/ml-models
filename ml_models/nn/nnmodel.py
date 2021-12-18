from typing import Union
import pandas as pd
import numpy as np
from nn.basic import Neuron
from utils import model_base, performance


class NeuralNet(model_base.BaseModel):

    def __init__(self):
        pass

    def add_layer(self, layer_type, n_neurons, weights):
        pass

    def forward(self, X):
        pass

    def backpropagation(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X: Union[np.array, pd.DataFrame]):
        pass

