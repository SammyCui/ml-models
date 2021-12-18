from typing import Union
import numpy as np
from utils import model_base, performance
import pandas as pd


class Neuron:

    def __init__(self):
        self.weights = []
        self.input = []
        self.output = []


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU(x):
    return max(0, x)


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1