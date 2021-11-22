from typing import Union
import numpy as np
from utils import model_base
import pandas as pd


class SVM(model_base.BaseModel):
    """
    SVM implemented following sklearn's convention
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self,
                 C: float = 1,
                 reg: str = 'l2',
                 kernal: str = 'linear',
                 class_weight: Union[dict, str] = 'balanced',
                 slack: str = 'slack',
                 S: float = 0.1,
                 max_iters: int = -1,
                 step_size: float = 0.01):
        """

        :param C: Strength of regularization
        :param reg: method of regularization
        :param kernal: type of kernal method
        :param class_weight: dict or 'balanced' for adjusting class imbalanceness.
                Detail refer to sklearn svm implementation
        :param slack: slack variable method: either slack or 'alternative'. Default 'slack'
        :param S: strength of slack variable
        :param max_iters: limit on solver's num of iteration
        :param step_size:
        """

        self.C = C
        self.reg = reg
        self.kernal = kernal
        self.class_weight = class_weight
        self.slack = slack
        self.S = S
        self.max_iters = max_iters
        self.step_size = step_size

    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.DataFrame]):

        def computeW(_X, _y, _lambdas):
            _w = np.zeros((X.shape[1]))
            for idx, data_pt in enumerate(_X):
                _w += data_pt * _lambdas[idx] * _y[idx]
            return _w

        lambdas = np.full(X.shape[0], 1.0)
        w = np.zeros((X.shape[1]))
        for _ in range(self.max_iters):
            lambdas += self.step_size * (1 - y * np.dot(X, w))
            lambdas = np.where(lambdas < 0, 0, lambdas)
            w = computeW(X, y, lambdas)

        sv = X[lambdas > 0, :]
        sv_lambdas = lambdas[lambdas > 0]
        self.sv = sv
        self.sv_lambdas = sv_lambdas
        self.sv_labels = np.hstack((sv, y[lambdas > 0].reshape(-1, 1)))
        self.b = -np.dot(w, X[0])

        return self

    def predict(self, X: Union[np.array, pd.DataFrame]):
        result = self.b
        sv, label = self.sv[:, :-1], self.sv[:, -1]
        for idx, v in enumerate(sv):
            result += self.sv_lambdas[idx] * kernel(v, X) * label[idx]

        return result


def kernel(x: np.array, v: np.array):
    return np.exp(np.dot(-np.transpose(x - v), (x - v)))


if __name__ == '__main__':
    dataMat = pd.read_csv('/Users/xuanmingcui/Downloads/hw2dataNorm.csv').to_numpy()
    svm = SVM(max_iters=50)
    svm.fit(dataMat[:40, :-1], dataMat[:40, -1])
    print(svm.sv_lambdas)
