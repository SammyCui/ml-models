from typing import Union
import numpy as np
from utils import performance
from utils.model_base import BaseModel, ModelNotTrainedError
import pandas as pd
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing


class SVM(BaseModel):
    """
    SVM implemented following sklearn's convention
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self,
                 C: float = 1,
                 kernel: str = 'rbf',
                 max_iters: int = -1,
                 step_size: float = 0.001,
                 gamma: str = 'auto',
                 degree: int = 2,
                 momentum: float = 0.007):
        """

        :param C: Strength of regularization
        :param kernal: type of kernal method
        :param max_iters: limit on solver's num of iteration
        :param step_size: step size for each learning iteration
        :param gamma: kernel coefficient; only for kernel=rbf, poly or sigmoid
        xx
        """
        self.C = C
        self.kernel = kernel
        self.max_iters = max_iters
        # TODO: step_size = 1 / K(X1,X2) will converge
        self.step_size = step_size
        self.gamma = gamma
        self.degree = degree
        self.sv, self.labels, self.alpha, self.b, self.eval_scores = None, None, None, None, []
        self.is_trained = False
        self.momentum = momentum
        # testing
        self.changes = []
        self.scores = []

    def fit(self, X, y, X_val, y_val):
        self.is_trained = True
        # TODO: randomize?
        self.alpha = np.zeros(X.shape[0])
        ones = np.ones(X.shape[0])
        constant = np.outer(y, y) * getattr(SVM, self.kernel)(self, X, X)
        # TODO: Stochastic update
        # Gradient descent with momentumR
        last_change = 0
        for _ in range(self.max_iters):
            gradient = ones - constant.dot(self.alpha)
            self.alpha += self.step_size * gradient + self.momentum * last_change
            self.changes.append(np.mean(gradient))
            last_change = ones - gradient
            self.alpha = np.where(self.alpha < 0, 0, self.alpha)
            # Slack
            self.alpha[self.alpha > self.C] = self.C

            index = np.where(self.alpha > 0 & (self.alpha < self.C))[0]
            b_i = y[index] - (self.alpha * y).dot(getattr(SVM, self.kernel)(self, X, X[index]))
            self.b = np.mean(b_i)
            self.labels, self.sv = y, X

            score = performance.accuracy(self.predict(X_val), y_val)
            self.scores.append(score)

        self.sv, self.labels, self.alpha = X[self.alpha > 0, :], y[self.alpha > 0], self.alpha[self.alpha > 0]

        return self

    def predict(self, X: Union[list, np.array, pd.DataFrame, pd.Series]):
        """
        return classification result
        :param X: target array-like shape(1,0) to predict
        """
        if not self.is_trained:
            raise ModelNotTrainedError()
        decision_list = (self.alpha * self.labels).dot(getattr(SVM, self.kernel)(self, self.sv, X)) + self.b
        return np.sign(decision_list)

    def linear(self, x: np.array, v: np.array) -> np.array:
        """
        linear kernel
        """
        return np.dot(x, v)

    def rbf(self, x: np.array, v: np.array) -> np.array:
        """
        rbf kernel
        """
        if self.gamma == 'auto':
            gamma = 1 / (len(x) * x.var()) if x.var() != 0 else 1
        else:
            gamma = 1 / len(x)

        return np.exp(- gamma * np.linalg.norm(x[:, np.newaxis] - v[np.newaxis, :], axis=2) ** 2)

    def poly(self, x, v):
        return (1 + x.dot(v.T)) ** self.degree


# TODO: slack variable, multiclass, more kernel func,

if __name__ == '__main__':
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    data_train = pd.read_csv(os.path.join(fileDir, '../data/SPECTF_train.csv'), header=None).to_numpy()
    data_test = pd.read_csv(os.path.join(fileDir, '../data/SPECTF_test.csv'), header=None).to_numpy()
    X_train, X_test = data_train[:, 1:], data_test[:, 1:]
    y_train, y_test = data_train[:, 0], data_test[:, 0]
    y_train, y_test = np.where(y_train == 0, -1, y_train), np.where(y_test == 0, -1, y_test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(X_train)
    normalized_val_X, normalized_test_X = normalizer.transform(X_val), normalizer.transform(X_test)

    svm = SVM(C=1, max_iters=2000)
    svm.fit(normalized_train_X, y_train, normalized_val_X, y_val)
    print('Accuracy of SVM: ', performance.accuracy(svm.predict(normalized_test_X), y_test))
    benchmark_svc = SVC(max_iter=1000)
    benchmark_svc.fit(normalized_train_X, y_train)
    benchmark_svc_prediction = benchmark_svc.predict(normalized_test_X)
    print('Accuracy of sklearn.svm.SVC: ', performance.accuracy(benchmark_svc_prediction, y_test))
    plt.plot(svm.scores)
    plt.show()
    plt.plot(svm.changes)
    plt.show()
