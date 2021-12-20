import copy
from typing import Union
import numpy as np
from utils import performance
from utils.model_base import BaseModel, ModelNotTrainedError, ModelAlreadyTrainedError
import pandas as pd
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
import matplotlib


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
                 momentum: float = 0.007,
                 tol: float = 0.01,
                 eps: float = 0.01):
        """

        :param C: Strength of regularization
        :param kernal: type of kernal method
        :param max_iters: limit on solver's num of iteration
        :param step_size: step size for each learning iteration
        :param gamma: kernel coefficient; only for kernel=rbf, poly or sigmoid
        :param tol: error tolerance for SMO. Only used when using fit_smo
        :param eps: epsilon. alpha tolerance for SMO. Only used when using fit_smo

        """
        self.C = C
        self.kernel = kernel
        self.max_iters = max_iters
        # TODO: step_size = 1 / K(X1,X2) will converge
        self.step_size = step_size
        self.gamma = gamma
        self.degree = degree
        self.sv, self.labels, self.alpha, self.b, self.eval_scores = None, None, None, 0, []
        self.is_trained = False
        self.momentum = momentum
        self.tol = tol
        self.eps = eps
        # testing
        self.changes = []
        self.scores = []
        self.obj_scores = []

    def fit(self, X, y, X_val, y_val):
        if not self.is_trained:
            self.is_trained = True
        else:
            raise ModelAlreadyTrainedError()
        # TODO: randomize?
        self.alpha = np.zeros(X.shape[0])
        ones = np.ones(X.shape[0])
        constant = np.outer(y, y) * getattr(SVM, self.kernel)(self, X, X)
        # TODO: Stochastic update
        # Gradient descent with momentum
        last_change = 0
        for _ in range(self.max_iters):
            gradient = ones - constant.dot(self.alpha)
            self.alpha += self.step_size * gradient + self.momentum * last_change
            self.changes.append(np.mean(gradient))
            last_change = gradient
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

    def fit_smo(self, X, y):
        if not self.is_trained:
            self.is_trained = True
        else:
            raise ModelAlreadyTrainedError()

        def take_step(i1, i2):
            if i1 == i2:
                return 0
            alph1, alph2 = self.alpha[i1], self.alpha[i2]
            y1, y2 = y[i1], y[i2]
            e1, e2 = errors[i1], errors[i2]
            s = y1 * y2

            # calculate bounds applied to alpha2
            if y1 != y2:
                L = max(0, alph2 - alph1)
                H = min(self.C, self.C + alph2 - alph1)
            else:
                L = max(0, alph1 + alph2 - self.C)
                H = min(self.C, alph1 + alph2)
            if L == H:
                return 0

            k11 = getattr(SVM, self.kernel)(self, np.array([X[i1]]), np.array([X[i1]]))
            k12 = getattr(SVM, self.kernel)(self, np.array([X[i1]]), np.array([X[i2]]))
            k22 = getattr(SVM, self.kernel)(self, np.array([X[i2]]), np.array([X[i2]]))

            eta = 2 * k12 - k11 - k22
            if eta < 0:
                # TODO: Check
                a2 = alph2 - y2 * (e1 - e2) / eta
                if a2 <= L:
                    a2 = L
                elif a2 >= H:
                    a2 = H
            else:
                alphas_new = copy.deepcopy(self.alpha)
                alphas_new[i2] = L
                lobj = self.objective_function(X, y, alphas_new)
                alphas_new[i2] = H
                hobj = self.objective_function(X, y, alphas_new)
                if lobj < hobj - self.eps:
                    a2 = H
                elif lobj > hobj + self.eps:
                    a2 = L
                else:
                    a2 = alph2

            # bound alpha between 0 <= alpha <= C, allow close-to-0 computational error to be 0
            if a2 < 1e-8:
                a2 = 0
            elif a2 > self.C - 1e-8:
                a2 = self.C

            # if the current alpha pair does not need to be optimized, skip
            if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
                return 0

            # calculate new alpha 1 based on alpha2
            a1 = alph1 + s * (alph2 - a2)
            # update threshold
            b1 = e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
            b2 = e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b

            # b1, b2 is valid when a1(new alpha1), a2 is not at the bounds
            # if they are both on the bounds, take the mean
            if 0 < a1 < self.C:
                b_new = b1
            elif 0 < a2 < self.C:
                b_new = b2
            else:
                b_new = (b1 + b2) / 2

            # update alpha1 and alpha2
            self.alpha[i1], self.alpha[i2] = a1, a2
            # update saved errors
            # if optimized alphas are not on the bounds, set their error to 0
            for index, alph in zip([i1, i2], [a1, a2]):
                if 0 < alph < self.C:
                    errors[index] = 0

            non_opt = [n for n in range(len(X)) if (n != i1 and n != i2)]
            errors[non_opt] = errors[non_opt] + \
                              y1 * (a1 - alph1) * getattr(SVM, self.kernel)(self, np.array([X[i1]]), X[non_opt]) + \
                              y2 * (a2 - alph2) * getattr(SVM, self.kernel)(self, np.array([X[i2]]),
                                                                            X[non_opt]) + self.b - \
                              b_new

            # Update model threshold
            self.b = b_new
            return 1

        def examine_example(i2):
            y2 = y[i2]
            alpha2 = self.alpha[i2]
            e2 = errors[i2]
            r2 = e2 * y2
            if (r2 < - self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
                # non-boundary set
                if ((self.alpha < self.C) & (self.alpha > 0)).sum() > 1:
                    # second choice heuristic for selecting working set
                    if e2 > 0:
                        i1 = np.argmin(errors)
                    else:
                        i1 = np.argmax(errors)
                    if take_step(i1, i2):
                        return 1
                # loop through the non-boundary working set from a random point
                idx_non_boundary = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
                for i1 in np.roll(idx_non_boundary, np.random.choice(np.arange(len(X)))):
                    if take_step(i1, i2):
                        return 1

                for i1 in np.roll(np.arange(len(X)), np.random.choice(np.arange(len(X)))):
                    if take_step(i1, i2):
                        return 1

            return 0

        self.alpha = np.zeros(X.shape[0])
        self.sv = X
        self.labels = y
        errors = self.predict(X) - y
        num_iters = 0
        num_changed = 0
        examine_all = 1

        while ((num_changed > 0) or examine_all) and ((num_iters < self.max_iters) or self.max_iters == -1):
            num_changed = 0
            if examine_all:
                for i in range(X.shape[0]):
                    result = examine_example(i)
                    num_changed += result
                    if result:
                        obj_score = self.objective_function(X, y, self.alpha)
                        self.obj_scores.append(obj_score)
            else:
                # loop over working set when alphas are not 0 or C
                for i in np.where((self.alpha != 0) & (self.alpha != self.C))[0]:
                    result = examine_example(i)
                    num_changed += result
                    if result:
                        obj_score = self.objective_function(X, y, self.alpha)
                        self.obj_scores.append(obj_score)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
            num_iters += 1

        return self

    def objective_function(self, X, y, alphas):
        """
        SVM objective function
        """

        return np.sum(alphas) - 0.5 * np.sum(
            (y[:, None] * y[None, :]) * getattr(SVM, self.kernel)(self, X, X) * (alphas[:, None] * alphas[None, :]))

    def predict(self, X):
        """
        return classification result
        :param X: target array-like shape(1,0) to predict
        """
        if not self.is_trained:
            raise ModelNotTrainedError()
        decision_list = (self.alpha * self.labels).dot(getattr(SVM, self.kernel)(self, self.sv, X)) + self.b
        return np.sign(decision_list).flatten()

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


# TODO: multiclass, more kernel func,

if __name__ == '__main__':
    matplotlib.matplotlib_fname()
    """
    Tested on two datasets, SPECTF dataset from UC Irvine, and sklearn breast cancer dataset
    Uncommon any of the following chunks to 
    """
    # fileDir = os.path.dirname(os.path.realpath('__file__'))
    # data_train = pd.read_csv(os.path.join(fileDir, '../data/SPECTF_train.csv'), header=None).to_numpy()
    # data_test = pd.read_csv(os.path.join(fileDir, '../data/SPECTF_test.csv'), header=None).to_numpy()
    # X_train, X_test = data_train[:, 1:], data_test[:, 1:]
    # y_train, y_test = data_train[:, 0], data_test[:, 0]
    # y_train, y_test = np.where(y_train == 0, -1, y_train), np.where(y_test == 0, -1, y_test)
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    # normalizer = preprocessing.Normalizer()
    # normalized_train_X = normalizer.fit_transform(X_train)
    # normalized_val_X, normalized_test_X = normalizer.transform(X_val), normalizer.transform(X_test)

    breast_cancer_dataset = datasets.load_breast_cancer()
    bc_X, bc_y = breast_cancer_dataset.data, breast_cancer_dataset.target
    bc_y = np.where(bc_y == 0, -1, bc_y)
    X_train, X_test, y_train, y_test = train_test_split(bc_X, bc_y, test_size=0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(X_train)
    normalized_val_X, normalized_test_X = normalizer.transform(X_val), normalizer.transform(X_test)

    svm = SVM(C=0.5, max_iters=1000)
    svm.fit(X_train, y_train, X_val, y_val)
    result_base = svm.predict(X_test)
    print('Accuracy of SVM: ', performance.accuracy(result_base, y_test))
    print(result_base)
    print(sum(y_test == 1) / len(y_test))
    print()
    benchmark_svc = SVC()
    benchmark_svc.fit(X_train, y_train)
    benchmark_svc_prediction = benchmark_svc.predict(X_test)
    print('Accuracy of sklearn.svm.SVC: ', performance.accuracy(benchmark_svc_prediction, y_test))
    # plt.plot(svm.scores)
    # plt.show()

    svm_smo = SVM(C=0.5, tol=0.01, eps=0.01, max_iters=-1)
    svm_smo.fit_smo(X_train, y_train)
    result_smo = svm_smo.predict(X_test).flatten()
    print('Accuracy of SVM using SMO: ', performance.accuracy(result_smo, y_test))
    print(result_smo)