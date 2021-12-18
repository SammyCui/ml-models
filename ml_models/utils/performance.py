import numpy as np


def TP(pred_labels: np.array, true_labels: np.array):
    return np.sum(np.logical_and(pred_labels == 1, true_labels == 1))


def TN(pred_labels: np.array, true_labels: np.array):
    return np.sum(np.logical_and(pred_labels == -1, true_labels == -1))


def FP(pred_labels: np.array, true_labels: np.array):
    return np.sum(np.logical_and(pred_labels == 1, true_labels == -1))


def FN(pred_labels: np.array, true_labels: np.array):
    return np.sum(np.logical_and(pred_labels == -1, true_labels == 1))


def precision(pred_labels, true_labels):
    tp, fp = TP(pred_labels, true_labels), FP(pred_labels, true_labels)
    return tp / (tp + fp)


def recall(pred_labels, true_labels):
    tp, fn = TP(pred_labels, true_labels), FN(pred_labels, true_labels)
    return tp / (tp + fn)


def accuracy(pred_labels, true_labels):
    tp, tn = TP(pred_labels, true_labels), TN(pred_labels, true_labels)
    return (tp + tn) / len(pred_labels)


if __name__ == '__main__':
    _pred_labels = np.asarray([-1, 1, 1, -1, 1, -1, -1])
    _true_labels = np.asarray([-1, -1, 1, -1, -1, 1, -1])
    print(accuracy(np.array([1,-1,1,-1]), np.array([1,1,1,-1])))
