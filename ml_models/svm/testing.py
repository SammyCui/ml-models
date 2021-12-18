import pandas as pd
import numpy as np
from utils import performance
from sklearn.model_selection import train_test_split
from svm import SVM


if __name__ == '__main__':
    spectf_train = pd.read_csv(r'../data/SPECTF_train.csv')
    spectf_test = pd.read_csv(r'../data/SPECTF_test.csv')



    svm = SVM()
    svm.fit()
