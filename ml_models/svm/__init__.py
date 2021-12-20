import os
import sys

from svm.svc import SVM

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
__all__ = ['SVM']