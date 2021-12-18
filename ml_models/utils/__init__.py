import os
import sys

from utils.model_base import ModelNotTrainedError, BaseModel

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

__all__ = ['ModelNotTrainedError', 'BaseModel']
