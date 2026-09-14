import numpy as np
from typing import Optional

from dataclasses import dataclass
@dataclass
class Node:
    characteristic:Optional[int]=None
    threshold: Optional[float] = None
    lside: Optional["Node"] = None
    rside: Optional["Node"] = None
    val: Optional[object] = None
    depth: int = 0
class DecisionTree:
    def __init__(
        self,
        task:str="clsf",
        max_depth:Optional[int]=None,
    ):
        self.max_depth=max_depth
        self.task=task
        self.root=None
    def _gini(self, y):
     if len(y) == 0:
            return 0.0
     _, counts=np.unique(y, return_counts=True)
     probabilities=counts / len(y)
     return 1.0 - np.sum(probabilities ** 2)
    def _mse(self, y):
     if len(y) == 0:
        return 0.0
     avg = np.mean(y)
     return np.mean((y - avg) ** 2)
    def _split(self, X, y, characteristic, threshold):
     left_mask = X[:, characteristic] <= threshold
     right_mask = X[:, characteristic] > threshold
     return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    def _split_score(self, y_left, y_right):
     n = len(y_left) + len(y_right)
     if n == 0:
        return 0.0
     left_weight = len(y_left) / n
     right_weight = len(y_right) / n
     score = (
        left_weight * self._gini(y_left)
        + right_weight * self._gini(y_right)
    )
     return score
    def _get_thresholds(self, values):
     values = np.sort(np.unique(values))
     if len(values) < 2:
        return np.array([])
     return (values[:-1] + values[1:]) / 2
    def _best_split(self, X, y):
     best_score = float("inf")
     best_characteristic = None
     best_threshold = None
     n_features = X.shape[1]
     for characteristic in range(n_features):
        values = X[:, characteristic]
        thresholds = self._get_thresholds(values)
        for threshold in thresholds:
            _, _, y_left, y_right = self._split(
                X, y, characteristic, threshold
            )
            if len(y_left) == 0 or len(y_right) == 0:
               continue
            score = self._split_score(y_left, y_right)
            if score < best_score:
                best_score = score
                best_characteristic = characteristic
                best_threshold = threshold
     return best_characteristic, best_threshold, best_score
