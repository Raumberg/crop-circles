from enum import Enum
from typing import Any, Optional, Union, cast, Tuple, Dict

import numpy as np
import scipy.special
import sklearn.metrics as skm

import deeputils as du

class PredictionType(Enum):
    LOGITS = 'logits'
    PROBS = 'probs'

class TaskType(Enum):
    BINARY = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value

class Metrica:
    def __init__(
            self,
            task_type: TaskType,
            prediction_type: Optional[PredictionType] = None,
            **kwargs: Any
        ):
        pass

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
    ) -> float:
        rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
        if std is not None:
            rmse *= std
        return rmse

    @staticmethod
    def _get_labels_and_probs(
        y_pred: np.ndarray, 
        task_type: TaskType, 
        prediction_type: Optional[PredictionType]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        if prediction_type is None:
            return y_pred, None

        if prediction_type == PredictionType.LOGITS:
            probs = (
                scipy.special.expit(y_pred)
                if task_type == TaskType.BINARY
                else scipy.special.softmax(y_pred, axis=1)
            )
        elif prediction_type == PredictionType.PROBS:
            probs = y_pred
        else:
            du.raise_unknown('prediction_type', prediction_type)

        assert probs is not None
        labels = np.round(probs) if task_type == TaskType.BINARY else probs.argmax(axis=1)
        return labels.astype('int64'), probs

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: Union[str, TaskType],
        prediction_type: Optional[Union[str, PredictionType]],
        y_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
        task_type = TaskType(task_type)
        if prediction_type is not None:
            prediction_type = PredictionType(prediction_type)

        if task_type == TaskType.REGRESSION:
            assert prediction_type is None
            assert 'std' in y_info
            rmse = Metrica.calculate_rmse(y_true, y_pred, y_info['std'])
            result = {'rmse': rmse}
        else:
            labels, probs = Metrica._get_labels_and_probs(y_pred, task_type, prediction_type)
            result = cast(
                Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
            )
            if task_type == TaskType.BINCLASS:
                result['roc_auc'] = skm.roc_auc_score(y_true, probs)
        return result
