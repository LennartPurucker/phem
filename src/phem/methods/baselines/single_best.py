from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from phem.base_utils.metrics import AbstractMetric
from phem.framework.abstract_ensemble import AbstractEnsemble


class SingleBest(AbstractEnsemble):
    """Single Best Selector.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric : AbstractMetric function, default=None
        The metric function that should be used to determine the single best algorithm
        Special format required due to OpenML's metrics and our usage.
    predict_method: {"predict", "predict_proba"}, default="predict"
        If "predict" is selected, we determine the SB by passing the raw predictions to the metric.
        If "predict_proba" is selected, we determine the SB by passing the confidences to the metric.
        If the metric can not handle confidences and "predict_proba" is selected, the behavior is identical to
            when "predict" would have been selected.

    Attributes:
    ----------
    best_model_index_ : int
        The Index of Single Best Model found during :meth:`ensemble_fit`.
    selected_indices_: ndarray, shape (n_samples,)
        The selected indices per samples found during :meth:`ensemble_predict`. Used to determine selection performance.
    """

    def __init__(self, base_models, metric: AbstractMetric, predict_method="predict"):
        super().__init__(base_models, predict_method, predict_method_ensemble_predict="dynamic")
        self.predict_method = predict_method
        self.metric = metric
        self.predict_proba_input = "predict_proba"

    def ensemble_fit(
        self, base_models_predictions: list[np.ndarray], labels: np.ndarray
    ) -> AbstractEnsemble:
        """Find the single best algorithm and store it for later."""
        if not isinstance(self.metric, AbstractMetric):
            raise ValueError(
                "The provided metric must be an instance of a AbstractMetric, "
                f"nevertheless it is {self.metric}({type(self.metric)})"
            )

        performances = [
            self.metric(labels, bm_prediction, to_loss=True)
            for bm_prediction in base_models_predictions
        ]
        self.best_model_index_ = np.argmin(performances)
        self.validation_loss_ = np.min(performances)

        return self

    def ensemble_predict(self, base_models_predictions: list[np.ndarray]) -> np.ndarray:
        """Return the predictions of the Single Best."""
        check_is_fitted(self, ["best_model_index_"])

        n_samples = base_models_predictions[0].shape[0]
        self.selected_indices_ = np.full(n_samples, self.best_model_index_)

        return np.array(base_models_predictions)[self.selected_indices_, np.arange(n_samples)]

    def ensemble_predict_proba(self, base_models_predictions: list[np.ndarray]) -> np.ndarray:
        """Return the predictions of the Single Best."""
        check_is_fitted(self, ["best_model_index_"])

        n_samples = base_models_predictions[0].shape[0]
        self.selected_indices_ = np.full(n_samples, self.best_model_index_)

        return base_models_predictions[self.best_model_index_]

    @property
    def _to_save_metadata(self):
        return {
            "best_model_index_": int(self.best_model_index_),
            "validation_loss_": float(self.validation_loss_),
        }
