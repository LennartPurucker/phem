# Potentially useful metrics for evaluation wrapped in an easier to use object
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.validation import _check_y, check_array

if TYPE_CHECKING:
    import pandas as pd


# -- Metric Utils
def make_metric(
    metric_func: Callable,
    metric_name: str,
    maximize: bool,
    classification: bool,
    always_transform_conf_to_pred: bool,
    optimum_value: int,
    pos_label: int = 1,
    requires_confidences: bool = False,
    only_positive_class: bool = False,
):
    """Make a metric that has additional information.

    Parameters
    ----------
    metric_func: Callable
        The metric function to call.
        We expect it to be metric_func(y_true, y_pred) with y_pred potentially being
        probabilities instead of classes.
    metric_name: str
        Name of the metric
    maximize: bool
        Whether to maximize the metric or not
    classification: bool
        If it is a classification metric or not
    always_transform_conf_to_pred: bool
        Set to Ture if the metric can not handle confidences and only accepts predictions (only for classification)
    optimum_value: int
        The maximal value the metric can reach (used to compute the loss).
    pos_label: int, default=1
        Index of the label used as positive label (relevant only for binary classification metrics)
    requires_confidences: bool, default=False
        If the metric requires confidences.
    only_positive_class: bool, default=False
        Only relevant if requires_confidences is True. If only_positive_class is true, only the positive class
        values are passed. This is only needed for binary classification. Ignored if always_transform_conf_to_pred is
        True.
    """
    return AbstractMetric(
        metric_func,
        metric_name,
        maximize,
        classification,
        always_transform_conf_to_pred,
        optimum_value,
        pos_label,
        requires_confidences,
        only_positive_class,
    )


class AbstractMetric:
    """Abstract Metric used in some codes.

    We transform confidences to prediction if needed for a metric and the model does it not by itself.
    Thereby, we assume y_true to be integers because we transform y_pred into integers as well.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(
        self,
        metric,
        name,
        maximize,
        classification,
        transform_conf_to_pred,
        optimum_value,
        pos_label,
        requires_confidences,
        only_positive_class,
    ):
        self.metric = metric
        self.maximize = maximize
        self.name = name
        self.classification = classification
        self.transform_conf_to_pred = transform_conf_to_pred
        self.optimum_value = optimum_value
        self.pos_label = pos_label
        self.threshold = 0.5
        self.requires_confidences = requires_confidences
        self.only_positive_class = only_positive_class

    def __call__(
        self,
        y_true: pd.DataFrame | np.ndarray,
        y_pred: pd.DataFrame | np.ndarray,
        to_loss: bool = False,
        checks=True,
    ):
        """Parameters
        ----------
        y_true: array-like
            ground truth, assumed to be integers!
        y_pred: array-like
            Either confidences/probabilities matrix (n_samples, n_classes) or prediction vector (n_samples, )
            If not classification, only prediction vector is allowed for now.
            If confidences, we expect the order of n_classes to be identical to the order of np.unique(y_true).
        to_loss: bool
            Whether to return the loss or not
        """
        # -- Input validation
        if checks:
            y_true = _check_y(y_true)

        if not self.classification:
            if checks:
                y_pred = _check_y(y_pred, y_numeric=True)
        elif y_pred.ndim == 1:
            if checks:
                y_pred = _check_y(y_pred)

                if self.requires_confidences:
                    raise ValueError(
                        "Confidences are needed for this metric but predictions are passed."
                    )
        elif y_pred.ndim == 2:
            if checks:
                y_pred = check_array(y_pred)

            # - Special case if metric can not handle confidences
            if self.transform_conf_to_pred:
                y_pred = np.argmax(y_pred, axis=1)
            elif self.only_positive_class:
                y_pred = y_pred[:, self.pos_label]

        else:
            raise ValueError(f"y_pred has to many dimensions! Found ndim: {y_pred.ndim}")

        # --- Call metric
        metric_value = self.metric(y_true, y_pred)

        # --- Return
        if to_loss:
            return self.to_loss(metric_value)

        return metric_value

    def to_loss(self, metric_value):
        # General Purpose Loss: the absolute difference to the optimum
        #   -> smaller is always better
        return abs(self.optimum_value - metric_value)

    def inverse_loss(self, loss_value):
        if (self.optimum_value == 0) and (not self.maximize):
            return loss_value

        # FIXME: this ignores negative metric_values or optima
        return self.optimum_value - loss_value
