from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _check_y, check_array


# --- Class Stuff
class DiversityMetric:
    """An abstract class for metrics to measure diversity."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(
        self,
        metric_func,
        metric_name,
        requires_weights=False,
        requires_y_ensemble_pred=False,
        requires_raw_predictions=False,
        more_diversity_if_higher=False,
        single_model_ensemble_default=1.0,
    ):
        self.metric_func = metric_func
        self.name = metric_name
        self.requires_weights = requires_weights
        self.requires_y_ensemble_pred = requires_y_ensemble_pred
        self.y_ensemble_pred_signature_name = "y_pred_ensemble"
        self.weights_signature_name = "weights"
        self.requires_raw_predictions = requires_raw_predictions
        self.more_diversity_if_higher = more_diversity_if_higher
        self.single_model_ensemble_default = single_model_ensemble_default

    def __call__(
        self,
        y_true: np.ndarray,
        Y_pred_base_models: list[np.ndarray],
        y_pred_ensemble: np.ndarray = None,
        weights: np.ndarray = None,
        checks=True,
        transform_confidences: bool = False,
    ):
        """Call which determines what to pass to the metric function.

        Moreover, we define the diversity to be 1 if the ensemble consists of only one model.
        In our case, lower scores of the metric always correspond to more diverse ensembles.
        And 1 means the least amount of diversity.

        Parameters
        ----------
        Y_pred_base_models: List[ndarray] (n_base_models, n_samples) or (n_base_models, n_samples, n_classes)
            Contains the (probability) predictions of each base model in the ensemble
        y_true: ndarray (n_samples,)
            Ground truth / target vector
        y_pred_ensemble: ndarray (n_samples, ) or (n_samples, n_classes)
            Contains the (probability) predictions of the ensemble
        weights: ndarray (n_base_models, ), default=None
            The weights used to compute the weighted average.
        """
        if len(Y_pred_base_models) == 1:
            return self.single_model_ensemble_default

        if checks:
            # -- Input validation
            y_true = _check_y(y_true)

        # -- Build Kwargs that the metric needs
        kwargs = {}

        # - Handle Weight
        if self.requires_weights:
            if weights is None:
                raise ValueError("Metric requires weights but weights are None.")
            if len(weights) != len(Y_pred_base_models) and sum(weights != 0) != len(
                Y_pred_base_models
            ):
                raise ValueError("Weight vector and base model count do not match!")

            kwargs[self.weights_signature_name] = weights

        if self.requires_y_ensemble_pred:
            y_pred_ensemble = check_array(y_pred_ensemble)
            kwargs[self.y_ensemble_pred_signature_name] = y_pred_ensemble

        # -- Pass data to metric
        if isinstance(self, OrdinalClassificationDiversityMetric):
            if checks:
                # Verify Dimensions
                if (y_pred_ensemble is not None) and y_pred_ensemble.ndim != 2:
                    raise ValueError(
                        "OrdinalClassificationDiversityMetric requires prediction probabilities as input! "
                        "Ensemble Predictions (y_pred_ensemble) has more or less than 2 dimensions."
                    )

                if any(Y_pred_base_model.ndim != 2 for Y_pred_base_model in Y_pred_base_models):
                    raise ValueError(
                        "OrdinalClassificationDiversityMetric requires prediction probabilities as input! "
                        "The predictions of at least one base model in Y_pred_base_models is not 2 dimensional",
                    )

                if (y_pred_ensemble is not None) and any(
                    bm.shape[1] < y_pred_ensemble.shape[1] for bm in Y_pred_base_models
                ):
                    raise ValueError(
                        "y_true has more classes than some base model has predicted for."
                    )

                if (y_pred_ensemble is not None) and any(
                    bm.shape[1] != y_pred_ensemble.shape[1] for bm in Y_pred_base_models
                ):
                    raise ValueError(
                        "The number of classes does not match between ensemble and some base model predictions."
                    )

                # Verify Array
                Y_pred_base_models = [
                    check_array(Y_pred_base_model) for Y_pred_base_model in Y_pred_base_models
                ]

            # Get Score
            diversity_score = self.metric_func(y_true, Y_pred_base_models, **kwargs)

        elif isinstance(self, NonOrdinalClassificationDiversityMetric):
            if checks:
                Y_pred_base_models = [
                    _check_y(Y_pred_base_model) for Y_pred_base_model in Y_pred_base_models
                ]

            diversity_score = self.metric_func(y_true, Y_pred_base_models, **kwargs)

        else:
            raise NotImplementedError()

        return diversity_score


class OrdinalClassificationDiversityMetric(DiversityMetric):
    def __init__(
        self,
        metric_func,
        metric_name,
        requires_weights,
        requires_y_ensemble_pred,
        more_diversity_if_higher,
        single_model_ensemble_default,
    ):
        super().__init__(
            metric_func,
            metric_name,
            requires_weights=requires_weights,
            requires_y_ensemble_pred=requires_y_ensemble_pred,
            requires_raw_predictions=False,
            more_diversity_if_higher=more_diversity_if_higher,
            single_model_ensemble_default=single_model_ensemble_default,
        )


class NonOrdinalClassificationDiversityMetric(DiversityMetric):
    def __init__(
        self,
        metric_func,
        metric_name,
        requires_weights,
        requires_y_ensemble_pred,
        more_diversity_if_higher,
        single_model_ensemble_default,
    ):
        super().__init__(
            metric_func,
            metric_name,
            requires_weights=requires_weights,
            requires_y_ensemble_pred=requires_y_ensemble_pred,
            requires_raw_predictions=True,
            more_diversity_if_higher=more_diversity_if_higher,
            single_model_ensemble_default=single_model_ensemble_default,
        )


# --- Metric Usability Stuff
def make_diversity_metric(
    metric_type: str,
    metric_name: str,
    metric_func: Callable,
    requires_weights: bool = False,
    requires_y_ensemble_pred: bool = False,
    more_diversity_if_higher: bool = False,
    single_model_ensemble_default: float = 1,
):
    """Create an abstract usable diversity metric based on the input.

    Parameters
    ----------
    metric_type: str in {"OrdinalClassification", "NonOrdinalClassification", "Regression"}
        The type of diversity metric. Determines what type of input is passed to the metric function.
    metric_name: str
        Name of the metric.
    metric_func: Callable
        The function that is called to compute the diversity.
        We assume the function f to have a specific signature based on "metric_type":

            - "OrdinalClassification": f(y_pred_ensemble, Y_pred_base_models, y_true)

        With:
            - y_pred_ensemble: ndarray, (n_samples, ) or (n_samples, n_classes)
                Contains the (probability) predictions of the ensemble
            - Y_pred_base_models: List[ndarray], (n_samples, n_base_models) or (n_samples, n_classes, n_base_models)
                Contains the (probability) predictions of each base model
            - y_true: ndarray, (n_samples,)
                Ground truth / target vector
    requires_weights: bool, default=False
        If the metric requires the weights (for weighted average based metrics).
    requires_y_ensemble_pred: Bool, default=False
        If the metric requires the predictions from the ensemble.
    more_diversity_if_higher: bool, default=False
        If false, the diversity metrics represents more ensemble diversity with lower values. Otherwise, set this to
        true.
    single_model_ensemble_default: float, default=1
        What value to return if the ensemble consists only of a single model.


    Returns:
    -------
    Diversity Metric Depending on the Input

    """
    if metric_type == "OrdinalClassification":
        d_m = OrdinalClassificationDiversityMetric(
            metric_func,
            metric_name,
            requires_weights,
            requires_y_ensemble_pred,
            more_diversity_if_higher,
            single_model_ensemble_default,
        )

    elif metric_type == "NonOrdinalClassification":
        d_m = NonOrdinalClassificationDiversityMetric(
            metric_func,
            metric_name,
            requires_weights,
            requires_y_ensemble_pred,
            more_diversity_if_higher,
            single_model_ensemble_default,
        )
    else:
        raise NotImplementedError()

    return d_m


# - Error Correlation
def average_loss_correlation(
    y_true, Y_pred_base_models: list[np.ndarray], weights=None, aggregation_method=np.mean
):
    """Loss correlation implementation. Compute the correlation of the loss for each instance w.r.t. the true class
    between all pairs of base models.

    """
    # TODO: add transformed labels as parameter in signature
    lb = LabelBinarizer()
    lb.fit(y_true)

    # Try to fix bad split situation
    # TODO: make this a global or model based fix somehow, pass classes?
    if len(lb.classes_) != Y_pred_base_models[0].shape[-1]:
        lb.classes_ = list(range(Y_pred_base_models[0].shape[-1]))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels,
            transformed_labels,
            axis=1,
        )

    loss_per_bm = [
        not_aggregated_loss(transformed_labels, bm_pred) for bm_pred in Y_pred_base_models
    ]

    # Get pairwise correlations
    # following https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/
    corr_matrix = np.abs(np.corrcoef(loss_per_bm))

    # Set all values to nan except for the upper triangle of the matrix
    upper = corr_matrix[np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)]

    # Replace nan values with 0 because it is equal to no correlation if something is just a constant predictor
    upper = np.nan_to_num(upper)

    return aggregation_method(upper)


def not_aggregated_loss(transformed_labels, y_pred, eps=1e-15):
    """Sklearn log loss adapted to stop early."""
    return 1 - (transformed_labels * y_pred).sum(axis=1)


# --- Initialization
LossCorrelation = make_diversity_metric(
    "OrdinalClassification", "Average Loss Correlation", average_loss_correlation
)
