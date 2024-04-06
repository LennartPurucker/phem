"""Code for behavior space.

Example:
----------
    # --- Space for Behavior space with two diversity metrics
    bs = BehaviorSpace([
        BehaviorFunction(partial(ErrorCorrelation, checks=False),
                         ["y_true", "Y_pred_base_models"],
                         (0, 1),
                         "proba"),
        BehaviorFunction(partial(Q_statistic, checks=False),
                         ["y_true", "Y_pred_base_models"],
                         (0, 1),
                         "raw")
    ])

"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

ALLOWED_ARGUMENTS = ["y_true", "y_pred_ensemble", "Y_pred_base_models", "weights",
                     "input_metadata", "y_pred"]


# --- Class
class BehaviorFunction:
    """Behavior function class.

    Parameters
    ----------
    function: Callable function returning a float
        Takes as input the arguments specified in  required_arguments and returns a float representing
        the behavior
    required_arguments: List of str
        The required arguments are supposed to be a list of strings defining which arguments are required and in
        which order. Assume that the available arguments' name are identical to the parameter names in the function.
            FIXME: add option to allow alternative parameter names in function
        Available arguments are: {"y_true", "y_pred_ensemble", "Y_pred_base_models", "weights", "base_models_metadata"}
    range_tuple: Tuple of floats
        Define the upper and lower bound of this behavior function
    required_prediction_format: str in {"raw", "proba", "none"}
        Defines in which format the predictions are passed to the function, if at all.
    name: str, default=None
        A name of the behavior function.
    """

    def __init__(self, function: Callable[[], float], required_arguments: list[str],
                 range_tuple: tuple[float, float], required_prediction_format: str, name: str | None = None):
        self.function = function

        if any(v not in ALLOWED_ARGUMENTS for v in required_arguments):
            raise ValueError(f"Not allowed argument name used. Allowed are: {ALLOWED_ARGUMENTS}. Got: {required_arguments}")
        self.required_arguments = required_arguments
        self.range_tuple = range_tuple

        if required_prediction_format not in ["raw", "proba", "none"]:
            raise ValueError("Unknown prediction format. Expected: {}. Got: .{}".format(["raw", "proba", "none"],
                                                                                        required_prediction_format))
        self.required_prediction_format = required_prediction_format
        self.name = "Placeholder" if name is None else name
        self.requires_base_model_metadata = "input_metadata" in required_arguments


class BehaviorSpace:
    """A class for behavior functions for QDO."""

    def __init__(self, behavior_functions: list[BehaviorFunction]):
        self.behavior_functions = behavior_functions

    @property
    def ranges(self):
        return [bf.range_tuple for bf in self.behavior_functions]

    @property
    def n_dims(self):
        return len(self.behavior_functions)

    @property
    def required_prediction_types(self):
        return {bf.required_prediction_format for bf in self.behavior_functions}

    @property
    def requires_base_model_metadata(self):
        return any(bf.requires_base_model_metadata for bf in self.behavior_functions)

    def __call__(self, weights: np.ndarray, y_true: np.ndarray,
                 raw_preds: tuple[np.ndarray, list[np.ndarray]] | None = (None, None),
                 proba_preds: tuple[np.ndarray, list[np.ndarray]] | None = (None, None),
                 input_metadata: Any | None = None) -> list[float]:
        """Get an instance of the behavior space.

        Parameters
        ----------
        y_true: ndarray (n_samples,)
            Ground truth / target vector
        raw_preds, proba_preds: Tuple containing the following for either raw ore proba predictions
            y_pred_ensemble: ndarray (n_samples, ) or (n_samples, n_classes)
                Contains the (probability) predictions of the ensemble
            Y_pred_base_models: List[ndarray] (n_base_models, n_samples) or (n_base_models, n_samples, n_classes)
                Contains the (probability) predictions of each base model in the ensemble
        weights: ndarray (n_base_models, ), default=None
            The weights used to compute the weighted average.
        input_metadata: List[dict], default=None
            The metadata for the metric if required.

        Returns:
        ----------
        behavior_space_instance: np.ndarray
            Float values for each dimension representing the behavior in that dimension.
        """
        raw_y_pred_ensemble, raw_Y_pred_base_models = raw_preds
        proba_y_pred_ensemble, proba_Y_pred_base_models = proba_preds

        b_space_instance = []

        for bf in self.behavior_functions:
            # Can ignore the "none" case as in that case the arguments are not used.
            requires_raw = bf.required_prediction_format == "raw"

            potential_args = {"y_true": y_true,
                                  "y_pred": raw_y_pred_ensemble if requires_raw else proba_y_pred_ensemble,
                                  "y_pred_ensemble": raw_y_pred_ensemble if requires_raw else proba_y_pred_ensemble,
                                  "Y_pred_base_models": raw_Y_pred_base_models if requires_raw else proba_Y_pred_base_models,
                                  "weights": weights,
                                  "input_metadata": input_metadata}

            b_space_instance.append(bf.function(**{para: potential_args[para] for para in bf.required_arguments}))

        return b_space_instance
