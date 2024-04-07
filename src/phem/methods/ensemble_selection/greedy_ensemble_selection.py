# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.utils import check_random_state

from phem.base_utils.metrics import AbstractMetric
from phem.framework.abstract_weighted_ensemble import AbstractWeightedEnsemble


class EnsembleSelection(AbstractWeightedEnsemble):
    """An ensemble of selected algorithms.

    Fitting an EnsembleSelection generates an ensemble from the models
    generated during the search process. Can be further used for prediction.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric: AbstractMetric
        The metric used to evaluate the models
    n_jobs: int, default=-1
        Number of processes to use for parallelization. -1 means all available.
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    use_best: bool = True
        After finishing all iterations, use the best found ensemble instead of the last found ensemble.
    """

    def __init__(
        self,
        base_models: list[Callable],
        n_iterations: int,
        metric: AbstractMetric,
        n_jobs: int = -1,
        random_state: int | np.random.RandomState | None = None,
        use_best: bool = True,
    ) -> None:
        # base_models = base_models[:20]

        super().__init__(base_models, "predict_proba")
        self.ensemble_size = n_iterations
        self.metric = metric
        self.use_best = use_best

        # -- Code for multiprocessing
        if (n_jobs == 1) or (os.name == "nt"):
            self._use_mp = False
            if os.name == "nt":
                pass
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

        # Behaviour similar to sklearn
        #   int - Deteriministic with succesive calls to fit
        #   RandomState - Successive calls to fit will produce differences
        #   None - Uses numpmys global singleton RandomState
        # https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
        self.random_state = random_state

    def ensemble_fit(
        self, predictions: list[np.ndarray], labels: np.ndarray
    ) -> AbstractWeightedEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if not isinstance(self.metric, AbstractMetric):
            raise ValueError(
                "The provided metric must be an instance of a AbstractMetric, "
                f"nevertheless it is {self.metric}({type(self.metric)})"
            )

        self._fit(predictions, labels)
        self.apply_use_best()
        self._calculate_final_weights()

        # -- Set metadata correctly
        self.iteration_batch_size_ = len(predictions)

        return self

    def _fit(self, predictions: list[np.ndarray], labels: np.ndarray) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []  # contains iteration best
        self.val_loss_over_iterations_ = []  # contains overall best
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )

        for _i in range(ensemble_size):
            # print(i)
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # -- Process Iteration Solutions
            if self._use_mp:
                losses = self._compute_losses_mp(
                    weighted_ensemble_prediction, labels, predictions, s
                )
            else:
                losses = np.zeros(
                    (len(predictions)),
                    dtype=np.float64,
                )

                # Memory-efficient averaging!
                for j, pred in enumerate(predictions):
                    # fant_ensemble_prediction is the prediction of the current ensemble
                    # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                    # We overwrite the contents of fant_ensemble_prediction
                    # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                    np.add(
                        weighted_ensemble_prediction,
                        pred,
                        out=fant_ensemble_prediction,
                    )
                    np.multiply(
                        fant_ensemble_prediction,
                        (1.0 / float(s + 1)),
                        out=fant_ensemble_prediction,
                    )

                    losses[j] = self.metric(labels, fant_ensemble_prediction, to_loss=True)
            # print("LOSSES ",losses)

            # -- Eval Iteration results
            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()

            best = rand.choice(all_best)
            ensemble_loss = losses[best]

            ensemble.append(predictions[best])
            # print("ENSEMBLE ", ensemble)
            trajectory.append(ensemble_loss)
            order.append(best)
            # print("order ", order)

            # Build Correct Validation loss list
            if (
                not self.val_loss_over_iterations_
                or self.val_loss_over_iterations_[-1] > ensemble_loss
            ):
                self.val_loss_over_iterations_.append(ensemble_loss)
            else:
                self.val_loss_over_iterations_.append(self.val_loss_over_iterations_[-1])

            # -- Handle special cases
            if len(predictions) == 1:
                break

            # If we find a perfect ensemble/model, stop early
            if ensemble_loss == 0:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        # print("TRAJECTORY ", trajectory)

    def _compute_losses_mp(self, weighted_ensemble_prediction, labels, predictions, s):
        # -- Process Iteration Solutions
        func_args = (weighted_ensemble_prediction, labels, s, self.metric, predictions)
        pred_i_list = list(range(len(predictions)))

        with ProcessPoolExecutor(self._n_jobs, initializer=_pool_init, initargs=func_args) as ex:
            results = ex.map(_init_wrapper_evaluate_single_solution, pred_i_list)

        return np.array(list(results))

    def apply_use_best(self):
        if self.use_best:
            # Basically from autogluon the code
            min_score = np.min(self.trajectory_)
            idx_best = self.trajectory_.index(min_score)
            self.indices_ = self.indices_[: idx_best + 1]
            self.trajectory_ = self.trajectory_[: idx_best + 1]
            self.ensemble_size = idx_best + 1
            self.validation_loss_ = self.trajectory_[idx_best]
        else:
            self.validation_loss_ = self.trajectory_[-1]
            self.val_loss_over_iterations_ = self.trajectory_

    def _calculate_final_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights


def _pool_init(_weighted_ensemble_prediction, _labels, _sample_size, _score_metric, _predictions):
    global p_weighted_ensemble_prediction, p_labels, p_sample_size, p_score_metric, p_predictions
    p_weighted_ensemble_prediction = _weighted_ensemble_prediction
    p_labels = _labels
    p_sample_size = _sample_size
    p_score_metric = _score_metric
    p_predictions = _predictions


def _init_wrapper_evaluate_single_solution(pred_index):
    return evaluate_single_solution(
        p_weighted_ensemble_prediction,
        p_labels,
        p_sample_size,
        p_score_metric,
        p_predictions[pred_index],
    )


def evaluate_single_solution(weighted_ensemble_prediction, labels, sample_size, score_metric, pred):
    fant_ensemble_prediction = np.add(weighted_ensemble_prediction, pred)
    np.multiply(
        fant_ensemble_prediction, (1.0 / float(sample_size + 1)), out=fant_ensemble_prediction
    )

    return score_metric(labels, fant_ensemble_prediction, to_loss=True)
