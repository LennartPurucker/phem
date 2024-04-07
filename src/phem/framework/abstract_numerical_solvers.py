from __future__ import annotations

import os
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from sklearn.utils import check_random_state, resample

from phem.base_utils.metrics import AbstractMetric
from phem.framework.abstract_weighted_ensemble import AbstractWeightedEnsemble


class NumericalSolverBase(AbstractWeightedEnsemble):
    """Using a numerical solver to find a weight vector (ensemble weighting).

    Moreover, it includes several options to avoid overfititng for the numerical solver based methods as described in [1]


    [1] Purucker, Lennart, and Joeran Beel. "CMA-ES for Post Hoc Ensembling in AutoML: A Great Success and Salvageable Failure." International Conference on Automated Machine Learning. PMLR, 2023.


    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    score_metric: AbstractMetric
        The metric used to evaluate the models' performance.
    n_iterations: int
        The number of iterations determines the number of evaluations. By default, number of evaluations
        = n_iterations * n_base_models. So this parameter could also be understood as
        "number of evaluations per base model".
    batch_size: int
        The batch size of the optimization algorithm. That is, the number of function evaluations per iteration.
    normalize_weights: str in {"no", "softmax"}, default="no"
        Determines how the weights are normalized.
            - "no", no normalization is applied.
            - "softmax", the weights are normalized using the softmax function.
        If ture, weights are normalized and not combined prediction probabilities. Makes sure that the weights are
        in [0,1] and sum up to 1. Moreover, it makes the weights sparse by rounding down too small weight values to 0.
    trim_weights: str in {"no", "ges-like"}, default="no"
        How to trim the weights to make them sparse. Only used if `normalize_weights != "no"`.
            - "no", no trimming is applied, as used for CMA-ES-Softmax from the paper, if `normalize_weights="softmax"`.
            - "ges-like", as used for CMA-ES-ExplicitGES from the paper, if `normalize_weights="softmax"`.
    single_best_fallback: bool, default=False
        If true, run CMA-ES with re-sampled data and afterwards determine on differently re-sampled data if the final
        result of CMA-ES is truly better than the single best.
    weight_vector_ensemble: bool, default=False
        Aggregate the top50 weights found to increase robustness. Aggregation rule is the unweighted mean.
    start_weight_vector_method: str in {"average_ensemble", "sb"}, default=False
        Determines the initial weight vector for the optimization algorithm.
            - "average_ensemble", all models have the same weight.
            - "sb", the single best model has a weight of 1, all other 0.
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    n_jobs: int, default=-1
        Cores to use for parallelization. If -1, use all available cores. Only works on Linux right now. On windows
        we fall back to n_jobs=1.
    supports_mp: bool, default=True
        If false, the method does not support multiprocessing and thus can not benefit from n_jobs > 1.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        base_models: list[Callable],
        n_iterations: int,
        score_metric: AbstractMetric,
        batch_size: int,
        n_jobs=-1,
        normalize_weights: str = "no",
        trim_weights: str = "no",
        single_best_fallback: bool = False,
        weight_vector_ensemble: bool = False,
        start_weight_vector_method: str = False,
        show_analysis=False,
        supports_mp=True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(base_models, "predict_proba")

        self.n_iterations = n_iterations
        self.n_base_models = len(base_models)
        self.batch_size = batch_size if batch_size > 0 else len(base_models)
        self.score_metric = score_metric
        self.show_analysis = show_analysis
        self.random_state = check_random_state(random_state)
        self.n_evaluations = self.n_iterations * self.n_base_models
        self.single_best_fallback = single_best_fallback
        self._fallback_to_single_best = False
        self.weight_vector_ensemble = weight_vector_ensemble
        self.buffer = [[], []]
        self.supports_mp = supports_mp
        self.start_weight_vector_method = (
            start_weight_vector_method if start_weight_vector_method else "sb"
        )

        # -- Multi-processing
        if (n_jobs == 1) or (os.name == "nt") or (self.batch_size == 1) or (not self.supports_mp):
            self._use_mp = False
            self._n_jobs = 1
            if (os.name == "nt") and (n_jobs != 1):
                warnings.warn(
                    "n_jobs != 1 is not supported on Windows. Setting n_jobs=1.",
                    UserWarning,
                )
            if self.batch_size == 1:
                warnings.warn(
                    "batch size is 1 and thus we can not use multiple jobs. Setting n_jobs=1.",
                    UserWarning,
                )
            if not self.supports_mp:
                warnings.warn(
                    "This method does not support multiprocessing. Setting n_jobs=1.",
                    UserWarning,
                )
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

        self.normalize_weights = normalize_weights != "no"
        self.trim_weights = trim_weights
        self._normalize_weights = partial(
            self._normalize_weights_static,
            method=normalize_weights,
            trim=trim_weights,
            n_iterations=self.n_iterations,
        )

        if self.score_metric.requires_confidences and (not self.normalize_weights):
            self.normalize_predict_proba_ = True

    # --- Fit
    def ensemble_fit(
        self,
        predictions: list[np.ndarray],
        labels: np.ndarray,
    ) -> AbstractWeightedEnsemble:
        # -- Input Validation
        if not isinstance(self.score_metric, AbstractMetric):
            raise ValueError(
                "The provided metric must be an instance of a AbstractMetric, "
                f"nevertheless it is {self.score_metric}({type(self.score_metric)})",
            )

        if not self.single_best_fallback:
            self._fit(predictions, labels)

        else:
            self._single_best_fallback_fit(predictions, labels)

        self._determine_best()

        if self.show_analysis:
            self.plot_opt_history()

        return self

    def _find_x0(self, predictions, labels):
        # -- Get initial set of solutions to check x0
        base_weight_vector = np.zeros((self.n_base_models,))
        initial_solutions = []

        for i in range(self.n_base_models):
            tmp_w = base_weight_vector.copy()
            tmp_w[i] = 1
            initial_solutions.append(tmp_w)

        # -- Find sb
        losses = self._evaluate_batch_of_solutions(np.array(initial_solutions), predictions, labels)
        best_score = np.min(losses)
        sb = initial_solutions[np.argmin(losses)]

        # -- Determine x0
        if self.start_weight_vector_method == "sb":
            _start_weight_vector = sb
        elif self.start_weight_vector_method == "average_ensemble":
            _start_weight_vector = np.full_like(base_weight_vector, 1 / len(base_weight_vector))
        else:
            raise ValueError(
                f"Unknown start weight vector method! Got: {self.start_weight_vector_method}",
            )

        # -- Save results for later
        self.single_best_stats = [best_score, sb]
        self.n_init_evaluations = len(initial_solutions)

        return _start_weight_vector, best_score

    def _fit(self, predictions: list[np.ndarray], labels: np.ndarray) -> None:
        # -- Run Optimization
        _start_weight_vector, best_score = self._find_x0(predictions, labels)

        fbest, xbest, val_loss_over_iterations = self._minimize(
            predictions,
            labels,
            _start_weight_vector,
        )

        # -- Set metadata values
        self.val_loss_over_iterations_ = [best_score, *val_loss_over_iterations]
        self.iteration_batch_size_ = self.batch_size

        if self.weight_vector_ensemble:
            self.opt_best_stats = self._build_weight_vector_ensemble(predictions, labels)
        else:
            self.opt_best_stats = [fbest, xbest]

    @abstractmethod
    def _minimize(
        self,
        predictions: list[np.ndarray],
        labels: np.ndarray,
        _start_weight_vector: np.ndarray,
    ) -> tuple[float, np.ndarray, list[float]]:
        pass

    def _determine_best(self):
        # - Make sure the Single Best is picked if it is better
        # This is needed as many numerical optimization methods do not evaluate x0

        sb_score, sb_w = self.single_best_stats
        opt_score, opt_w = self.opt_best_stats

        if (sb_score <= opt_score) or self._fallback_to_single_best:
            best_w = sb_w
            self.validation_loss_ = sb_score
        else:
            best_w = opt_w
            self.validation_loss_ = opt_score

        self.weights_ = self._normalize_weights(best_w) if self.normalize_weights else best_w
        self.weights_raw_ = best_w

        # print("SB, Optimizer Best:", self.single_best_stats[0], self.opt_best_stats[0])
        # print("Final Weights:", self.weights_)

    # -- Anti-Overfitting Tricks
    @staticmethod
    def _normalize_weights_static(
        weight_vector,
        method="softmax",
        trim="ges-like",
        n_iterations=50,
    ):
        """Normalize the weights to sum up to 1 and such that each element is greater or equal to zero."""
        # -- Step one normalize to sum up to 1
        if method == "softmax":
            exp = np.nan_to_num(np.exp(weight_vector))

            # Set 1 to 0
            exp[exp == 1] = 0

            weight_vector = exp / exp.sum()
        else:
            raise ValueError("The provided weight normalization method is not supported.")

        if trim == "ges-like":
            # Following work by Lennart's (under-review) idea for normalization
            trim_val = 0.5 * (1 / n_iterations)
            weight_vector[weight_vector <= trim_val] = 0

            if sum(weight_vector) == 0:
                weight_vector = np.full_like(weight_vector, 1 / len(weight_vector))
            else:
                rounded_counts = np.array([round(x) for x in weight_vector * n_iterations])

                # -- Re-normalize if needed
                # - Too much
                if sum(rounded_counts) > n_iterations:
                    # here, at most sum(rounded_counts) = n_iterations + n_base_models (if each base model rounded up)
                    # to solve this we go from lowest to highest weighted and remove 1
                    #   Slice to ignore the 0s which can not round up and should not be touched
                    order = np.argsort(rounded_counts)[sum(rounded_counts == 0) :]
                    counter = 0
                    while sum(rounded_counts) > n_iterations:
                        rounded_counts[order[counter]] -= 1
                        counter += 1

                # - Not enough
                #   here, we can have a lot of missing (the more getting trimmed initially)
                #   the solution is to equally distribute among non-zero members
                if sum(rounded_counts) < n_iterations:
                    n_diff = n_iterations - sum(rounded_counts)
                    # Only select non-zero members from large to small weight
                    order = np.argsort(-rounded_counts)
                    if sum(rounded_counts == 0) > 0:
                        order = order[: -sum(rounded_counts == 0)]

                    # Need to equally distribute the weight but make sure that important models get more
                    q, r = divmod(n_diff, len(order))

                    # Equally distribute among all models
                    rounded_counts[order] += q  # [order] to ignore zeros

                    # Equally distribute the remainder among the most weighted models
                    if r != 0:
                        rounded_counts[order[:r]] += 1

                weight_vector = rounded_counts / n_iterations
        elif trim != "no":
            raise ValueError("The provided weight trimming method is not supported.")

        return weight_vector

    def _single_best_fallback_fit(self, predictions, labels):
        # -- Single Best fallback code (try to guarantee better the single best using resampling)
        *fit_pred, fit_labels = resample(
            *predictions,
            labels,
            random_state=self.random_state,
            stratify=labels,
        )
        self._fit(fit_pred, fit_labels)

        # Check if final score is better than single best score
        *sel_pred, sel_labels = resample(
            *predictions,
            labels,
            random_state=self.random_state,
            stratify=labels,
        )
        sel_scores = self._evaluate_batch_of_solutions(
            np.array([self.single_best_stats[1], self.opt_best_stats[1]]),
            sel_pred,
            sel_labels,
        )

        print("SB, ES Scores VAL:", self.single_best_stats[0], self.opt_best_stats[0])
        print("SB, ES Scores SEL:", sel_scores)
        print("ES Weights:", self.opt_best_stats[1])

        sel_es_score = sel_scores[1]

        if (
            (self.single_best_stats[0] < sel_es_score)
            or (sel_scores[0] < sel_es_score)
            or (self.single_best_stats[0] < self.opt_best_stats[0])
        ):
            print(
                "Results of search do not generalize to resampled data -> fallback to single best to be sure.",
            )
            self._fallback_to_single_best = True

            # -- Find true single best with original data(this would usually be known in practice and
            #   thus can be skipped)
            self._find_x0(predictions, labels)

    def _build_weight_vector_ensemble(self, predictions, labels):
        # Get Top 50 Weights
        sel_mask = np.argsort(self.buffer[1])[:50]
        b_weights = [self.buffer[0][i] for i in sel_mask]

        # -- Weighted
        # b_scores = [self.buffer[1][i] for i in sel_mask]
        # r = np.exp(-np.array(b_scores))
        # w = r / r.sum()
        # w_ens = (np.array(b_weights) * w[:, np.newaxis]).sum(axis=0)

        # -- UN weighted
        w_ens = np.array(b_weights).mean(axis=0)
        score = self._wrapper_evaluate_single_solution(w_ens, predictions, labels)

        return [score, w_ens]

    # -- Evaluation Procedure
    def _evaluate_batch_of_solutions(
        self,
        solutions: np.ndarray,
        predictions,
        labels,
    ) -> np.ndarray:
        func_args = [predictions, labels, self.score_metric, self.normalize_predict_proba_]
        internal_solutions = solutions

        # -- Normalize if enabled
        if self.normalize_weights:
            internal_solutions = np.array([self._normalize_weights(s) for s in internal_solutions])

        if len(internal_solutions) > 1:
            if self._use_mp:
                func_args.append(internal_solutions)
                sol_i_list = list(range(len(internal_solutions)))

                with ProcessPoolExecutor(
                    self._n_jobs,
                    initializer=_pool_init,
                    initargs=func_args,
                ) as ex:
                    results = ex.map(_init_wrapper_evaluate_single_solution, sol_i_list)
                losses = np.array(list(results))
            else:
                losses = np.apply_along_axis(
                    partial(evaluate_single_solution, *func_args),
                    axis=1,
                    arr=internal_solutions,
                )
        else:
            losses = np.array([evaluate_single_solution(*func_args, internal_solutions[0])])

        # -- Add to buffer
        self.buffer[0].extend(list(solutions))
        self.buffer[1].extend(list(losses))

        return losses

    def _wrapper_evaluate_single_solution(self, weight_vector, predictions, labels):
        return self._evaluate_batch_of_solutions(np.array([weight_vector]), predictions, labels)[0]


# -- Multiprocessing Stuff
def _pool_init(_predictions, _labels, _score_metric, _normalize_predict_proba_, _solutions):
    global p_predictions, p_labels, p_score_metric, p_normalize_predict_proba_, p_solutions

    p_predictions = _predictions
    p_labels = _labels
    p_score_metric = _score_metric
    p_normalize_predict_proba_ = _normalize_predict_proba_
    p_solutions = _solutions


def _init_wrapper_evaluate_single_solution(sol_index):
    return evaluate_single_solution(
        p_predictions,
        p_labels,
        p_score_metric,
        p_normalize_predict_proba_,
        p_solutions[sol_index],
    )


# -- Func Evals
def evaluate_single_solution(
    predictions,
    labels,
    score_metric,
    normalize_predict_proba,
    weight_vector,
):
    y_pred_ensemble = AbstractWeightedEnsemble._ensemble_predict(
        predictions,
        weight_vector,
        normalize_predict_proba=normalize_predict_proba,
    )
    return score_metric(labels, y_pred_ensemble, to_loss=True, checks=False)
