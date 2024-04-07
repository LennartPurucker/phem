from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import cma
import numpy as np

from phem.framework.abstract_numerical_solvers import NumericalSolverBase

if TYPE_CHECKING:
    from phem.base_utils.metrics import AbstractMetric

import warnings


class CMAES(NumericalSolverBase):
    """Numerical Solver CMA-ES to find a weight vector (ensemble weighting).

    To use CMA-ES-ExplicitGES, set `normalize_weights="softmax"` and `trim_weights="ges-like"`.

    Super Parameters
    ----------
        See NumericalSolverBase for more details.

    --- Method Parameters
    batch_size: str or int, default=25
        The batch size of CMA-ES ("popsize" for CMAES).
    sigma0: float, default=0.2
        Initial step-size for CMA-ES.
    verbose: bool, default=False
        If True, print CMA-ES progress.
    """

    def __init__(
        self,
        base_models: list[Callable],
        score_metric: AbstractMetric,
        n_iterations: int,
        *,
        batch_size: int | str = "dynamic",
        sigma0: float = 0.2,
        verbose: bool = False,
        normalize_weights: Literal["no", "softmax"] = "no",
        trim_weights: Literal["no", "ges-like"] = "no",
        single_best_fallback: bool = False,
        weight_vector_ensemble: bool = False,
        start_weight_vector_method: Literal["average_ensemble", "sb"] = "sb",
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int = -1,
    ) -> None:
        if isinstance(batch_size, int):
            tmp_batch_size = batch_size
        elif batch_size == "dynamic":
            # Following CMA-ES default
            tmp_batch_size = int(4 + 3 * np.log(len(base_models)))
        else:
            raise ValueError(f"Unknown batch size argument! Got: {batch_size}")

        super().__init__(
            base_models=base_models,
            score_metric=score_metric,
            n_iterations=n_iterations,
            batch_size=tmp_batch_size,
            normalize_weights=normalize_weights,
            trim_weights=trim_weights,
            single_best_fallback=single_best_fallback,
            weight_vector_ensemble=weight_vector_ensemble,
            start_weight_vector_method=start_weight_vector_method,
            random_state=random_state,
            n_jobs=n_jobs,
            supports_mp=True,
        )
        self.sigma0 = sigma0
        self.verbose = verbose

    def _compute_internal_iterations(self):
        # -- Determine iteration handling
        n_evals = self.n_evaluations - self.n_init_evaluations
        internal_n_iterations = n_evals // self.batch_size
        n_rest_evaluations = 0 if n_evals % self.batch_size == 0 else n_evals % self.batch_size

        return internal_n_iterations, n_rest_evaluations

    def _minimize(
        self,
        predictions: list[np.ndarray],
        labels: np.ndarray,
        _start_weight_vector: np.ndarray,
    ):
        internal_n_iterations, n_rest_evaluations = self._compute_internal_iterations()
        es = self._setup_cma(_start_weight_vector)
        val_loss_over_iterations = []

        # Iterations
        for _itr in range(1, internal_n_iterations + 1):
            # Ask/tell
            solutions = es.ask()
            es.tell(solutions, self._evaluate_batch_of_solutions(solutions, predictions, labels))
            if self.verbose:
                es.disp(modulo=1)

            # Iteration finalization
            val_loss_over_iterations.append(es.result.fbest)

            # -- ask/tell rest solutions
        if n_rest_evaluations > 0:
            solutions = es.ask(n_rest_evaluations)
            es.best.update(
                solutions,
                arf=self._evaluate_batch_of_solutions(solutions, predictions, labels),
            )
            if self.verbose:
                warnings.warn(
                    f"Evaluated {n_rest_evaluations} rest solutions in a remainder iteration.",
                )
            val_loss_over_iterations.append(es.result.fbest)

        return es.result.fbest, es.result.xbest, val_loss_over_iterations

    def _setup_cma(self, _start_weight_vector) -> cma.CMAEvolutionStrategy:
        # Setup CMA
        opts = cma.CMAOptions()
        opts.set("seed", self.random_state.randint(0, 1000000))
        opts.set("popsize", self.batch_size)
        # opts.set("maxfevals", self.remaining_evaluations_)  # Not used because we control by hand.

        return cma.CMAEvolutionStrategy(_start_weight_vector, self.sigma0, inopts=opts)
