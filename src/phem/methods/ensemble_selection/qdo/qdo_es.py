from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from ribs.optimizers import Optimizer
from sklearn.utils import check_random_state

from phem.base_utils.metrics import AbstractMetric
from phem.framework.abstract_weighted_ensemble import AbstractWeightedEnsemble
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace
from phem.methods.ensemble_selection.qdo.custom_archives.custom_sliding_boundaries_archive import (
    SlidingBoundariesArchive,
)
from phem.methods.ensemble_selection.qdo.custom_archives.quality_archive import QualityArchive
from phem.methods.ensemble_selection.qdo.emitters import DiscreteWeightSpaceEmitter


class QDOEnsembleSelection(AbstractWeightedEnsemble):
    """Using QDO to find a good weight vector for post hoc ensembling.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    n_iterations: int
        The number of iterations determines the number of evaluations. By default, number of evaluations
        = n_iterations * n_base_models. The true number of iterations is also determined by the batch size.
        If n_base_models == batch_size, then used n_iterations is the value that is passed. Otherwise, we need to adapt
        the number of iterations.
            FIXME: change stopping criterion based on time or number of evaluations instead of iterations.
    score_metric: AbstractMetric
        The metric used to evaluate the models' performance.
    behavior_space: BehaviorSpace, default=None
        The behavior space used to determine the behavior of a solution. If None, we assume that
        the framework is used for quality search and we supply a dummy behavior space.
    archive_type: str in {"sliding", "quality"}, default="sliding"
        Defines which archive type to use (i.e., how the bins are set up).
        *   "sliding": Sliding Boundaries Archive; remaps bounds of each dimension in set intervals.
            We use this as default as diversity values in ensembling have no guarantee to be uniformly distributed
            in the boundaries of the values. This is usually the reason why one uses Grid or CVT based archives.
        *   "quality": Traditional population based search without diversity. To do so, we use an archive that
            represents our population of good solutions. That is, a collection of top N elites based on objective value.
            Consequently, any behavior values computed by the behavior space are ignored. "quality" can be used as a
            baseline to see if diversity helps at all.
    buffer_ratio: float, default=1.
        Only relevant for sliding-based archives. The size of the buffer relative to the number of seen solutions.
        For example, 1. means we use all seen solutions for remaps; with 0.5, we use a buffer of size 50% of the total
        solutions. It must hold that 0 < buffer_ratio <= 1.
    max_elites: int, default=49
        Defines the archive size (number of bins). Must be a number that we can split evenly by the number of
        behavior dimensions if we use archives (e.g. a square, a cube, ...).
    batch_size: int, default=40
        Defines the batch size for ask/tell loop.
    emitter_method: str in {"DiscreteWeightSpaceEmitter"}, default="DiscreteWeightSpaceEmitter"
        Defined which hard-coded configuration you want to use for the emitter.
        *   DiscreteWeightSpaceEmitter: An emitter working in the discrete weight space.
    emitter_initialization_method: str in
                {"AllL1", "RandomL2Combinations", "L2ofSingleBest"}, default="RandomL2Combinations"
        Defines the initialization method for the first batch of solutions. That is, the set of initial weight
        vectors.
        *   "AllL1": The first batch of solutions are all base models. That is, the weight vector is a one-hot vector.
        *   "RandomL2Combinations": The first batch of solutions contains random combination of base models.
            As a result, it will start with ensembles of size 2, i.e., Layer 2 (L2). The combinations are done
            exhaustively such that we will produce n_base_models/2 many solutions.
        *   "L2ofSingleBest": Extend the single best model to an L2 ensemble for all possible combinations. Produces
            n_base_models-1 many initial solutions.
    emitter_vars: dict, default=None
        Optional variables passed to the emitter. See emitter documentation to find out what you can pass here.
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    n_jobs: int, default=-1
        Cores to use for parallelization. If -1, use all available cores.
        Please be aware that multi-processing introduces a time overhead.
    show_analysis: bool, default=False
        Show fit analysis code.
    """

    allowed_emitter_methods = {"DiscreteWeightSpaceEmitter"}
    allowed_emitter_initialization_methods = {"AllL1", "RandomL2Combinations", "L2ofSingleBest"}

    def __init__(
        self,
        base_models: list[Callable],
        n_iterations: int,
        score_metric: AbstractMetric,
        behavior_space: BehaviorSpace | None = None,
        archive_type: str = "sliding",
        buffer_ratio: float = 1.0,
        max_elites: int = 49,
        emitter_vars: dict | None = None,
        batch_size: int = 40,
        emitter_method: str = "DiscreteWeightSpaceEmitter",
        emitter_initialization_method: str = "RandomL2Combinations",
        random_state: int | np.random.RandomState | None = None,
        show_analysis=False,
        n_jobs: int = -1,
    ) -> None:
        # The following just tells our wrapper/supper that we want to have prediction probabilities as input for fit
        super().__init__(base_models, "predict_proba")

        # -- Iteration management
        self.n_iterations = n_iterations
        self.n_base_models = len(base_models)
        self.batch_size = batch_size
        self.n_evaluations = self.n_iterations * self.n_base_models
        self.internal_n_iterations, self.n_rest_evaluations = self._compute_n_iterations(
            self.n_evaluations,
            self.batch_size,
        )

        # -- Basic Init
        self.archive_type = archive_type
        self.max_elites = max_elites
        self.emitter_method = emitter_method
        self.show_analysis = show_analysis
        self.buffer_ratio = buffer_ratio
        self.random_state = check_random_state(random_state)

        # -- Metrics / Behavior
        self.score_metric = score_metric
        self.behavior_space = behavior_space if behavior_space is not None else BehaviorSpace([])

        # -- More Advanced
        self.emitter_vars = {} if emitter_vars is None else emitter_vars
        self.init_iteration_results = None  # type: Optional[List[float, np.ndarray]]
        self._res_buffer = np.ndarray((0, 1 + len(self.behavior_space.behavior_functions)))
        self.emitter_initialization_method = emitter_initialization_method
        self._n_init_evals = 0

        # -- Pre-process metadata from base models
        self.config_key_to_range_ = None  # type: Optional[Dict[str, float]]
        self.qdo_base_models_metadata_ = None  # type: Optional[List[dict]]
        if self.behavior_space.requires_base_model_metadata:
            self.qdo_base_models_metadata_ = []  # Only consists of the config and no other metadata
            for bm_metadata in self.base_models_metadata:
                if not (bm_metadata.get("auto-sklearn-model")):
                    raise NotImplementedError(
                        "We currently only support base model metadata created by the ",
                        "Auto-sklearn Assembler.",
                    )
                self.qdo_base_models_metadata_.append(bm_metadata["config"])

            # -- Compute ranges
            all_numeric_keys = [
                (k, v)
                for bm in self.base_models_metadata
                for k, v in bm["config"].items()
                if not isinstance(v, str)
            ]
            # Get bounds
            key_to_bounds = {
                k: [float("+inf"), float("-inf")] for k in {k for k, v in all_numeric_keys}
            }
            for k, v in all_numeric_keys:
                curr_min = key_to_bounds[k][0]
                curr_max = key_to_bounds[k][1]
                if v < curr_min:
                    key_to_bounds[k][0] = v
                if v > curr_max:
                    key_to_bounds[k][1] = v

            # Transform to ranges
            self.config_key_to_range_ = {
                k: abs(bds[1] - bds[0]) for k, bds in key_to_bounds.items()
            }

        # -- Multi-processing
        if (n_jobs == 1) or (os.name == "nt"):
            self._use_mp = False
            if os.name == "nt":
                pass
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

    def ensemble_fit(
        self,
        predictions: list[np.ndarray],
        labels: np.ndarray,
    ) -> AbstractWeightedEnsemble:
        # -- Input Validation
        self.n_iterations = int(self.n_iterations)
        if self.n_iterations < 1:
            raise ValueError("Number of evaluations cannot be less than one!")

        if not isinstance(self.score_metric, AbstractMetric):
            raise ValueError(
                "The provided metric must be an instance of a AbstractMetric, "
                f"nevertheless it is {self.score_metric}({type(self.score_metric)})",
            )

        if not isinstance(self.behavior_space, BehaviorSpace):
            raise ValueError(
                "The provided behavior space must be an instance of a BehaviorSpace, "
                f"nevertheless it is {self.behavior_space}({type(self.behavior_space)})",
            )

        if self.emitter_method not in self.allowed_emitter_methods:
            raise ValueError(
                f"The emitter methods is not in {self.allowed_emitter_methods}. Got: {self.emitter_method}",
            )

        if self.emitter_initialization_method not in self.allowed_emitter_initialization_methods:
            raise ValueError(
                f"The emitter initialization method is not in {self.allowed_emitter_initialization_methods}. Got: {self.emitter_initialization_method}",
            )

        if (not isinstance(self.buffer_ratio, float)) or (
            isinstance(self.buffer_ratio, float)
            and ((self.buffer_ratio <= 0) or (self.buffer_ratio > 1))
        ):
            raise ValueError(
                "Buffer ratio must be a float larger zero and smaller or equal to 1.",
                f"Got: {self.buffer_ratio}, {type(self.buffer_ratio)}",
            )

        if (self.archive_type not in ["quality"]) and int(
            round(self.max_elites ** (1.0 / self.behavior_space.n_dims)),
        ) ** self.behavior_space.n_dims != self.max_elites:
            raise ValueError(
                "Number of max elites can not be used to split the dimensions evenly!",
                f"Got: {self.max_elites}, {self.behavior_space.n_dims}.",
            )

        # -- Init Prerequisites
        self._init_optimize(predictions)

        # -- Init QDO Elements
        self._init_archive()
        self._init_emitters(predictions, labels)

        # -- Call optimizer
        self._optimize(predictions, labels)

        # -- Get final weights
        self._compute_weights(predictions, labels)

        if self.show_analysis:
            self._analysis_fit(predictions, labels)

        return self

    def _init_archive(self):
        # -- Build Archive
        if self.archive_type == "sliding":
            archive = SlidingBoundariesArchive(
                dims=[
                    int(round(self.max_elites ** (1.0 / self.behavior_space.n_dims)))
                    for _ in range(self.behavior_space.n_dims)
                ],
                ranges=self.behavior_space.ranges,
                seed=self.random_state.randint(1, 100000),
                remap_frequency=self.n_base_models,
                initial_remap=self.n_base_models + self.batch_size,
                buffer_capacity=int((self.n_evaluations + 10) * self.buffer_ratio),
                show_analysis=self.show_analysis,
            )
        elif self.archive_type == "quality":
            archive = QualityArchive(
                archive_size=self.max_elites,
                behavior_n_dim=self.behavior_space.n_dims,
                seed=self.random_state.randint(1, 100000),
                show_analysis=self.show_analysis,
            )
        else:
            raise ValueError("Unknown Archive Type!")

        self.archive = archive

    def _update_n_iterations(self, n_evals_outside_of_loop: int):
        # Set parameter for logging to see how many values were evaluated outside the loop
        self._n_init_evals += n_evals_outside_of_loop
        # Set number of iterations to reflect this and to guarantee that we do not exceed the allowed
        # number of evaluations
        self.internal_n_iterations, self.n_rest_evaluations = self._compute_n_iterations(
            self.n_evaluations - self._n_init_evals,
            self.batch_size,
        )

    @staticmethod
    def _compute_n_iterations(n_eval, batch_size):
        internal_n_iterations = n_eval // batch_size

        n_rest_evaluations = 0 if n_eval % batch_size == 0 else n_eval % batch_size

        return internal_n_iterations, n_rest_evaluations

    def _init_start_weight_vectors(self, predictions: list[np.ndarray], y_true: np.ndarray):
        """Initialize the weight vectors for the first batch of solutions to be evaluated.

        We must set self.init_iteration_results if the start weight vectors do not include the Single Best.
        """
        # -- Evaluate Base Models and to determine start weight vectors
        base_weight_vector = np.zeros((self.n_base_models,))
        _start_weight_vectors = []

        # -- Get L1 Weight Vectors
        for i in range(self.n_base_models):
            tmp_w = base_weight_vector.copy()
            tmp_w[i] = 1
            _start_weight_vectors.append((tmp_w, 1))

        # -- Switch case for different initialization methods
        if self.emitter_initialization_method != "AllL1":
            # -- Get Single Best to guarantee performance improvement based on validation data
            objs, _ = self._evaluate_batch_of_solutions(
                np.array([w for w, _ in _start_weight_vectors]),
                predictions,
                y_true,
            )
            sb_weight_vector = _start_weight_vectors[np.argmax(objs)]
            self._update_n_iterations(self.n_base_models)

            # -- Switch case
            if self.emitter_initialization_method == "RandomL2Combinations":
                self.init_iteration_results = (np.max(objs), sb_weight_vector[0])

                l2_combinations = self.random_state.choice(
                    self.n_base_models,
                    (self.n_base_models // 2, 2),
                    replace=False,
                )
                # Fill combinations
                _start_weight_vectors = []
                for i, j in l2_combinations:
                    tmp_w = base_weight_vector.copy()
                    tmp_w[i] = 0.5
                    tmp_w[j] = 0.5
                    _start_weight_vectors.append((tmp_w, 2))

            elif self.emitter_initialization_method == "L2ofSingleBest":
                self.init_iteration_results = (np.max(objs), sb_weight_vector[0])
                index_single_best = np.argmax(objs)

                # Fill combinations
                _start_weight_vectors = []
                for i in range(self.n_base_models):
                    if i == index_single_best:
                        continue
                    tmp_w = base_weight_vector.copy()
                    tmp_w[i] = 0.5
                    tmp_w[index_single_best] = 0.5
                    _start_weight_vectors.append((tmp_w, 2))

        # If it is "AllL1" we do need to anything and just return all of them
        return _start_weight_vectors

    def _init_emitters(self, predictions, y_true):
        """Init emitter.

        Step Emitters does most of the initialization internally.

        The following must be set in the init:
        ---------
        self.discrete_merge_ : bool
            True if we need to make weights discrete for merging at the end during
            compute weights. Otherwise, false.
        self.init_iteration_results: Optional[List[float, np.ndarray]]
            Set to the best initial result; needed if the archive does not evaluate
            the initial solutions (e.g. x0). Otherwise, validation score could be
            worse than the single best.
        """
        _start_weight_vectors = self._init_start_weight_vectors(predictions, y_true)

        # Code for analysis
        if self.show_analysis:
            o, b = self._evaluate_batch_of_solutions(
                np.array([w for w, _ in _start_weight_vectors]),
                predictions,
                y_true,
            )
            self._sb_stats = [o[np.argmax(o)], b[np.argmax(o), :]]

        # -- Build emitters
        if self.emitter_method == "DiscreteWeightSpaceEmitter":
            for req_time_atr in ["mutation_probability_after_crossover", "crossover_probability"]:
                if (req_time_atr in self.emitter_vars) and (
                    isinstance(self.emitter_vars[req_time_atr], Callable)
                ):
                    self.emitter_vars[req_time_atr] = partial(
                        self.emitter_vars[req_time_atr],
                        max_time=self.internal_n_iterations,
                    )

            emitters = [
                DiscreteWeightSpaceEmitter(
                    self.archive,
                    self.n_base_models,
                    _start_weight_vectors,
                    batch_size=self.batch_size,
                    seed=self.random_state.randint(1, 100000),
                    **self.emitter_vars,
                ),
            ]
            self.discrete_merge_ = True
        else:
            raise RuntimeError("You should not be here!")

        self.emitters = emitters

        # Set proba normalization if needed
        #   If negative weights are proposed, we need to normalize probabilities
        #   for some metrics. Here, we set a flag to do so later.
        metrics_need_proba = self.score_metric.requires_confidences or (
            "proba" in self.behavior_space.required_prediction_types
        )
        if any(emt.proposes_negative_weights for emt in self.emitters) and metrics_need_proba:
            self.normalize_predict_proba_ = True

    def _init_optimize(self, predictions):
        # Pre-compute values that are regularly needed for evaluation

        self._raw_predictions = None  # type: Optional[np.ndarray]

        if "raw" in self.behavior_space.required_prediction_types:
            self._raw_predictions = [self._confidences_to_predictions(bm) for bm in predictions]

    def _optimize(self, predictions: list[np.ndarray], labels: np.ndarray) -> None:
        # -- Build Optimizer
        opt = Optimizer(self.archive, self.emitters)

        # -- Set up Results collector
        optimize_stats = {
            "Archive Size": [],
            "Max Objective": [],
        }
        if self.init_iteration_results is not None:
            optimize_stats["Archive Size"].append(0)
            optimize_stats["Max Objective"].append(self.init_iteration_results[0])

        for _itr in range(1, self.internal_n_iterations + 1):
            # Get solutions
            sols = opt.ask()

            # Evaluate solutions
            objs, bcs = self._evaluate_batch_of_solutions(sols, predictions, labels)

            # Report back and restart
            opt.tell(objs, bcs)

            # Log stats
            optimize_stats["Archive Size"].append(len(opt.archive))
            optimize_stats["Max Objective"].append(float(opt.archive.stats.obj_max))

        # -- Rest Iteration, required if wanted number of evaluations can not be evenly
        #    distributed across batches.
        if self.n_rest_evaluations:
            sols = opt.ask()
            org_length = len(sols)
            sols = sols[: self.n_rest_evaluations, :]

            objs, bcs = self._evaluate_batch_of_solutions(sols, predictions, labels)

            # Get some existing behavior values and a worse objective value
            dummy_obj = objs[0]
            dummy_bc = bcs[0, :]

            # fill rest of the solutions with dummy values
            objs = np.hstack([objs, [dummy_obj] * (org_length - len(sols))])
            bcs = np.vstack([bcs, [dummy_bc] * (org_length - len(sols))])

            opt.tell(objs, bcs)
            optimize_stats["Archive Size"].append(len(opt.archive))
            optimize_stats["Max Objective"].append(float(opt.archive.stats.obj_max))

        self.optimize_stats_ = optimize_stats

    def _compute_weights(self, predictions, labels):
        """Code to compute the final weight vector (among other things).

        Code does the following:
            1. First it tests if merging all found solutions is better than the single
                best solution. If yes, the merged solution is used.
            2. Second if verifies (if needed) if the best found solution improved
                over the single best base model and returns the single best
                base model's results.
            3. Lastly, it fills metadat about the training process such that we can analysis them later.
        """
        # -- Get Best Weight Vector
        elites = list(self.archive)
        performances = [e.obj for e in elites]

        # - Get merged weights
        if self.discrete_merge_:
            disc_weights = np.array([elite.sol * elite.meta for elite in elites])
            merged_weights = np.sum(disc_weights, axis=0) / np.sum(np.sum(disc_weights, axis=1))
        else:
            merged_weights = np.array([elite.sol for elite in elites]).sum(axis=0) / len(elites)
        # Get performance for merged weights
        merge_obj = self._evaluate_batch_of_solutions(
            np.array([merged_weights]),
            predictions,
            labels,
        )[0][0]

        # max/Argmax because we made this a maximization problem to work with ribs
        if merge_obj > np.max(performances):
            self.optimize_stats_["merged_weights"] = True
            self.weights_ = merged_weights
            self.validation_loss_ = -float(merge_obj)

        else:
            self.optimize_stats_["merged_weights"] = False
            self.weights_ = elites[np.argmax(performances)].sol
            self.validation_loss_ = -float(np.max(performances))

        # -- Verify that the optimization method improved over the single best
        #   Only done if the method does/can not do this by itself
        if self.init_iteration_results is not None:
            init_score, init_weights = self.init_iteration_results

            # >= because ini iteration has most likely a smaller ensemble size
            if self.validation_loss_ >= -float(init_score):
                self.optimize_stats_["merged_weights"] = False
                self.weights_ = init_weights
                self.validation_loss_ = -float(init_score)

        # -- Set to save metadata
        self.iteration_batch_size_ = self.batch_size
        self.val_loss_over_iterations_ = [-i for i in self.optimize_stats_["Max Objective"]]

        # - set additional metadata
        add_var = isinstance(self.emitters[0], DiscreteWeightSpaceEmitter)
        self.model_specific_metadata_ = {
            "evaluation_types": {
                "total": int(self.n_base_models * self.n_iterations),
                "explore": sum(int(em.explore) for em in self.emitters) if add_var else -1,
                "exploit": sum(int(em.exploit) for em in self.emitters) if add_var else -1,
                "init": int(self._n_init_evals) if add_var else -1,
                "rejects": sum(int(em._total_rejects) for em in self.emitters) if add_var else -1,
                "crossover_rejects": sum(int(em._total_crossover_rejects) for em in self.emitters)
                if add_var
                else -1,
                "n_mutate": sum(int(em.n_mutate) for em in self.emitters) if add_var else -1,
                "n_crossover": sum(int(em.n_crossover) for em in self.emitters) if add_var else -1,
            },
            "internal_n_iterations": int(self.internal_n_iterations)
            + int(self.n_rest_evaluations > 0),
            "archive_size": [int(i) for i in self.optimize_stats_["Archive Size"]],
        }

    def _evaluate_batch_of_solutions(self, solutions: np.ndarray, predictions, y_true):
        """Return objective value and BC for a batch of solutions.

        Parameters
        ----------
        solutions: np.ndarray (batch_size, n_base_models)
            A batch of weight vectors.

        Returns:
        -------
            objs (np.ndarray): (batch_size,) array with objective values.
            bcs (np.ndarray): (batch_size,) array with a BC in each row.
        """
        # Determine function arguments
        input_required = self.behavior_space.required_prediction_types
        raw_req = "raw" in input_required
        proba_req = "proba" in input_required
        bm_meta_req = self.behavior_space.requires_base_model_metadata

        # Create static function arguments list
        func_args = [
            predictions,
            y_true,
            self.score_metric,
            self.behavior_space,
            self.normalize_predict_proba_,
            proba_req,
            raw_req,
            self._raw_predictions,
            bm_meta_req,
            self.config_key_to_range_,
            self.qdo_base_models_metadata_,
        ]

        if self._use_mp:
            func_args.append(solutions)
            sol_i_list = list(range(len(solutions)))

            with ProcessPoolExecutor(
                self._n_jobs,
                initializer=_pool_init,
                initargs=func_args,
            ) as ex:
                results = ex.map(_init_wrapper_evaluate_single_solution, sol_i_list)
            res = np.array(list(results))
        else:
            res = np.apply_along_axis(
                partial(evaluate_single_solution, *func_args),
                axis=1,
                arr=solutions,
            )

        if self.show_analysis:
            self._res_buffer = np.vstack([self._res_buffer, res])

        return res[:, 0], res[:, 1:]

    def _analysis_fit(self, predictions, labels, plot=False, plot_animation=False):
        """Some basic stuff to look at the result of fit."""
        raise NotImplementedError()


def _pool_init(
    _predictions,
    _y_true,
    _score_metric,
    _behavior_space,
    _normalize_predict_proba_,
    _proba_req,
    _raw_req,
    _raw_predictions,
    _bm_meta_req,
    _config_key_to_range_,
    _qdo_base_models_metadata_,
    _solutions,
):
    global \
        predictions, \
        y_true, \
        score_metric, \
        behavior_space, \
        normalize_predict_proba_, \
        proba_req, \
        raw_req, \
        raw_predictions, \
        bm_meta_req, \
        config_key_to_range_, \
        qdo_base_models_metadata_, \
        solutions

    predictions = _predictions
    y_true = _y_true
    score_metric = _score_metric
    behavior_space = _behavior_space
    normalize_predict_proba_ = _normalize_predict_proba_
    proba_req = _proba_req
    raw_req = _raw_req
    raw_predictions = _raw_predictions
    bm_meta_req = _bm_meta_req
    config_key_to_range_ = _config_key_to_range_
    qdo_base_models_metadata_ = _qdo_base_models_metadata_
    solutions = _solutions


def _init_wrapper_evaluate_single_solution(sol_index):
    return evaluate_single_solution(
        predictions,
        y_true,
        score_metric,
        behavior_space,
        normalize_predict_proba_,
        proba_req,
        raw_req,
        raw_predictions,
        bm_meta_req,
        config_key_to_range_,
        qdo_base_models_metadata_,
        solutions[sol_index],
    )


def evaluate_single_solution(
    predictions,
    y_true,
    score_metric,
    behavior_space,
    normalize_predict_proba_,
    proba_req,
    raw_req,
    raw_predictions,
    bm_meta_req,
    config_key_to_range_,
    qdo_base_models_metadata_,
    weight_vector,
):
    """Static Out-of-Class Function to avoid pickle problems in multiprocessing.

    Parameters
    ----------
    predictions: (see ensemble_fit)
    y_true: (see ensemble_fit)
    score_metric: AbstractMetric
        Metric used to score the solution
    behavior_space: BehaviorSpace
        Behavior Space used to evaluate the diversity of the solution
    normalize_predict_proba_: bool
        Required for ensemble_predict to know if the proba scores need to be normalized
    proba_req: bool
        Whether the behavior space needs proba predictions
    raw_req: bool
        Whether the behavior space needs raw predictions
    raw_predictions: (see ensemble_fit and init_optimize)
        The pre-computed raw predictions of all base models. Only required if raw_req is True.
    bm_meta_req: bool
        Whether the behavior space needs base mode metadata
    config_key_to_range_: (see class init)
        Base Mode metadata. Only required if bm_meta_req is True.
    qdo_base_models_metadata_: (see class init)
        Base Mode metadata. Only required if bm_meta_req is True.
    weight_vector: array-like, shape (n_base_models,)
        A specific solution vector that is to be evaluated.

    Returns:
    -------
        s_m: Float
            Score of Solution (as loss)
        bcs: array-like, shape (behavior_space.n_dims,)
            The behavior function values of the solution.
    """
    # Get Score
    y_pred_ensemble = AbstractWeightedEnsemble._ensemble_predict(
        predictions,
        weight_vector,
        normalize_predict_proba_,
    )

    # Negative loss because we want to maximize
    s_m = -score_metric(y_true, y_pred_ensemble, to_loss=True, checks=False)

    # Base Model Y_preds
    rm_zero_mask = weight_vector != 0
    Y_pred_base_models = [
        bm for bm, selected in zip(predictions, rm_zero_mask, strict=False) if selected
    ]

    # Handle arguments
    pred_arguments = {}
    if raw_req:
        pred_arguments["raw_preds"] = [
            AbstractWeightedEnsemble._confidences_to_predictions(y_pred_ensemble),
            [bm for bm, selected in zip(raw_predictions, rm_zero_mask, strict=False) if selected],
        ]

    if proba_req:
        pred_arguments["proba_preds"] = [y_pred_ensemble, Y_pred_base_models]

    if bm_meta_req:
        pred_arguments["input_metadata"] = [
            config_key_to_range_,
            [
                md
                for md, selected in zip(qdo_base_models_metadata_, rm_zero_mask, strict=False)
                if selected
            ],
        ]

    b_i = behavior_space(weight_vector, y_true, **pred_arguments) if pred_arguments else []

    return s_m, *b_i
