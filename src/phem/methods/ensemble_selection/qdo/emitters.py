from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from enum import Enum

import numpy as np
from ribs.archives._add_status import AddStatus
from ribs.emitters._emitter_base import EmitterBase
from scipy.special import softmax


class DiscreteWeightSpaceEmitter(EmitterBase):
    """An emitter for ensemble selection that searches in the discrete weight space.

    The emitter mutates or uses crossover such that it changes the discrete weight vector (similar to greedy
    ensemble selection).
    As a result of only changing the discrete weight vector in an appropriate way, the weights produce by this emitter
    always sum to 1 and for all weights w_i it holds 0 < w_i <= 1.

    Parameters
    ----------
    archive: ArchiveBase
        pyribs like archive
    n_base_models: int
        Number of base models.
    batch_size: int
        Number of solutions that the emitter shall propose.
    start_weight_vectors: List[Tuple[np.array (n_base_models,), int]]
        A list of start weight vectors and sample sizes to consider for initial mutations/crossover.
    starting_step_size: int, default=1
        Starting step size used to adjust weight vectors.
    elite_selection_method: str in {"deterministic", "combined_dynamic", "tournament"}, default="combined_dynamic"
        Method used to select the next elite(s) for mutation/crossover.
        *   deterministic: Select the best performing elite as next elite
        *   combined_dynamic: Combination of stochastic and deterministic. Initial 50% chance of either, later on
            the % is dynamically updated based on the performance of the exploration vs exploitation.
        *  tournament: Select the best performing N elites from a tournament of size 10.
           For tournament selection, we first randomly sample 10 elites from the archive. Then, we select the elite
           that wins the tournament. The tournament is a simple tournament where each elite is compared to another.
           For crossover, N=2. Otherwise, N=1. If less than 10 elites are in the archive, we use all existing
           and random initial solutions for the tournament.
    crossover: Optional[str] in {"average", "two_point_crossover"}, default="average"
        If not none, str described the method used for crossover. This works as follows: first we select two parents
        using elite_selection_method. Next, do crossover. Finally, we mutate the new weight vector by taking a step in
        the discrete weights space (We mutate depending on mutation_probability_after_crossover).
        The crossover strategies are all in the the discrete weight space and respect the weight constraints.
        *   average: The new weight vector is the average of the two parents.
        *   two_point_crossover: Two points are selected and children are produced by crossing over at selected points.
    crossover_probability: Union[float, Callable], default=0.5
        If crossover is not None, this determines how likely crossover of children is.
        Allowed values are:
            * Floats between 0 < crossover_probability <= 1.
            * Functions that return the crossover portability (float) for the current internal iteration of the emitter.
        Internally, all values are converted to a function that returns the mutation probability.
    crossover_probability_dynamic: bool, default=True
        If True, we update the crossover probability over time dynamically based on the performance of crossover /
        no crossover. For this to work, `crossover_probability` must be a float. To have the performance of both
        choices, 0 < crossover_probability < 1.
    mutation_probability_after_crossover: Union[float, Callable], default=0.5
        If crossover is not None, this determines how likely mutation of children found with crossover is.
        Allowed values are:
            * Floats between 0 <= mutation_probability_after_crossover <= 1.
                If the value is 0, we employ an emergency mutation rate. That is, if crossover did not
                produce any new offspring too many times (n=50), then we increase the mutation rate by 0.1 such that new
                children might be found. This is reset everytime ask() is called.
                If the value is not 0, too many rejections are handled by the emergency step size.
            * Functions that return the mutation portability (float) for the current internal iteration of the emitter.
        Internally, all values are converted to a function that returns the mutation probability.
    mutation_probability_after_crossover_dynamic: bool, default=True
        If True, we update the probability over time dynamically based on the performance of mutation /
        no mutation. For this to work, `mutation_probability_after_crossover` must be a float. To have the performance
        of both choices, 0 < mutation_probability_after_crossover < 1.
    shared_seen_resources: Tuple(set,dict), default=None
        If not None, we assume this to be shared objects across multiple simple step emitters to avoid proposing
        the same solutions.  resources
    dynamic_updates_consider_rejections: bool, default=False
        If true, the dynamic updates for combined_dynamic, mutation_probability_after_crossover_dynamic, or
        crossover_probability_dynamic take the number of rejections into account.
    bounds: default=None
        Bounds of the solution space. Emitter should respect these. None means no bounds.
        TODO support this parameter. By default the DiscreteWeightSpaceEmitter is bounded in [0,1]
    seed:
        Random seed use by emitter.
        TODO: add documentation for seed, currently assumes that integer seed is given
    """

    allowed_elite_selection_methods = {"deterministic", "tournament", "combined_dynamic"}

    def __init__(
        self,
        archive,
        n_base_models: int,
        start_weight_vectors: list[tuple[np.array, int]],
        batch_size: int,
        starting_step_size: int = 1,
        elite_selection_method: str = "combined_dynamic",
        crossover: str | None = "average",
        crossover_probability: float | Callable = 0.5,
        crossover_probability_dynamic: bool = True,
        mutation_probability_after_crossover: float | Callable = 0.5,
        mutation_probability_after_crossover_dynamic: bool = True,
        negative_steps: bool = False,
        shared_seen_resources: tuple[set, dict] | None = None,
        weight_random_elite_selection: bool = False,
        weight_random_step_selection: bool = False,
        dynamic_updates_consider_rejections: bool = False,
        bounds=None,
        seed=None,
    ):
        # Input Validation (can not do at "fit" time like in sklearn)
        if starting_step_size < 1:
            raise ValueError("starting_step_size must be larger than 1!")
        if elite_selection_method not in self.allowed_elite_selection_methods:
            raise ValueError(
                f"Unknown elite_selection_method! Allowed are: {self.allowed_elite_selection_methods}. Got: {elite_selection_method}."
            )

        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self.n_base_models = n_base_models
        self.starting_step_size = starting_step_size
        self._emergency_step_size = 0
        self.crossover_reject_counter = 0
        self._emergency_mutation_rate = 0
        self._elite_selection_method = elite_selection_method
        self.negative_steps = negative_steps
        self.crossover = crossover

        # -- Handle mutation and crossover probabilities input
        if (not isinstance(mutation_probability_after_crossover, float)) and (
            not isinstance(mutation_probability_after_crossover, Callable)
        ):
            raise ValueError(
                "mutation_probability_after_crossover must be float or  Callable",
                f"Got: {type(mutation_probability_after_crossover)}",
            )
        if isinstance(mutation_probability_after_crossover, float) and (
            (mutation_probability_after_crossover < 0) or (mutation_probability_after_crossover > 1)
        ):
            raise ValueError(
                "for mutation_probability_after_crossover it must hold that hold that 0 <= mutation_probability_after_crossover <=1",
                f"Got: {mutation_probability_after_crossover}",
            )
        if isinstance(mutation_probability_after_crossover, float):
            self._mutation_probability_after_crossover = (
                lambda x: mutation_probability_after_crossover
            )
        else:
            self._mutation_probability_after_crossover = mutation_probability_after_crossover

        if (not isinstance(crossover_probability, float)) and (
            not isinstance(crossover_probability, Callable)
        ):
            raise ValueError(
                "crossover_probability must be float or  Callable",
                f"Got: {type(crossover_probability)}",
            )
        if isinstance(crossover_probability, float) and (
            (crossover_probability <= 0) or (crossover_probability > 1)
        ):
            raise ValueError(
                "for crossover_probability it must hold that hold that 0 < crossover_probability <=1",
                f"Got: {crossover_probability}",
            )
        if isinstance(crossover_probability, float):
            self._crossover_probability = lambda x: crossover_probability
        else:
            self._crossover_probability = crossover_probability

        if crossover_probability_dynamic and (not isinstance(crossover_probability, float)):
            raise ValueError(
                "crossover_probability_dynamic can only be used if crossover_probability is a float!"
            )
        self._crossover_probability_dynamic = crossover_probability_dynamic

        if mutation_probability_after_crossover_dynamic and (
            not isinstance(mutation_probability_after_crossover, float)
        ):
            raise ValueError(
                "_mutation_probability_after_crossover_dynamic can only be "
                + "used if mutation_probability_after_crossover is a float!"
            )
        self._mutation_probability_after_crossover_dynamic = (
            mutation_probability_after_crossover_dynamic
        )

        # Init super class
        EmitterBase.__init__(
            self,
            archive,
            n_base_models,
            bounds,
        )

        # -- Get Start weight vectors (all but one base model have weight zero for all base models)
        self._start_weight_vectors = start_weight_vectors
        self._original_start_weight_vectors = self._start_weight_vectors[:]  # copy

        # -- Set shared resources if needed
        if shared_seen_resources is not None:
            self._seen_percentages = shared_seen_resources[0]
            self._remaining_steps_for_hash = shared_seen_resources[1]
            for w, _ in self._start_weight_vectors:
                self._seen_percentages.add(hash(tuple(w)))
        else:
            self._seen_percentages = {hash(tuple(w)) for w, _ in self._start_weight_vectors}
            self._remaining_steps_for_hash = {}

        # -- Default vars for explore/exploit trade-off (only used if elite_selection_method=dynamic)
        self.explore = 0
        self.exploit = 0
        self.n_mutate = 0
        self.n_crossover = 0
        self._total_rejects = 0
        self._total_crossover_rejects = 0
        self._tmp_crossover_mutation_reject = 0
        self._crossover_mutation_reject = 0
        self._tmp_no_crossover_mutation_reject = 0
        self._no_crossover_mutation_reject = 0
        self._elite_origin_reject_counter = {k: 0 for k in EliteOrigins}
        self._tmp_elite_origin_reject_counter = None

        # -- Weighted Randomness (not used)
        self._weight_random_elite_selection = weight_random_elite_selection
        self._weight_random_step_selection = (
            weight_random_step_selection  # weight_random_step_selection
        )
        if self._weight_random_step_selection:
            np.seterr(invalid="ignore")  # since we abuse dividing by zero to obtain nan values.
            self._base_model_sample_weights = np.full(
                n_base_models, 1 / n_base_models
            )  # step direction weights

        # -- Analysis Stats
        self._n_ensemble_sizes = [0] * (n_base_models + 1)
        self.add_status_counts = [0, 0, 0]  # Not Added, improved, new

        # Counts how often ask() was called (while tell() is also part of an iteration, we ignore it here)
        self._internal_iterations_counter = 0
        self._mutation_probabilities_after_crossover_over_time = []
        self._crossover_probabilities_over_time = []
        self._random_elite_selection_probabilities_over_time = []

        # -- Stuff for Dynamic Updates
        self._origin_performance_template = {
            SolutionOrigins.no_crossover: [
                0,
                0,
                0,
            ],  # Objective value, improvement count, total count
            SolutionOrigins.crossover: [0, 0, 0],
            EliteOrigins.deterministic: [0, 0, 0],
            EliteOrigins.stochastic: [0, 0, 0],
            SolutionOrigins.co_mutation: [0, 0, 0],
            SolutionOrigins.co_no_mutation: [0, 0, 0],
        }
        self._random_elite_selection_probability = 0.5
        self._origin_performance_over_time = []
        self._rejections_over_time = []
        self.dynamic_updates_consider_rejections = dynamic_updates_consider_rejections

    @property
    def mutation_probability_after_crossover(self):
        return self._mutation_probability_after_crossover(self._internal_iterations_counter)

    @property
    def crossover_probability(self):
        return self._crossover_probability(self._internal_iterations_counter)

    @property
    def proposes_negative_weights(self):
        return self.negative_steps

    # -- Ask / Tell code
    def ask(self):
        """Return potential solutions to evaluate.

        Proposes self._batch_size many solutions using the methods in following order:
            1. First it proposes the start weight vectors (created during initialization) as long as they still exist
            2. Next use the elite_selection_method to find a next elite to mutate.

        Added tricks:
            * Rejections: Previous proposed solutions are not proposed again if they are created as a result of a
                mutation.
            * Emergency Step Size: Adapt the step size if too many rejects happened during a single call of ask().
                The emergency step size is used during the mutation to reach regions of the weight space that might be
                further away from the current position. Thus, we might stop proposing already proposed vectors.

        """
        # Solutions to return
        w_vecs_to_return = []

        # Handle too many rejections
        max_number_rejects = 50
        reject_counter = 0

        # Analysis values
        self._tmp_crossover_rejects = 0
        self._tmp_crossover_mutation_reject = 0
        self._tmp_no_crossover_mutation_reject = 0
        self._tmp_elite_origin_reject_counter = {k: 0 for k in EliteOrigins}

        # Emergency values
        self.crossover_reject_counter = 0  # used for emergency mutation rate increase
        self._emergency_step_size = 0
        self._emergency_mutation_rate = (
            0  # Only use when cross is not None and mutation_probability_after_crossover is 0
        )
        use_crossover = self.crossover is not None

        # A step counter needed to discretize weight vectors correctly
        number_of_steps = []
        # An origin counter needed for analysis and dynamic updates
        solution_origins = []
        elite_origins = []

        # Fill solutions to return
        while len(w_vecs_to_return) < self._batch_size:
            # Reject fail save, change step size to escape problematic reject region
            if reject_counter >= max_number_rejects:
                reject_counter = 0
                self._emergency_step_size += 1

            if self.crossover_reject_counter >= max_number_rejects:
                # Unable to produce new children using crossover often enough, increase mutation rate
                self.crossover_reject_counter = 0
                self._emergency_mutation_rate += 1

            # Fill with start weight vectors first
            if self._start_weight_vectors:
                # -- No mutation or selection needed, we just want to evaluate all base models once.
                elite_float_w, sample_size = self._start_weight_vectors.pop()

                w_vecs_to_return.append(elite_float_w)

                # -- Add ask/tell state
                number_of_steps.append(sample_size)
                solution_origins.append(SolutionOrigins.initialization)
                elite_origins.append(EliteOrigins.initialization)
            else:
                mutated_children = self._get_mutated_children(use_crossover)

                # Loop over mutated children and add if needed
                for (
                    proposed_float_weights,
                    new_sample_size,
                    reject_flag,
                ), sol_org, e_org in mutated_children:
                    # Handle reject
                    if reject_flag:
                        # No need to continue as this is the end of the loop anyway
                        reject_counter += 1
                    # Handle insert into batch
                    elif len(w_vecs_to_return) < self._batch_size:
                        # Need to check if list is not too large
                        w_vecs_to_return.append(proposed_float_weights)

                        # -- Add ask/tell state
                        number_of_steps.append(new_sample_size)
                        solution_origins.append(sol_org)
                        elite_origins.append(e_org)

                    # Handle edge case
                    else:
                        # We do not add to the list anymore. Otherwise, we would create a too long list
                        # - Only a problem if crossover returns 2 solutions.

                        # If list got filled in the previous iteration of this loop, we need to correctly
                        # set what we have seen. Because now we might have seen stuff that never gets proposed.
                        pfw_hash = hash(tuple(proposed_float_weights))
                        self._seen_percentages.remove(pfw_hash)

        # -- Collect Meta Info
        self._total_rejects += reject_counter
        self._total_crossover_rejects += self._tmp_crossover_rejects
        self._crossover_mutation_reject += self._tmp_crossover_mutation_reject
        self._no_crossover_mutation_reject += self._tmp_no_crossover_mutation_reject
        self._elite_origin_reject_counter = {
            k: self._tmp_elite_origin_reject_counter[k] + self._elite_origin_reject_counter[k]
            for k in self._elite_origin_reject_counter
        }

        # -- Important return values to keep state inbetween ask/tell
        self.last_ask_sample_sizes = (
            number_of_steps  # Required to correctly transform continuous to discrete weights
        )
        self.last_solutions_origins = solution_origins
        self.last_elite_origins = elite_origins

        # -- Get some analysis stats
        for sol in w_vecs_to_return:
            self._n_ensemble_sizes[np.sum(sol > 0)] += 1

        self._mutation_probabilities_after_crossover_over_time.append(
            self.mutation_probability_after_crossover
        )
        self._crossover_probabilities_over_time.append(self.crossover_probability)
        self._random_elite_selection_probabilities_over_time.append(
            self._random_elite_selection_probability
        )
        self._internal_iterations_counter += 1

        return np.array(w_vecs_to_return)

    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        """Gather results from tell.

        Here, an important different is that we add the sample size to the archive as metadata such that we can
        correctly discretize weight vectors later. The self.last_ask_sample_sizes was set during the last call to ask().
        """
        if (metadata is not None) and (
            not np.array_equal(metadata, np.empty(len(solutions), dtype=object))
        ):
            raise ValueError("This emitter does not (yet) store external metadata.")

        loop_data = zip(
            solutions,
            objective_values,
            behavior_values,
            self.last_ask_sample_sizes,
            self.last_solutions_origins,
            self.last_elite_origins,
            strict=False,
        )

        tmp_origin_performance = copy.deepcopy(self._origin_performance_template.copy())

        for _i, (sol, obj, beh, meta, sol_origin, e_origin) in enumerate(loop_data):
            status, value = self.archive.add(sol, obj, beh, meta)

            # Store status for analysis
            self.add_status_counts[status] += 1

            # -- Collect data about Origin Performance
            if sol_origin != SolutionOrigins.initialization:
                tmp_origin_performance[sol_origin][0] += obj
                if status == AddStatus.IMPROVE_EXISTING:
                    tmp_origin_performance[sol_origin][1] += 1
                tmp_origin_performance[sol_origin][2] += 1

            if self._elite_selection_method == "combined_dynamic":
                if e_origin in [EliteOrigins.deterministic, EliteOrigins.stochastic]:
                    tmp_origin_performance[e_origin][0] += obj
                    if status == AddStatus.IMPROVE_EXISTING:
                        tmp_origin_performance[e_origin][1] += 1
                    tmp_origin_performance[e_origin][2] += 1

        # -- Update Origin summary where needed
        tmp_origin_performance[SolutionOrigins.crossover] = list(
            np.array(tmp_origin_performance[SolutionOrigins.co_mutation])
            + np.array(tmp_origin_performance[SolutionOrigins.co_no_mutation])
        )
        self._origin_performance_over_time.append(tmp_origin_performance)
        self._rejections_over_time.append(
            (
                self._tmp_crossover_rejects,
                self._tmp_crossover_mutation_reject,
                self._tmp_no_crossover_mutation_reject,
                self._tmp_elite_origin_reject_counter,
            )
        )

        # -- Only keep origin performance of last N iterations
        self._origin_performance = copy.deepcopy(self._origin_performance_template)
        self._rejections_counts = (0, 0, 0, {k: 0 for k in EliteOrigins})
        N = -10  # last 10 iterations
        for o_p in self._origin_performance_over_time[N:]:
            for k in self._origin_performance:
                self._origin_performance[k] = list(
                    np.array(self._origin_performance[k]) + np.array(o_p[k])
                )

        # Accumulate rejections
        for rejcs in self._rejections_over_time[N:]:
            tmp_val = list(np.array(self._rejections_counts[:-1]) + np.array(rejcs[:-1]))
            tmp_val_2 = {
                k: rejcs[-1][k] + self._rejections_counts[-1][k]
                for k in self._rejections_counts[-1]
            }
            tmp_val.append(tmp_val_2)
            self._rejections_counts = tmp_val

        # -- Update functions
        self._update_crossover_probability()
        self._update_mutation_probability_after_crossover()
        self._update_random_elite_selection_probability()
        self._update_base_model_sample_weights(solutions, objective_values)

    # -- Dynamic Update Functions
    def _update_crossover_probability(self):
        # "> 2" to guarantee that we are after the initial solution
        if (
            self._crossover_probability_dynamic
            and (self._origin_performance[SolutionOrigins.crossover][2] > 2)
            and (self._origin_performance[SolutionOrigins.no_crossover][2] > 2)
        ):
            # Consider reject rate in performance assessment
            if self.dynamic_updates_consider_rejections:
                reject_ratio_crossover = self._origin_performance[SolutionOrigins.crossover][-1] / (
                    self._origin_performance[SolutionOrigins.crossover][-1]
                    + self._rejections_counts[0]
                    + self._rejections_counts[1]
                )
                reject_ratio_no_crossover = self._origin_performance[SolutionOrigins.no_crossover][
                    -1
                ] / (
                    self._origin_performance[SolutionOrigins.no_crossover][-1]
                    + self._rejections_counts[2]
                )

                if self._origin_performance[SolutionOrigins.crossover][0] < 0:
                    reject_ratio_crossover = 1 + (1 - reject_ratio_crossover)

                if self._origin_performance[SolutionOrigins.no_crossover][0] < 0:
                    reject_ratio_no_crossover = 1 + (1 - reject_ratio_no_crossover)

                self._origin_performance[SolutionOrigins.crossover][0] *= reject_ratio_crossover
                self._origin_performance[SolutionOrigins.no_crossover][0] *= (
                    reject_ratio_no_crossover
                )

            # Update
            cr_prb = _avg_obj_update(
                self._origin_performance[SolutionOrigins.crossover],
                self._origin_performance[SolutionOrigins.no_crossover],
            )

            self._crossover_probability = lambda x: max(min(cr_prb, 0.9), 0.1)

    def _update_mutation_probability_after_crossover(self):
        if (
            self._mutation_probability_after_crossover_dynamic
            and (self._origin_performance[SolutionOrigins.co_mutation][2] > 2)
            and (self._origin_performance[SolutionOrigins.co_no_mutation][2] > 2)
        ):
            # Consider reject rate in performance assessment
            if self.dynamic_updates_consider_rejections:
                reject_ratio_co_mutation = self._origin_performance[SolutionOrigins.co_mutation][
                    -1
                ] / (
                    self._origin_performance[SolutionOrigins.co_mutation][-1]
                    + self._rejections_counts[1]
                )
                reject_ratio_co_no_mutation = self._origin_performance[
                    SolutionOrigins.co_no_mutation
                ][-1] / (
                    self._origin_performance[SolutionOrigins.co_no_mutation][-1]
                    + self._rejections_counts[0]
                )

                if self._origin_performance[SolutionOrigins.co_mutation][0] < 0:
                    reject_ratio_co_mutation = 1 + (1 - reject_ratio_co_mutation)

                if self._origin_performance[SolutionOrigins.co_no_mutation][0] < 0:
                    reject_ratio_co_no_mutation = 1 + (1 - reject_ratio_co_no_mutation)

                self._origin_performance[SolutionOrigins.co_mutation][0] *= reject_ratio_co_mutation
                self._origin_performance[SolutionOrigins.co_no_mutation][0] *= (
                    reject_ratio_co_no_mutation
                )

            cr_prb = _avg_obj_update(
                self._origin_performance[SolutionOrigins.co_mutation],
                self._origin_performance[SolutionOrigins.co_no_mutation],
            )

            self._mutation_probability_after_crossover = lambda x: max(min(cr_prb, 0.9), 0.1)

    def _update_random_elite_selection_probability(self):
        if (
            (self._elite_selection_method == "combined_dynamic")
            and (self._origin_performance[EliteOrigins.stochastic][2] > 2)
            and (self._origin_performance[EliteOrigins.deterministic][2] > 2)
        ):
            # Consider reject rate in performance assessment
            if self.dynamic_updates_consider_rejections:
                reject_ratio_stochastic = self._origin_performance[EliteOrigins.stochastic][-1] / (
                    self._origin_performance[EliteOrigins.stochastic][-1]
                    + self._rejections_counts[-1][EliteOrigins.stochastic]
                )

                reject_ratio_deterministic = self._origin_performance[EliteOrigins.deterministic][
                    -1
                ] / (
                    self._origin_performance[EliteOrigins.deterministic][-1]
                    + self._rejections_counts[-1][EliteOrigins.deterministic]
                )

                if self._origin_performance[EliteOrigins.stochastic][0] < 0:
                    reject_ratio_stochastic = 1 + (1 - reject_ratio_stochastic)

                if self._origin_performance[EliteOrigins.deterministic][0] < 0:
                    reject_ratio_deterministic = 1 + (1 - reject_ratio_deterministic)

                self._origin_performance[EliteOrigins.stochastic][0] *= reject_ratio_stochastic
                self._origin_performance[EliteOrigins.deterministic][0] *= (
                    reject_ratio_deterministic
                )

            rng_prb = _avg_obj_update(
                self._origin_performance[EliteOrigins.stochastic],
                self._origin_performance[EliteOrigins.deterministic],
            )

            self._random_elite_selection_probability = max(min(rng_prb, 0.95), 0.05)

    def _update_base_model_sample_weights(self, solutions, objective_values):
        if self._weight_random_step_selection:
            raise NotImplementedError

            baseline_performance = (
                np.max(objective_values) if self.archive.empty else self.archive.stats.obj_max
            )

            # Compute the contribution to the performance for each base model; scale it by how much it contributed
            avg_perf_per_bm = np.divide(
                (solutions * objective_values.reshape(-1, 1)).sum(axis=0), np.sum(solutions, axis=0)
            )
            # Read as: The base model's contribution was X times worse than the baseline on average
            rel_dist_to_baseline = abs(avg_perf_per_bm - baseline_performance) / abs(
                baseline_performance
            )

            # FIXME: fix this to be something smarter
            # 1 is a constant factor that is subtracted from the weights; smaller distances to baseline makes
            #   the subtracted value smaller
            update_per_weight = 1 * np.nan_to_num(rel_dist_to_baseline, nan=1)
            # Update and re-normalize
            self._base_model_sample_weights = softmax(
                self._base_model_sample_weights - update_per_weight
            )

    # -- Wrapper Function for Selection, Crossover, and mutation
    def _get_mutated_children(self, use_crossover):
        # -- Set where to sample from
        # Since rng random produces [0, 1), "<" is correct such that for 1 it is True and for 0% it is false
        do_crossover = use_crossover and (self._rng.random() < self.crossover_probability)

        # -- Check if crossover is used
        if do_crossover:
            parents = self.get_next_elites(n_elites=2)

            # Handle origins (if two different origins, use combined origin, which can only be stoch+deter)
            parent_origins = [o for _, o in parents]
            if (EliteOrigins.stochastic in parent_origins) and (
                EliteOrigins.deterministic in parent_origins
            ):
                elite_origin = EliteOrigins.stochastic_and_deterministic
            else:
                elite_origin = parent_origins[0]
            sol_origin = None

            parent_elites = [e for e, _ in parents]
            children = self._crossover_elites(parent_elites[0], parent_elites[1])
            tmp_mutated_children = []
            tmp_elite_origins = []

            for c_elite_float_w, c_sample_size in children:
                # -- Decide if mutation is used
                do_mutation_after_crossover = self._rng.random() < (
                    self.mutation_probability_after_crossover + self._emergency_mutation_rate
                )

                # -- Do mutation or not
                if do_mutation_after_crossover:
                    tmp_mutated_children.append(self._mutate_elite(c_elite_float_w, c_sample_size))
                    tmp_elite_origins.append(elite_origin)
                    sol_origin = SolutionOrigins.co_mutation

                    # -- Track numbers
                    int_reject = int(tmp_mutated_children[-1][-1])
                    self._tmp_crossover_mutation_reject += int_reject
                    self._tmp_elite_origin_reject_counter[elite_origin] += int_reject

                else:
                    # -- Rejection Handling for crossover
                    pfw_hash = hash(tuple(c_elite_float_w))
                    if pfw_hash in self._seen_percentages:
                        self.crossover_reject_counter += 1
                        self._tmp_crossover_rejects += 1
                        self._tmp_elite_origin_reject_counter[elite_origin] += 1
                    else:
                        self._seen_percentages.add(pfw_hash)
                        tmp_mutated_children.append((c_elite_float_w, c_sample_size, False))
                        tmp_elite_origins.append(elite_origin)
                        sol_origin = SolutionOrigins.co_no_mutation

        # -- Default to mutation otherwise
        else:
            (elite_float_w, sample_size), elite_origin = self.get_next_elites()[0]
            tmp_mutated_children = [self._mutate_elite(elite_float_w, sample_size)]
            tmp_elite_origins = [elite_origin]
            sol_origin = SolutionOrigins.no_crossover

            # -- Track numbers
            int_reject = int(tmp_mutated_children[-1][-1])
            self._no_crossover_mutation_reject += int_reject
            self._tmp_elite_origin_reject_counter[elite_origin] += int_reject

        # -- Add flag (extra iteration most simple implementation)
        mutated_children = []
        for m_c, elite_origin in zip(tmp_mutated_children, tmp_elite_origins, strict=False):
            mutated_children.append((m_c, sol_origin, elite_origin))

        return mutated_children

    # -- Selection + Exploration/Exploitation Code
    def _get_random_elite(self):
        if self._weight_random_elite_selection:
            # Performance-weighted Random
            #   First implementation.
            elites = list(self.archive)
            if len(elites) == 1:
                elite_float_w, _, _, _, sample_size = elites[0]
            else:
                # Get proba_weights based on performances
                performances = np.array([e.obj for e in self.archive])
                sel_idx = self._rng.choice(
                    len(elites), p=np.exp(performances) / np.exp(performances).sum()
                )
                elite_float_w, _, _, _, sample_size = elites[sel_idx]
        else:
            # Uniform Random
            elite_float_w, _, _, _, sample_size = self.archive.get_random_elite()
        return elite_float_w, sample_size

    def _get_deterministic_elite(self, n=1):
        # Take the best elite
        performances = [e.obj for e in self.archive]
        elites = list(self.archive)

        if n == 1:
            i_best = np.argmax(performances)
            elite_float_w = elites[i_best].sol
            sample_size = elites[i_best].meta
            return [(elite_float_w, sample_size)]
        else:
            return [
                (elites[i_best].sol, elites[i_best].meta)
                for i_best in np.argpartition(performances, -n)[-n:]
            ]

    def _edge_case_select_next_elite(self, n_elites=1):
        # Randomly select something from the original start weight vector for mutation
        #   Choice workaround due to deprecated ndarray creation inside of choice.

        # -- Edge case where we need to create new weight vectors
        if n_elites > len(self._original_start_weight_vectors):
            # Get all existing
            elites = self._original_start_weight_vectors[:]  # copy

            # while due to possible rejection
            while len(elites) < n_elites:
                # Get list of possible start weight vector to mutate
                sel_idx = self._rng.choice(
                    len(self._original_start_weight_vectors),
                    min(n_elites - len(elites), len(self._original_start_weight_vectors)),
                    replace=False,
                )
                for ix in sel_idx:
                    # Mutate and add to list
                    *sampled_elite, reject_flag = self._mutate_elite(
                        *self._original_start_weight_vectors[ix]
                    )
                    if not reject_flag:
                        elites.append(sampled_elite)

            # Shuffle to remove order
            elites = [elites[ix] for ix in self._rng.permutation(len(elites))]

        # -- Default case
        else:
            sel_idx = self._rng.choice(
                len(self._original_start_weight_vectors), n_elites, replace=False
            )
            elites = [self._original_start_weight_vectors[ix] for ix in sel_idx]

        return elites

    def _select_next_elite(self, return_origin=False):
        if self._elite_selection_method in ["combined_dynamic"]:
            if self._rng.random() < self._random_elite_selection_probability:
                self.explore += 1
                origin = EliteOrigins.stochastic
                elite_float_w, sample_size = self._get_random_elite()
            else:
                self.exploit += 1
                origin = EliteOrigins.deterministic
                elite_float_w, sample_size = self._get_deterministic_elite()[0]
        else:
            raise ValueError("Unknown elite selection method! Got:", self._elite_selection_method)

        if return_origin:
            return (elite_float_w, sample_size), origin

        return elite_float_w, sample_size

    def _edge_case_selection(self, n_elites) -> list[tuple[np.ndarray, int]]:
        elites = []

        # Handle edge case where this is the first call of ask() and the archive is still empty, but
        # we already proposed all start weight vectors.
        # This happens if batch size > len(self._start_weight_vectors).
        if self.archive.stats.num_elites == 0:
            elites = self._edge_case_select_next_elite(n_elites)
        elif (n_elites > 1) and (self.archive.stats.num_elites < n_elites):
            elites = self._get_deterministic_elite(self.archive.stats.num_elites)
            rng_elites = self._edge_case_select_next_elite(n_elites - self.archive.stats.num_elites)
            elites.extend(rng_elites)

        return elites

    def _tournament_selection(self, n_elites):
        # Following https://en.wikipedia.org/wiki/Tournament_selection
        n_competitors = n_elites * 10
        p = 0.8
        p_elites = [p * ((1 - p) ** i) for i in range(n_competitors)]
        p_elites = np.array(p_elites) / np.sum(p_elites)  # Guarantee sum of probabilities is 1

        # -- Edge case not enough elites for real tournament
        # TODO: handle not enough elites differently
        elites = self._edge_case_selection(n_competitors)

        if not elites:
            # -- Tournament
            all_elites = list(self.archive)
            # Select without replacement
            elites = [
                all_elites[j]
                for j in self._rng.choice(len(all_elites), n_competitors, replace=False)
            ]
            perf_elites = [e.obj for e in elites]
            elites = np.array(
                [(elite_float_w, sample_size) for elite_float_w, _, _, _, sample_size in elites],
                dtype="object",
            )
            elites = elites[np.argsort(-np.array(perf_elites))]  # Sort by performance
            elites = [tuple(x) for x in elites]

        # Transform into specific format
        sel_idx = self._rng.choice(n_competitors, n_elites, p=p_elites, replace=False)
        return [tuple(elites[idx]) for idx in sel_idx]

    def get_next_elites(self, n_elites=1):
        # -- Tournament is an extra case
        if self._elite_selection_method == "tournament":
            elites = self._tournament_selection(n_elites)
            return [(elite, EliteOrigins.tournament) for elite in elites]

        # -- Handle edge case
        elites = self._edge_case_selection(n_elites)
        if elites:
            return [(elite, EliteOrigins.edge_case) for elite in elites]

        # -- Non Edge Case
        # Special case
        if self._elite_selection_method == "deterministic":
            # Edge case to avoid same deterministic elite everytime
            self.exploit += n_elites
            elites = self._get_deterministic_elite(n=n_elites)

            return [(elite, EliteOrigins.deterministic) for elite in elites]

        # Default case
        return [self._select_next_elite(return_origin=True) for _ in range(n_elites)]

    # -- Crossover
    def _crossover_elites(self, p1, p2):
        self.n_crossover += 1

        p1_elite_float_w, p1_sample_size = p1
        p2_elite_float_w, p2_sample_size = p2
        p1_elite_dis_w = _calculate_counts(p1_elite_float_w, p1_sample_size)
        p2_elite_dis_w = _calculate_counts(p2_elite_float_w, p2_sample_size)

        if self.crossover == "average":
            children = self._crossover_average(p1_elite_dis_w, p2_elite_dis_w)
        elif self.crossover == "two_point_crossover":
            children = self._crossover_two_point(p1_elite_dis_w, p2_elite_dis_w)
        else:
            raise ValueError(f"Unknown crossover strategy. Got: {self.crossover}")

        return children

    def _crossover_average(self, p1_elite_dis_w, p2_elite_dis_w):
        # Naive average crossover that returns ints
        proposed_discrete_weights = np.ceil((p1_elite_dis_w + p2_elite_dis_w) / 2)
        new_sample_size = int(sum(proposed_discrete_weights))
        proposed_float_weights = _calculate_weights(proposed_discrete_weights)
        return [(proposed_float_weights, new_sample_size)]

    def _crossover_two_point(self, p1_elite_dis_w, p2_elite_dis_w):
        non_zero_mask = (p1_elite_dis_w != 0) | (p2_elite_dis_w != 0)
        n_non_zero = sum(non_zero_mask)

        # Handle Edge Case
        if n_non_zero < 3:
            # Not enough unique ensemble member for two point, simply return average
            children = self._crossover_average(p1_elite_dis_w, p2_elite_dis_w)

        # Default Two Point
        else:
            # Find crossover points
            co_points = sorted(self._rng.choice(list(range(n_non_zero)), size=2, replace=False))
            org_p1 = np.copy(p1_elite_dis_w)
            org_p2 = np.copy(p2_elite_dis_w)
            child_1_to_fill = np.copy(p1_elite_dis_w[non_zero_mask])
            child_2_to_fill = np.copy(p2_elite_dis_w[non_zero_mask])

            # Get what to fill and fill it
            fill_p1 = p2_elite_dis_w[non_zero_mask][co_points[0] : co_points[1]]
            child_1_to_fill[co_points[0] : co_points[1]] = fill_p1

            fill_p2_1 = p1_elite_dis_w[non_zero_mask][: co_points[0]]
            fill_p2_2 = p1_elite_dis_w[non_zero_mask][co_points[1] :]
            child_2_to_fill[: co_points[0]] = fill_p2_1
            child_2_to_fill[co_points[1] :] = fill_p2_2

            # Create full children
            p1_elite_dis_w[non_zero_mask] = child_1_to_fill
            p2_elite_dis_w[non_zero_mask] = child_2_to_fill

            # Fill children list and make sure to catch bad mutations
            children = []
            if sum(p1_elite_dis_w) != 0:
                children.append((_calculate_weights(p1_elite_dis_w), int(sum(p1_elite_dis_w))))

            if sum(p2_elite_dis_w) != 0:
                children.append((_calculate_weights(p2_elite_dis_w), int(sum(p2_elite_dis_w))))

            # Edge case where nothing happened
            if not children:
                # Fall Back to average
                children = self._crossover_average(org_p1, org_p2)

        return children

    # -- Mutation
    def _mutate_elite(self, elite_float_w, sample_size):
        """Mutate existing elite to create a new elite.

        We mutate existing weight vectors by taking one or multiple steps in the discrete weight vector space.
        The discrete weight vector space consists of the number of times a base model was sampled.
        """
        self.n_mutate += 1

        # -- Current default method: mutate the discrete weight vector
        # Get discrete weight vector
        elite_dis_w = _calculate_counts(elite_float_w, sample_size)
        # Step (i.e., mutate) the weight vector
        proposed_discrete_weights, step_distance = self._step_for_weights(elite_dis_w)
        # Compute new weights
        proposed_float_weights = _calculate_weights(proposed_discrete_weights)

        # Verify new weight vector and reject if necessary, otherwise add to known weights
        pfw_hash = hash(tuple(proposed_float_weights))
        if pfw_hash in self._seen_percentages:
            return None, None, True
        self._seen_percentages.add(pfw_hash)

        # Employ correct step size here to represent samples correct and fix discretization error
        new_sample_size = sample_size + step_distance

        return proposed_float_weights, new_sample_size, False

    # - Step Selection Code
    def _update_step_lookup(self, idx_for_steps, step_size, dwv_hash):
        possible_steps = [(idx, step_size) for idx in idx_for_steps]

        if self.negative_steps:
            possible_steps.extend([(idx, -step_size) for idx in idx_for_steps])

        self._remaining_steps_for_hash[dwv_hash] = possible_steps

    def _init_step_lookup(self, discrete_weight_vector, dwv_hash):
        idx_for_steps = list(range(self.n_base_models))
        # Remove step for same base model if only 1 model seen so far
        if sum(discrete_weight_vector) == 1:
            idx_for_steps.remove(np.argmax(discrete_weight_vector))

        self._update_step_lookup(idx_for_steps, self.starting_step_size, dwv_hash)

    def _refill_step_lookup(self, discrete_weight_vector, dwv_hash, new_step_size):
        idx_for_steps = list(range(self.n_base_models))
        # Remove step for same base model if only one model has all steps so far
        if sum(discrete_weight_vector) == np.max(discrete_weight_vector):
            idx_for_steps.remove(np.argmax(discrete_weight_vector))

        self._update_step_lookup(idx_for_steps, new_step_size, dwv_hash)

    def _step_for_weights(self, discrete_weight_vector):
        """Main Code that does the steps for a given weight vector.

        Returns:
            new discrete weight vector
            step distance (absolute step size), also identical to how many "samples" were taken.
        """
        tmp_dmw = discrete_weight_vector.copy()

        # --- Setup hash and steps
        dwv_hash = hash(tuple(discrete_weight_vector))
        if dwv_hash not in self._remaining_steps_for_hash:
            self._init_step_lookup(discrete_weight_vector, dwv_hash)

        # --- Select a step direction (i.e., which base model's weight to change)
        if self._weight_random_step_selection:
            selected_step = tuple(
                self._rng.choice(
                    self._remaining_steps_for_hash[dwv_hash],
                    p=softmax(
                        [
                            self._base_model_sample_weights[idx]
                            for idx, _ in self._remaining_steps_for_hash[dwv_hash]
                        ]
                    ),
                )
            )
        else:
            selected_step = tuple(self._rng.choice(self._remaining_steps_for_hash[dwv_hash]))
        self._remaining_steps_for_hash[dwv_hash].remove(selected_step)
        step_direction, step_size = selected_step

        # --- Take step and handle emergency step size correctly
        if step_size < 0:
            tmp_step_size = step_size - self._emergency_step_size
        else:
            tmp_step_size = step_size + self._emergency_step_size

        tmp_dmw[step_direction] += tmp_step_size

        # --- Increase step size if finished
        if len(self._remaining_steps_for_hash[dwv_hash]) == 0:
            self._refill_step_lookup(discrete_weight_vector, dwv_hash, abs(tmp_step_size) + 2)

        return tmp_dmw, abs(tmp_step_size)


# -- Utility functions
def _calculate_weights(discrete_weights) -> np.ndarray:
    # abs() to handle negative discrete steps!
    return discrete_weights / np.abs(discrete_weights).sum()


def _calculate_counts(float_weights, sample_size) -> np.ndarray:
    return float_weights * sample_size


def _avg_obj_update(origin_performance_stats_1, origin_performance_stats_2):
    # Get avg obj score
    avg_rng_perf = origin_performance_stats_1[0] / origin_performance_stats_1[2]
    avg_deter_perf = origin_performance_stats_2[0] / origin_performance_stats_2[2]
    eps = sys.float_info.epsilon

    return 1 - (avg_rng_perf + eps) / (avg_rng_perf + avg_deter_perf + eps)


# -- Enums


class SolutionOrigins(Enum):
    initialization = (0,)
    no_crossover = (1,)
    crossover = (2,)
    co_mutation = (3,)
    co_no_mutation = (4,)


class EliteOrigins(Enum):
    initialization = (0,)
    tournament = (1,)
    edge_case = (2,)
    stochastic = (3,)
    deterministic = (4,)
    stochastic_and_deterministic = 5
