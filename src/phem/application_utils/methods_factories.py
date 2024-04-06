"""Factories to call the post hoc ensembling methods (with a given configuration)."""
from __future__ import annotations

def _factory_SingleBest(metric=None, **kwargs):
    # "custom.SingleBest"
    from phem.methods.baselines.single_best import SingleBest

    return {
        "technique": SingleBest,
        "technique_args": {"metric": metric,
                           "predict_method": "predict_proba"},
        "pre_fit_base_models": True
    }

def _factory_es(rng_seed, metric, n_jobs, use_best):
    from numpy.random import RandomState

    from phem.methods.ensemble_selection.greedy_ensemble_selection import EnsembleSelection

    return {
        "technique": EnsembleSelection,
        "technique_args": {"n_iterations": 50,
                           "metric": metric,
                           "n_jobs": n_jobs,
                           "random_state": RandomState(rng_seed),
                           "use_best": use_best},
        "pre_fit_base_models": True,
    }

def _factory_qdo(rng_seed, metric, is_binary, labels, n_jobs, archive_type, behavior_space_choice, max_elites,
                 emitter_initialization_method, starting_step_size, elite_selection_method, crossover_choice,
                 crossover_probability, crossover_probability_dynamic, mutation_probability_after_crossover,
                 mutation_probability_after_crossover_dynamic, negative_steps, weight_random_elite_selection,
                 weight_random_step_selection, buffer_ratio_choice, dynamic_updates_consider_rejections,
                 batch_size):
    from numpy.random import RandomState

    from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection

    # -- Get Parameter
    if behavior_space_choice is None:
        bs = None
    elif behavior_space_choice == "bs_configspace_similarity_and_loss_correlation":
        from phem.methods.ensemble_selection.qdo.behavior_spaces import get_bs_configspace_similarity_and_loss_correlation
        bs = get_bs_configspace_similarity_and_loss_correlation()
    else:
        raise ValueError("Unknown choice for behavior_space!")

    buffer_ratio = buffer_ratio_choice if buffer_ratio_choice is not None else 1.

    # - Emitter Vars
    emitter_vars = {
        "starting_step_size": starting_step_size,
        "elite_selection_method": elite_selection_method,
    }

    if crossover_choice == "no_crossover":
        emitter_vars["crossover"] = None
    else:
        emitter_vars["crossover"] = crossover_choice

        if isinstance(crossover_probability, float):
            emitter_vars["crossover_probability"] = crossover_probability
        else:
            raise ValueError("crossover_probability must be a float!")

        if isinstance(mutation_probability_after_crossover, float):
            emitter_vars["mutation_probability_after_crossover"] = mutation_probability_after_crossover
        else:
            raise ValueError("mutation_probability_after_crossover must be a float!")

        if crossover_probability_dynamic is not None:
            emitter_vars["crossover_probability_dynamic"] = crossover_probability_dynamic
        if mutation_probability_after_crossover_dynamic is not None:
            emitter_vars["mutation_probability_after_crossover_dynamic"] = mutation_probability_after_crossover_dynamic

    if negative_steps is not None:
        emitter_vars["negative_steps"] = negative_steps

    if weight_random_elite_selection is not None:
        emitter_vars["weight_random_elite_selection"] = weight_random_elite_selection

    if weight_random_step_selection is not None:
        emitter_vars["weight_random_step_selection"] = weight_random_step_selection

    if dynamic_updates_consider_rejections is not None:
        emitter_vars["dynamic_updates_consider_rejections"] = dynamic_updates_consider_rejections

    return {
        "technique": QDOEnsembleSelection,
        "technique_args": {
            "n_iterations": 50,
            "batch_size": batch_size,
            "score_metric": metric,
            "max_elites": max_elites,
            "archive_type": archive_type,
            "buffer_ratio": buffer_ratio,
            "behavior_space": bs,
            "emitter_initialization_method": emitter_initialization_method,
            "emitter_vars": emitter_vars,
            "random_state": RandomState(rng_seed),
            "n_jobs": n_jobs,
        },
        "pre_fit_base_models": True,
    }
