"""Selection of Pre-defined behavior spaces."""

from __future__ import annotations


def get_bs_configspace_similarity_and_loss_correlation():
    # "bs_configspace_similarity_and_loss_correlation"
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import LossCorrelationMeasure
    from phem.methods.ensemble_selection.qdo.behavior_functions.implicit_diversity_metrics import (
        ConfigSpaceGowerSimilarity,
    )
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace

    return BehaviorSpace([ConfigSpaceGowerSimilarity, LossCorrelationMeasure])


def get_bs_ensemble_size_and_loss_correlation():
    # "bs_configspace_similarity_and_loss_correlation"
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import (
        EnsembleSize,
        LossCorrelationMeasure,
    )
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace

    return BehaviorSpace([EnsembleSize, LossCorrelationMeasure])
