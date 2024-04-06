from __future__ import annotations

from functools import partial

from phem.base_utils.diversity_metrics import LossCorrelation
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction


def ensemble_size(weights) -> float:
    return sum(weights != 0)

# -- Make Behavior Functions
# - Diversity Metrics
LossCorrelationMeasure = BehaviorFunction(partial(LossCorrelation, checks=False), ["y_true", "Y_pred_base_models"],
                                          (0, 1), "proba", name=LossCorrelation.name + "(Lower is more Diverse)")
EnsembleSize = BehaviorFunction(ensemble_size, ["weights"], (0, 50), "none", name="Ensemble Size")
