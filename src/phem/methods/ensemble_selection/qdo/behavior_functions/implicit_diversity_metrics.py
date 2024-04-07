"""Implicit Diversity Metrics.

Any metric that tries to implicit measure the diversity of an ensemble.
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean

from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction


def config_space_gower_similarity(input_metadata: tuple[dict[str, int | float], list[dict]]):
    """Gower Distance used to compute the similarity between the identical keys of all pairs of configurations.

    The input metadata consists of one directory contain the ranges for numeric-valued keys of configurations and
        the configurations of the base models in the ensemble.

    The gower distance is transformed into similarity. This allows us the following interpretation:
        1 - No Diversity
        0 - Maximal Diversity
    """
    ranges, base_models_metadata = input_metadata

    # Default Case
    if len(base_models_metadata) == 1:
        return 1

    # Iterate over all pairs
    scores_over_pairs = []
    for bm_md_1, bm_md_2 in combinations(base_models_metadata, 2):
        identical_keys = set(bm_md_1.keys()).intersection(set(bm_md_2.keys()))
        distance_scores = []  # lower is better

        for i_k in identical_keys:
            if isinstance(bm_md_1[i_k], str):
                # -- Assume it is a categorical hyperparameter/choice
                if bm_md_1[i_k] == bm_md_2[i_k]:
                    distance_scores.append(0)  # Identical means absolute similar / close
                else:
                    # Unequal means punishing by maximal distance (no config space interpretation added)
                    distance_scores.append(1)
            elif bm_md_1[i_k] == bm_md_2[i_k]:
                # Catch edge case where the values are identical to avoid potential divide by zero error if
                #   bm_md_1 and bm_md_2 equal the only existing unique value.
                distance_scores.append(0)
            else:
                distance_scores.append(abs(bm_md_1[i_k] - bm_md_2[i_k]) / ranges[i_k])

        sim_scores = [1 - v for v in distance_scores]
        scores_over_pairs.append(mean(sim_scores))

    return mean(scores_over_pairs)


ConfigSpaceGowerSimilarity = BehaviorFunction(
    config_space_gower_similarity,
    ["input_metadata"],
    (0, 1),
    "none",
    name="Gower Similarity in Config Space (Lower is more Diverse)",
)
