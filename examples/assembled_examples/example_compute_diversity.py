"""Minimal Example - Computing Diversity.

This is an example on how to compute diversity given a metatask.
"""
from __future__ import annotations

import os
from pathlib import Path

from assembled.metatask import MetaTask

from phem.base_utils.diversity_metrics import LossCorrelation


def _read_mt(openml_task_id) -> MetaTask:
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_input_dir = file_path.parent / "data"
    print(f"Path to Metatask: {tmp_input_dir}")

    # -- Rebuild The Metatask
    print("Load Metatask")
    mt = MetaTask()
    mt.read_metatask_from_files(tmp_input_dir, openml_task_id)
    return mt


def _run(mt: MetaTask):
    # If needed, I can transform this into a notebook.

    # select a fold to focus on for this example (we always have all folds for an openml task)
    fold = 0

    # -- Print the names of all base models in the metatask for this fold
    fold_bms = mt.get_predictors_for_fold(fold)
    print("Base Models:", fold_bms)

    # -- Get all algorithms (custom code for metatasks produced by auto-sklearn).
    # We have access to the config from auto-sklearn via mt.predictor_descriptions.
    fold_algos = {mt.predictor_descriptions[bm]["config"]["classifier:__choice__"] for bm in fold_bms}
    print("Algorithms:", list(fold_algos))

    # -- Get all data associated to this fold that is stored in the metatask object
    _, X_train, X_test, y_train, y_test, val_base_predictions, test_base_predictions, \
        val_base_confidences, test_base_confidences = next(mt.yield_evaluation_data([fold]))

    # The validation predictions were computed on a holdout set, hence we first need to select this holdout set.
    val_indices = mt.meta_data["validation_indices"][fold]
    val_y_train = y_train.loc[val_indices]
    val_base_confidences = val_base_confidences.loc[val_indices]

    # -- Compute diversity for the set of all base models for validation and test data
    div_metric = LossCorrelation  # specific diversity metric (requires specific input etc.)

    # transform to common format in post hoc ensembling / for the metrics
    val_list_of_preds = [val_base_confidences[mt.get_validation_confidences_columns([bm])].values for bm in fold_bms]
    test_list_of_preds = [test_base_confidences[mt.get_conf_cols([bm])].values for bm in fold_bms]

    print("Validation Diversity of all base models:", div_metric(val_y_train.values, val_list_of_preds))
    print("Test Diversity of all base models:", div_metric(y_test, test_list_of_preds))


if __name__ == "__main__":
    _run(_read_mt("-1"))
