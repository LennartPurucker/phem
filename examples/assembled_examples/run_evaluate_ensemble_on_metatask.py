"""Minimal Example - Running Post Hoc Ensembling Methods with Metatassk from the Assembled framework.

The following runs a hyperparameter configuration for each method: Greedy Ensemble Selection (GES), the single best,
QDO-ES, and QO-ES. Thereby, we run it on base model data generated for a toy dataset from sklearn (TaskID "-1") that is stored in
this repository.

FYI: You can also run this from your IDE by adjusting the default input parameters to or editing `run_evaluate_ensemble_on_metatask.py`.

```shell
python run_evaluate_ensemble_on_metatask.py -1 "SingleBest" balanced_accuracy -1
python run_evaluate_ensemble_on_metatask.py -1 "GES" balanced_accuracy -1
python run_evaluate_ensemble_on_metatask.py -1 "QDO-ES" balanced_accuracy -1
python run_evaluate_ensemble_on_metatask.py -1 "QO-ES" balanced_accuracy -1
python run_evaluate_ensemble_on_metatask.py -1 "CMA-ES" balanced_accuracy -1
python run_evaluate_ensemble_on_metatask.py -1 "CMA-ES-ExplicitGES" balanced_accuracy -1
```

The results of the runs are stored under `examples/output/minimal_example_ens`
One needs to parse and evaluate them to compute the results which we build our evaluation on.
Otherwise, one can use the output of the script to obtain scores immediately.

## Detail Usage Documentation

To evaluate an ensemble on a metatask, execute the following script with the appropriate parameters.

1) `python run_evaluate_ensemble_on_metatask.py task_id pruner ensemble_method_name metric_name benchmark_name evaluation_name isolate_execution load_method folds_to_run_on config_space_name ens_save_name`
    * `task_id`: an OpenML task ID (for testing, pass `-1`)
    * `ensemble_method_name`: Name of the ensemble method's configuration
        * see `default_configurations/config_mgmt.py`
    * `metric_name`: metric name of the metric to be optimized by the ensemble method, we expect the import name of the
      metric
        * "roc_auc" or "balanced_accuracy"
    * `folds_to_run_on`: If "-1" all folds are run sequentially. Else the number corresponds to the fold on which the
      ensemble is evaluated.



"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask
from assembled.metatask import MetaTask

from phem.application_utils.default_configurations import get_ensemble_switch_case_config
from phem.application_utils.supported_metrics import msc

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    ensemble_method_name = sys.argv[2]
    metric_name = sys.argv[3]
    folds_to_run_on = sys.argv[4]

    if folds_to_run_on == "-1":
        folds_to_run_on = None
        state_ending = ""
    else:
        folds_to_run_on = [int(folds_to_run_on)]
        state_ending = f"_{folds_to_run_on}"

    # Default to -1
    n_jobs = -1

    # -- Build Paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_input_dir = file_path / "example_metatask"
    print(f"Path to Metatask: {tmp_input_dir}")

    out_path = file_path.parent / "output" / "run_evaluate_ensemble_on_metatask" / f"task_{openml_task_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    # -- Rebuild The Metatask
    print("Load Metatask")
    mt = MetaTask()
    mt.read_metatask_from_files(tmp_input_dir, openml_task_id)

    # -- Setup Evaluation variables
    # Get the metric(s)
    is_binary = len(mt.class_labels) == 2
    # If the ensemble requires the metric, we assume the labels to be encoded
    ens_metric = msc(metric_name, is_binary, list(range(mt.n_classes)))
    # For the final score, we need the original labels
    score_metric = msc(metric_name, is_binary, mt.class_labels)
    predict_method = "predict_proba" if ens_metric.requires_confidences else "predict"

    # -- Handle Config Input
    rng_seed = 315185350 if folds_to_run_on is None else 315185350 + int(folds_to_run_on[0])

    technique_run_args = get_ensemble_switch_case_config(ensemble_method_name,
                                                         rng_seed=rng_seed, metric=ens_metric, n_jobs=n_jobs,
                                                         is_binary=is_binary, labels=list(range(mt.n_classes)))
    print("Run for Config:", ensemble_method_name)

    # -- Run Evaluation
    print(f"#### Process Task {mt.openml_task_id} for Dataset {mt.dataset_name} with Ensemble Technique {ensemble_method_name} ####")

    scores = evaluate_ensemble_on_metatask(mt, technique_name=ensemble_method_name, **technique_run_args,
                                           output_dir_path=out_path, store_results="parallel",
                                           save_evaluation_metadata=True,
                                           return_scores=score_metric, folds_to_run=folds_to_run_on,
                                           use_validation_data_to_train_ensemble_techniques=True,
                                           verbose=True, isolate_ensemble_execution=False,
                                           predict_method=predict_method,
                                           store_metadata_in_fake_base_model=True)
    print(scores)
    print("K-Fold Average Test Performance:", sum(scores) / len(scores))
