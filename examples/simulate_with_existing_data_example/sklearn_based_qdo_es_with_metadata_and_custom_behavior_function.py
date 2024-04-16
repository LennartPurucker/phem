"""Example of Using Metadata in the current Framework.

(extension of simulate_based_on_sklearn_data.py)
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ribs.visualize import sliding_boundaries_archive_heatmap
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection.qdo import (
    QDOEnsembleSelection,
)
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace


@dataclass
class FakedFittedAndValidatedClassificationBaseModel:
    name: str
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    model_metadata: dict
    return_val_data: bool = True

    @property
    def probabilities(self):
        if self.return_val_data:
            return self.val_probabilities

        return self.test_probabilities

    def predict(self, X):
        return np.argmax(self.probabilities, axis=1)

    def predict_proba(self, X):
        return self.probabilities

    def switch_to_test_simulation(self):
        self.return_val_data = False

    def switch_to_val_simulation(self):
        self.return_val_data = True


@dataclass
class SimulationData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    base_models_data: list[FakedFittedAndValidatedClassificationBaseModel]


def generate_data_for_ensembles_with_sklearn() -> SimulationData:
    """Extension to include fit and predict times as part of metadata for the base models."""
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        GaussianNB(),
        MLPClassifier(max_iter=100, random_state=42),
        MLPClassifier(max_iter=10, random_state=42),
        MLPClassifier(max_iter=50, random_state=42),
        MLPClassifier(max_iter=150, random_state=42),
        MLPClassifier(max_iter=200, random_state=42),
        MLPClassifier(max_iter=500, random_state=42),
    ]

    X, y = make_classification(n_classes=2, n_redundant=2, n_informative=4, random_state=1)

    # -- Example for holdout validation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    _base_models_data = []
    for clf in classifiers:
        st_time = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - st_time

        st_time = time.time()
        val_probabilities = clf.predict_proba(X_val)
        val_predict_time = time.time() - st_time

        st_time = time.time()
        test_probabilities = clf.predict_proba(X_test)
        test_predict_time = time.time() - st_time

        bm_data = FakedFittedAndValidatedClassificationBaseModel(
            name=clf.__class__.__name__,
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            model_metadata={
                "fit_time": fit_time,
                "val_predict_time": val_predict_time,
                "test_predict_time": test_predict_time,
            },
        )
        _base_models_data.append(bm_data)

    sim_data = SimulationData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        base_models_data=_base_models_data,
    )
    with open("simulation_data.pkl", "wb") as f:
        pickle.dump(sim_data, f)


def get_custom_behavior_space_with_inference_time(
    max_possible_inference_time: float,
) -> BehaviorSpace:
    # Using ensemble size (an existing behavior function) and a custom behavior function to create a 2D behavior space.
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import LossCorrelationMeasure
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction

    EnsembleInferenceTime = BehaviorFunction(
        ensemble_inference_time,  # function to call.
        # define the required arguments for the function `ensemble_inference_time`
        required_arguments=["input_metadata"],
        # Define the initial starting range of the behavior space (due to using a sliding boundaries archive, this will be re-mapped anyhow)
        range_tuple=(0, max_possible_inference_time + 1),  # +1 for safety.
        # Defines which kind of prediction data is needed as input (if  any)
        required_prediction_format="none",
        name="Ensemble Inference Time",
    )

    return BehaviorSpace([LossCorrelationMeasure, EnsembleInferenceTime])


def ensemble_inference_time(input_metadata: list[dict]):
    """A custom behavior function.

    Some Notes:
        - The input_metadata here is the metadata for each base model in the ensemble. How this is called is defined in
            phem.methods.ensemble_selection.qdo.qdo_es.evaluate_single_solution.
        - The behavior function definition and arguments depends on its definition in the BehaviorFunction class (see below).
            For all options, see phem.methods.ensemble_selection.qdo.behavior_space.BehaviorFunction.
    """
    return sum([md["val_predict_time"] for md in input_metadata])


def simulate_post_hoc_ensembling_from_prediction_data():
    # -- Obtain data from the simulation data
    with open("simulation_data.pkl", "rb") as f:
        simulation_data: SimulationData = pickle.load(f)
    # We do not need the training data here as we only simulate post hoc ensembling.
    X_val, y_val = simulation_data.X_val, simulation_data.y_val
    X_test, y_test = simulation_data.X_test, simulation_data.y_test
    base_models = simulation_data.base_models_data

    max_possible_ensemble_infer_time = sum(
        [bm.model_metadata["val_predict_time"] for bm in base_models],
    )

    qdo_es = QDOEnsembleSelection(
        base_models=base_models,
        n_iterations=50,
        score_metric=msc(metric_name="roc_auc", is_binary=True, labels=list(range(2))),
        behavior_space=get_custom_behavior_space_with_inference_time(
            max_possible_inference_time=max_possible_ensemble_infer_time,
        ),
        random_state=1,
        # Define that we added custom metadata and not the pre-defined format from the Assembled Metatasks from Auto-Sklearn
        # This will not support a behavior space including the ConfigSpaceGowerSimilarity behavior function.
        # To do so, one would need to adapt the function or create a new one, see phem/methods/ensemble_selection/qdo/behavior_functions/
        base_models_metadata_type="custom",
    )

    # Switch to simulating predictions on validation data  (i.e., the training data of the ensemble)
    for bm in qdo_es.base_models:
        bm.switch_to_val_simulation()

    qdo_es.fit(X_val, y_val)

    # Plot Archive
    n_elites = len(list(qdo_es.archive))
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(qdo_es.archive, cmap="viridis", square=False)
    plt.title(f"Final Archive Heatmap (Validation Loss) for {n_elites} elites")

    ax = plt.gca()
    x_boundary = qdo_es.archive.boundaries[0]
    y_boundary = qdo_es.archive.boundaries[1]
    ax.vlines(
        x_boundary,
        qdo_es.archive.lower_bounds[1],
        qdo_es.archive.upper_bounds[1],
        color="k",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.hlines(
        y_boundary,
        qdo_es.archive.lower_bounds[0],
        qdo_es.archive.upper_bounds[0],
        color="k",
        linewidth=1,
        alpha=0.5,
    )
    ax.set(xlabel="Diversity", ylabel="Inference Time")
    ax.set_xlim(
        min(x_boundary) * 0.95,
        max(x_boundary) * 1.05,
    )
    ax.set_ylim(
        min(y_boundary) * 0.95 - 0.0005,
        max(y_boundary) * 1.05,
    )
    plt.show()

    # Simulate to simulating predictions on test data  (i.e., the test data of the ensemble and base models)
    for bm in qdo_es.base_models:
        bm.switch_to_test_simulation()

    y_pred = qdo_es.predict_proba(X_test)
    print(f"ROC AUC Test Score from Simulate Data: {roc_auc_score(y_test,y_pred[:,1])}")


if __name__ == "__main__":
    generate_data_for_ensembles_with_sklearn()
    simulate_post_hoc_ensembling_from_prediction_data()
