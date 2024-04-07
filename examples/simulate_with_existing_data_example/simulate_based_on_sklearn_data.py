"""Example of simulating post hoc ensembling with existing data from sklearn models.

This is a re-implementation of the sklearn example from the README (examples/sklearn_examples/readme_example.py).
But instead of building ensembles with the sklearn models, here, we show how to simulate building ensemblels from prediction data only.

In essence, we first have to generate data (just like we would usually do) and then need to save it to disk.
Alternatively, we can use data from [Assembled](https://github.com/ISG-Siegen/assembled) or [TabRepo](https://github.com/autogluon/tabrepo).
Thus, allowing us to efficiently test many different ensembling methods without having to retrain the base models.

Later, we load the data and use it to simulate post hoc ensembling as we do not need the original sklearn model to do so.
Such a workflow is an extremely simplified version of what Metatasks do in the Assembled framework.

This script reproduces the score of the example from the README without having to store or manage the original sklearn model.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection import EnsembleSelection


@dataclass
class FakedFittedAndValidatedClassificationBaseModel:
    """Fake sklearn-like base model (classifier) usable by ensembles in the same way as real base models.

    To simulate validation and test predictions, we start by default with returning validation predictions.
    Then, after fitting the ensemble on the validation predictions, we switch to returning test predictions using `switch_to_test_simulation`.

    Parameters
    ----------
    name: str
        Name of the base model.
    val_probabilities: list[np.ndarray]
        The predictions of the base model on the validation data.
    test_probabilities: list[np.ndarray]
        The predictions of the base model on the test data.
    return_val_data : bool, default=True
        If True, the val_probabilities are returned. If False, the test_probabilities are returned.
    """

    name: str
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
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
    """Dataclass to store simulation data for one task."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    base_models_data: list[FakedFittedAndValidatedClassificationBaseModel]


def generate_data_for_ensembles_with_sklearn() -> SimulationData:
    """Generate data for ensembles with sklearn.

    This code creates a simpel holdout classification scenario and generates prediction data for it that can be saved and later used for post hoc ensembling.

    In detail, to simulate any kind of post hoc ensembling, we need:
        - Prediction data for a pool of base models, that is for each base model:
            - A name for the base model.
            - Predictions on the validation data. Here, validation data can be either from holdout or cross-validation.
            - Predictions on the test data.
        - Task data, that is:
            - X_train, y_train - the training data that was used to fit the model.
                If the base model was fitted on all training data, e.g. in cross-validation with bagging like inn AutoGluon, this is the entire training data.
                This is not necessary for post hoc ensembling, but we include it here for completeness.
            - X_val, y_val - the validation data, if cross-validation is used, this is the entire training data.
            - X_test, y_test - the test data that we want to predict for (holdout and never used for any model fitting).

    We can generate this data by fitting some base models on the training data and then predicting on the validation and test data.
    Then, we can save the predictions and the original data to disk and use it later for post hoc ensembling. This function does this for sklearn models.

    Alternatively, we can use data from [Assembled](https://github.com/ISG-Siegen/assembled) or [TabRepo](https://github.com/autogluon/tabrepo).

    Here, we store all data in dataclasses that we can use to simulate post hoc ensembling later.
    """
    classifiers = [
        KNeighborsClassifier(3),
        AdaBoostClassifier(n_estimators=10, random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=42),
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
        clf.fit(X_train, y_train)
        bm_data = FakedFittedAndValidatedClassificationBaseModel(
            name=clf.__class__.__name__,
            val_probabilities=clf.predict_proba(X_val),
            test_probabilities=clf.predict_proba(X_test),
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


def simulate_post_hoc_ensembling_from_preediction_data():
    # -- Obtain data from the simulation data
    with open("simulation_data.pkl", "rb") as f:
        simulation_data: SimulationData = pickle.load(f)
    # We do not need the training data here as we only simulate post hoc ensembling.
    X_val, y_val = simulation_data.X_val, simulation_data.y_val
    X_test, y_test = simulation_data.X_test, simulation_data.y_test
    base_models = simulation_data.base_models_data

    ges = EnsembleSelection(
        base_models=base_models,
        n_iterations=50,  # If the ensemble requires the metric, we assume the labels to be encoded
        metric=msc(metric_name="roc_auc", is_binary=True, labels=list(range(2))),
        random_state=1,
    )

    # Switch to simulating predictions on validation data  (i.e., the training data of the ensemble)
    for bm in ges.base_models:
        bm.switch_to_val_simulation()

    ges.fit(X_val, y_val)

    # Simulate to simulating predictions on test data  (i.e., the test data of the ensemble and base models)
    for bm in ges.base_models:
        bm.switch_to_test_simulation()

    y_pred = ges.predict_proba(X_test)
    print(f"ROC AUC Test Score from Simulate Data: {roc_auc_score(y_test,y_pred[:,1])}")


if __name__ == "__main__":
    generate_data_for_ensembles_with_sklearn()
    simulate_post_hoc_ensembling_from_preediction_data()
