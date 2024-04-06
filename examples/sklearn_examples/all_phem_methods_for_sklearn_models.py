"""Example Code for running post-hoc ensembling methods with sklearn models."""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from phem.application_utils.supported_metrics import msc

# PHEM Code
from phem.methods.baselines import SingleBest
from phem.methods.ensemble_selection import EnsembleSelection
from phem.methods.ensemble_selection.qdo import (
    QDOEnsembleSelection,
    get_bs_ensemble_size_and_loss_correlation,
)

# -- Obtain Base Models from Sklearn
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5,
        n_estimators=10,
        max_features=1,
        random_state=42,
    ),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
]

X, y = make_classification(
    n_samples=10000,
    n_classes=2,
    n_features=20,
    n_redundant=2,
    n_informative=4,
    random_state=1,
    n_clusters_per_class=5,
)

# -- Example for holdout validation. Using out-of-fold predictions from cross-validation, one use the entire training data as validation data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
)

base_models = [clf.fit(X_train, y_train) for clf in classifiers]

# We assume the labels to be encoded within the ensembles.
metric_for_post_hoc_ensembling = msc("roc_auc", True, list(range(2)))


for _name, _clf in [
    (
        "SingleBest",
        SingleBest(
            base_models=base_models,
            metric=metric_for_post_hoc_ensembling,
            predict_method="predict_proba",
        ),
    ),
    (
        "GreedyEnsembleSelection",
        EnsembleSelection(
            base_models=base_models,
            n_iterations=50,
            # If the ensemble requires the metric, we assume the labels to be encoded
            metric=metric_for_post_hoc_ensembling,
            random_state=1,
            use_best=True,
        ),
    ),
    (
        "QOEnsembleSelection",
        QDOEnsembleSelection(
            base_models=base_models,
            n_iterations=50,
            archive_type="quality",
            score_metric=metric_for_post_hoc_ensembling,
            random_state=1,
        ),
    ),
    (
        "QDOEnsembleSelection",
        QDOEnsembleSelection(
            base_models=base_models,
            n_iterations=50,
            score_metric=metric_for_post_hoc_ensembling,
            behavior_space=get_bs_ensemble_size_and_loss_correlation(),
            random_state=1,
        ),
    ),
]:
    _clf.fit(X_val, y_val)
    print(f"\nTesting {_name}")
    y_pred = _clf.predict_proba(X_test)
    print(f"ROC AUC Test Score: {roc_auc_score(y_test, y_pred[:, 1])}")
