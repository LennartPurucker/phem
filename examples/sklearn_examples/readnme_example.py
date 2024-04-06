from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection import EnsembleSelection

# -- Obtain Base Models from Sklearn
classifiers = [
    KNeighborsClassifier(3),
    AdaBoostClassifier(n_estimators=10, random_state=42),
    RandomForestClassifier(n_estimators=10, random_state=42),
]

X, y = make_classification(n_classes=2, n_redundant=2, n_informative=4, random_state=1)

# -- Example for holdout validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

fitted_base_models = [clf.fit(X_train, y_train) for clf in classifiers]

# Greedy Ensemble Selection from Caruana et al. 2004
ges = EnsembleSelection(
    base_models=fitted_base_models,
    n_iterations=50,
    # If the ensemble requires the metric, we assume the labels to be encoded
    metric=msc(metric_name="roc_auc", is_binary=True, labels=list(range(2))),
    random_state=1,
)


y_pred = ges.fit(X_val, y_val).predict_proba(X_test)
print(f"ROC AUC Test Score: {roc_auc_score(y_test, y_pred[:, 1])}")
