# PHEM: A collection of Post Hoc Ensemble Methods for (Auto)ML

PHEM is a place to share Post Hoc Ensembling Methods for (Auto)ML with tabular data.
The goal of post hoc ensembling (for AutoML) is to aggregate a pool of base models consisting of all models that are trained and validated during model selection or a subset thereof.

## Example on How to Use PHEM with Sklearn

```python
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
```

For more example, please see https://github.com/LennartPurucker/phem/tree/master/examples.

Based on [Assembled](https://github.com/ISG-Siegen/assembled) or [TabRepo](https://github.com/autogluon/tabrepo), we have efficient means to simulate post hoc ensembling methods.

## Installation

From source (latest version):
```bash
pip install git+https://github.com/LennartPurucker/phem
```

From pip (latest release, do not use this right now as pip might be outdated!):
```bash
pip install phem
```

## :warning: WORK IN PROGRESS REPOSITORY :warning:
This repository is a work in progress. The code is not yet fully tested and the documentation is not yet complete (or even started ;).
Yet, most of it works out of the box as they were used in research projects before and partially tested and implemented in AutoML systems.
Moreover, if you need anything or have any questions, feel free to open an issue or contact me directly. I am happy to help.

## Notes

The methods supported by PHEM aligns with (also not fully) the scikit-learn API.
This is a work in progress as transitioning from research code to a library is not trivial.
The algorithms and implementations behind the interfaces can also be (somewhat easily) adapted to be used in AutoML systems such as Auto-Sklearn or AutoGluon.

This repository only focuses on classification so far. But most methods can be trivially extended to regression (if I had the time).



## References
- Assembled: https://github.com/ISG-Siegen/assembled
- Background on Post Hoc Ensembling in AutoML: https://arxiv.org/abs/2307.08364
- Post Hoc Ensembling Methods
    - Greedy Ensemble Selection (GES): Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first international conference on Machine learning. 2004.
    - Q(D)O-ES: Quality (Diversity) Optimisation Ensemble Selection https://github.com/LennartPurucker/PopulationBasedQDO-PostHocEnsembleSelectionAutoML
    - (TODO ADD) CMA-ES for Post Hoc Ensembling: https://github.com/LennartPurucker/CMA-ES-PostHocEnsemblingAutoML
