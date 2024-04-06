# PHEM: A collection of Post Hoc Ensemble Methods for (Auto)ML

PHEM is a place to share Post Hoc Ensembling Methods for AutoML.

The method supported by PHEM aligns with (also not fully) the scikit-learn API.
This is a work in progress as transitioning from research code to a library is not trivial.
The algorithms and implementations behind the interfaces can also be (somewhat easily) adapted to be used in AutoML systems such as Auto-Sklearn or AutoGluon.

This repository only focuses on classification so far. But most methods could be trivially extended to regression (if I had the time).

Moreover, based on [Assembled](https://github.com/ISG-Siegen/assembled) or [TabRepo](https://github.com/autogluon/tabrepo) , we have efficient means to simulate ensembling methods.

## :warning: WORK IN PROGRESS REPOSITORY :warning:
This repository is a work in progress. The code is not yet fully tested and the documentation is not yet complete (or even started ;).

Yet, most of it works out of the box as they were used in research projects before and partially tested and implemented in AutoML systems.

Moreover, if you need anything or have any questions, feel free to open an issue or contact me directly. I am happy to help.

Finally, I would like to note that the code quality is also insufficient as this was build on Python 3.8, and before I knew of the existence of `ruff`.

### References
- Assembled: https://github.com/ISG-Siegen/assembled
- Post Hoc Ensembling Methods
    - Greedy Ensemble Selection (GES): Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first international conference on Machine learning. 2004.
    - Q(D)O-ES: Quality (Diversity) Optimisation Ensemble Selection https://github.com/LennartPurucker/PopulationBasedQDO-PostHocEnsembleSelectionAutoML
    - (TODO ADD) CMA-ES for Post Hoc Ensembling: https://github.com/LennartPurucker/CMA-ES-PostHocEnsemblingAutoML
