"""Tool to more easily get and support different metrics."""

from __future__ import annotations

from functools import partial

from sklearn.metrics import balanced_accuracy_score

from phem.base_utils.custom_metrics.roc_auc import roc_auc_score
from phem.base_utils.metrics import make_metric


def msc(metric_name, is_binary, labels):
    if metric_name == "balanced_accuracy":
        return make_metric(
            balanced_accuracy_score,
            metric_name="balanced_accuracy",
            maximize=True,
            classification=True,
            always_transform_conf_to_pred=True,
            optimum_value=1,
            requires_confidences=False,
        )

    elif metric_name == "roc_auc":
        if is_binary:
            return make_metric(
                roc_auc_score,
                metric_name="roc_auc",
                maximize=True,
                classification=True,
                always_transform_conf_to_pred=False,
                optimum_value=1,
                requires_confidences=True,
                only_positive_class=True,
                pos_label=1,
            )
        else:
            return make_metric(
                partial(roc_auc_score, average="macro", multi_class="ovr", labels=labels),
                metric_name="roc_auc",
                maximize=True,
                classification=True,
                always_transform_conf_to_pred=False,
                optimum_value=1,
                requires_confidences=True,
            )
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
