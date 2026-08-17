"""Epoch 选择协议检查 / Epoch-selection protocol checks."""

from typing import Mapping, Any


def validate_epoch_selection(training: Mapping[str, Any]) -> None:
    """只允许 fixed 或 inner-grouped / Allow only fixed or inner-grouped selection."""

    rule = training.get("epoch_rule")
    if training.get("outer_labels_visible_to_trainer") is not False:
        raise ValueError("outer labels must remain unavailable to epoch selection")
    if rule == "fixed_epoch":
        if int(training.get("fixed_epochs", 0)) <= 0 or int(training.get("inner_grouped_folds", -1)) != 0:
            raise ValueError("fixed_epoch requires positive fixed_epochs and zero inner folds")
    elif rule == "inner_grouped_selection":
        if int(training.get("inner_grouped_folds", 0)) < 2 or training.get("refit_on_all_outer_training") is not True:
            raise ValueError("inner selection requires grouped folds and outer-train refit")
    else:
        raise ValueError("unsupported epoch selection rule")


__all__ = ["validate_epoch_selection"]
