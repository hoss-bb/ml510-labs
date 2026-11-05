"""Regression comparisons within caller-defined, potentially overlapping groups."""
import pandas as pd
from ml_lib.data.validation import finite_array
from .metrics import regression_metrics


def _align(values, index, name):
    if not isinstance(values, pd.Series) or not values.index.is_unique or values.index.hasnans:
        raise ValueError(f"{name} requires a Series with unique, nonmissing case IDs.")
    if len(values) != len(index) or not index.isin(values.index).all():
        raise ValueError(f"{name} must contain exactly the expected case IDs.")
    return values.reindex(index)


def regression_group_comparisons(actual, predictions, *, baseline, groups, entity_ids=None):
    """Align all Series by case ID, then compare each model with one baseline.

    groups maps names to Boolean Series. Groups may overlap and may be empty;
    they need not exhaust the population. Entity counts are optional. No group
    adequacy, model selection or domain-specific validity rule is inferred.
    """
    if not isinstance(actual, pd.Series) or actual.empty:
        raise ValueError("Actual values require a nonempty indexed Series.")
    actual = _align(actual, actual.index, "Actual values")
    finite_array(actual, name="Actual values")
    if baseline not in predictions:
        raise ValueError("The named baseline must be included in predictions.")
    aligned = {}
    for name, values in predictions.items():
        aligned[name] = _align(values, actual.index, f"Predictions for {name}")
        finite_array(aligned[name], name=f"Predictions for {name}")
    entities = None
    if entity_ids is not None:
        entities = _align(entity_ids, actual.index, "Entity IDs")
        if entities.isna().any():
            raise ValueError("Entity IDs must be present.")
    rows = []
    for group, supplied_mask in groups.items():
        mask = _align(supplied_mask, actual.index, f"Group {group}")
        if not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any():
            raise ValueError("Group masks must contain Boolean values without missing entries.")
        mask = mask.astype(bool)
        n = int(mask.sum())
        baseline_mae = regression_metrics(actual[mask], aligned[baseline][mask])["mae"] if n else None
        for name, values in aligned.items():
            scores = regression_metrics(actual[mask], values[mask]) if n else {}
            mae = scores.get("mae")
            improvement = baseline_mae - mae if n else None
            rows.append({"group": group, "model": name, "n": n,
                         "entities": int(entities[mask].nunique()) if entities is not None else None,
                         "mae": mae, "rmse": scores.get("rmse"), "baseline_mae": baseline_mae,
                         "mae_improvement": improvement,
                         "mae_reduction_percent": 100 * improvement / baseline_mae if n and baseline_mae > 0 else None})
    return pd.DataFrame(rows, columns=["group", "model", "n", "entities", "mae", "rmse",
                                      "baseline_mae", "mae_improvement", "mae_reduction_percent"])
