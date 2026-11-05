"""Listing-specific partitions, reporting groups and interpretation."""
import pandas as pd
from ml_lib.data.validation import validate_partitions
from ml_lib.evaluation.groups import regression_group_comparisons

ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
CAPACITIES = ['1-2 guests', '3-4 guests', '5+ guests']


def check_partitions(data):
    return validate_partitions(data, case_id="listing_id", partition="split", group_id="host_id",
                               expected_partitions=["train", "validation", "test"])


def group_masks(part):
    """Two separate views, never their cross-product."""
    for room in ROOM_TYPES:
        yield 'room_type', room, part.room_type == room
    capacity = pd.cut(part.accommodates, [0, 2, 4, float('inf')], labels=CAPACITIES)
    for group in CAPACITIES:
        yield 'capacity', group, capacity == group


def group_coverage(parts):
    rows = []
    for split, part in parts.items():
        for view, group, mask in group_masks(part):
            rows.append({'split': split, 'view': view, 'group': group,
                         'n': int(mask.sum()), 'hosts': int(part.loc[mask, 'host_id'].nunique())})
    return pd.DataFrame(rows)


def group_comparisons(part, predictions, *, split):
    """Use generic aligned arithmetic; keep listing-specific judgments here."""
    index = pd.Index(part.listing_id)
    actual = pd.Series(part.advertised_price.to_numpy(), index=index)
    entities = pd.Series(part.host_id.to_numpy(), index=index)
    groups = {(view, group): pd.Series(mask.to_numpy(), index=index)
              for view, group, mask in group_masks(part)}
    report = regression_group_comparisons(actual, predictions, baseline="median",
                                         groups=groups, entity_ids=entities)
    rows = []
    for row in report.to_dict("records"):
        view, group = row.pop("group")
        mask = groups[(view, group)]
        selected = predictions[row["model"]].reindex(index)[mask]
        note = ""
        if view == "room_type" and group == "Shared room":
            note = "Sparse group in this snapshot; do not treat as a reliable ranking."
        if not row["n"]:
            note = "No observations in this split; metrics unavailable."
        row["hosts"] = row.pop("entities")
        rows.append({"split": split, "view": view, "group": group, **row,
                     "negative_predictions": int((selected < 0).sum()) if row["n"] else None,
                     "evidence_note": note})
    return pd.DataFrame(rows, columns=["split", "view", "group", "model", "n", "hosts", "mae", "rmse",
                                      "baseline_mae", "mae_improvement", "mae_reduction_percent",
                                      "negative_predictions", "evidence_note"])
