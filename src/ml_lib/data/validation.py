"""Check data shapes, values, and timestamps before modeling."""
from collections.abc import Iterable
import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate column names are not allowed.")


def require_time_index(frame: pd.DataFrame | pd.Series, *, regular: bool = False) -> None:
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:
        raise ValueError("Expected a DatetimeIndex without NaT.")
    if len(idx) == 0:
        raise ValueError("Time-indexed input is empty.")
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted in increasing order.")
    if not (idx == idx.floor("h")).all():
        raise ValueError("Expected exact hourly bucket labels.")
    if regular and len(idx) > 1:
        deltas = idx[1:] - idx[:-1]
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise ValueError("Expected a complete hourly index. Reindex before positional lags/windows.")


def hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose absent hours as NaN; never infer that missing demand equals zero."""
    require_time_index(frame)
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h", name=frame.index.name))


def finite_array(values, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and entirely finite.")
    return result


def validate_partitions(frame, *, case_id, partition, expected_partitions, group_id=None):
    """Return declared partitions after checking identities and optional group isolation.

    Partition order comes from expected_partitions, not row order. This validates
    an existing assignment; it neither creates splits nor assumes time ordering.
    """
    names = list(expected_partitions)
    if not names or len(set(names)) != len(names) or pd.Index(names).hasnans:
        raise ValueError("Expected partition names must be nonempty, unique and present.")
    columns = [case_id, partition] + ([group_id] if group_id is not None else [])
    require_columns(frame, columns)
    if frame[columns].isna().any().any():
        raise ValueError("Case, partition and group identities must be present.")
    if frame[case_id].duplicated().any():
        raise ValueError("Case IDs must be unique across partitions.")
    if set(frame[partition]) != set(names):
        raise ValueError("Expected nonempty declared partitions only.")
    if group_id is not None:
        memberships = frame.groupby(group_id, sort=False)[partition].nunique()
        if (memberships > 1).any():
            raise ValueError("Group identities overlap across partitions.")
    return {name: frame.loc[frame[partition] == name].copy() for name in names}
