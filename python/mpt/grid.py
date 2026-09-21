"""Sampling an event table on a grid.

``grid_events`` turns a table whose rows are notes into one whose rows are
notes at grid points, a held note replicating across the points it
occupies. The MATLAB sibling is ``gridEvents``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["grid_events"]

#: Occupancy is an overlap with the slice, tested against a tolerance, so
#: that a note ending exactly where a slice begins does not occupy it.
_TOL = 1e-9


def grid_events(table, step, *, time="beats", duration="duration",
                weights="coverage", limits=None):
    """Sample an event table on a regular grid.

    Each grid point opens a slice of the chosen length, and a note
    belongs to every slice it overlaps, so a held note replicates across
    the slices it spans and a note shorter than the step still lands in
    one. A slice with nothing sounding becomes one row whose note
    columns are all missing. Those empty rows are kept: they hold the
    place that makes the event index a uniform index of time, which is
    what binding and differencing read. Dropping them afterwards is one
    selection away.

    Parameters
    ----------
    table : DataFrame
        An event table, as :func:`mpt.read_score` returns.
    step : float
        The grid step, in the chosen time unit.
    time : {'beats', 'seconds'}
        Which time base the grid runs over. A metrical grid presupposes a
        beat map, which a score has and a bare performance may not.
    duration : {'duration', 'sounding_duration'}
        Which duration defines occupancy: the recorded one, or the one
        with the pedals resolved.
    weights : {'coverage', 'presence', 'item'}
        What a slice takes from a note that overlaps it.

        ``'coverage'`` (default) takes the fraction of the slice the
        note fills: how the span is filled. This is the weighting of
        Analysis 1.3 of the JMM article, "the fraction of the eighth each
        note sounds".

        ``'presence'`` takes the note's full weight in every slice it
        appears in at all, however briefly: which notes are here, rather
        than how much of the span each occupies. It is a membership
        reading: each slice records the set of what occurs in it,
        whatever the step. It separates from coverage as the step grows
        relative to the notes -- at a bar-length step, say, a slice holds
        the set of what occurs in that bar, where coverage would hold a
        duration-weighted profile of it.

        ``'item'`` takes the fraction of the *note* in the slice, so that
        the note's weight is distributed over the slices it spans and it
        counts once in total; for an attribute constant over the note
        this gives a density identical to the ungridded one at ``r = 1``,
        the kernel being linear in weight.

        Coverage and presence differ only where a note does not fill a
        slice, so on a grid at or finer than the shortest note they
        agree. For what is sounding at a given moment, the instrument is
        coverage on a fine grid: an instant has no duration, and coverage
        approaches the momentary reading as the step shrinks.
    limits : (float, float), optional
        The half-open span the grid covers. The default runs from 0 to
        the last note's end.

    Returns
    -------
    DataFrame
        The gridded table. It carries the source table's columns, with
        ``duration`` still meaning the note's own duration and the slice
        length being a property of the grid, plus:

        ``grid_index``
            0-based position of the grid point.
        ``grid_onset_beats`` or ``grid_onset_seconds``
            The grid point's time, named for the unit gridded over.
        ``note_id``
            0-based row of the source table, missing on an empty point,
            so that the grid collapses back to the note table by
            grouping and nothing is lost.
        ``weight``
            Under the chosen policy. Where the source carries no weight
            of its own, every note weighs one.

        Integer and boolean columns become pandas' nullable ``Int64`` and
        ``boolean``, since an empty grid point has no value for them.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be an event table; got {type(table).__name__}.")
    if time not in ("beats", "seconds"):
        raise ValueError("time must be 'beats' or 'seconds'.")
    if duration not in ("duration", "sounding_duration"):
        raise ValueError(
            "duration must be 'duration' or 'sounding_duration'.")
    if weights not in ("coverage", "presence", "item"):
        raise ValueError(
            "weights must be 'coverage', 'presence', or 'item'.")
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive number.")

    onset_col = f"onset_{time}"
    dur_col = f"{duration}_{time}"
    for name in (onset_col, dur_col):
        if name not in table.columns:
            raise KeyError(
                f"The table has no {name!r} column, so it cannot be gridded "
                f"over {time}; a metrical grid needs a beat map, which a "
                f"bare performance may not have.")

    onset = table[onset_col].to_numpy(dtype=float)
    end = onset + table[dur_col].to_numpy(dtype=float)

    if limits is None:
        lo = 0.0
        hi = float(np.nanmax(end)) if len(table) else 0.0
    else:
        lo, hi = float(limits[0]), float(limits[1])
        if not hi > lo:
            raise ValueError("limits must be increasing.")
    n_points = int(np.ceil((hi - lo) / step - _TOL)) if hi > lo else 0
    times = lo + step * np.arange(n_points)

    grid_of, note_of, overlap_of = [], [], []
    for i, g in enumerate(times):
        overlap = np.minimum(end, g + step) - np.maximum(onset, g)
        occupied = np.nonzero(overlap > _TOL)[0]
        if occupied.size:
            grid_of.extend([i] * occupied.size)
            note_of.extend(occupied.tolist())
            overlap_of.extend(overlap[occupied].tolist())
        else:
            grid_of.append(i)
            note_of.append(-1)
            overlap_of.append(np.nan)

    grid_of = np.asarray(grid_of, dtype=np.intp)
    note_of = np.asarray(note_of, dtype=np.intp)
    overlap_of = np.asarray(overlap_of, dtype=float)
    live = note_of >= 0

    out = table.iloc[np.where(live, note_of, 0)].reset_index(drop=True)
    out = _nullable(out)
    if not live.all():
        out.loc[~live, :] = pd.NA

    base = (table["weight"].to_numpy(dtype=float) if "weight" in table.columns
            else np.ones(len(table)))
    dur = table[dur_col].to_numpy(dtype=float)
    weight = np.full(len(note_of), np.nan)
    if weights == "presence":
        weight[live] = base[note_of[live]]
    elif weights == "coverage":
        weight[live] = base[note_of[live]] * overlap_of[live] / step
    else:
        weight[live] = (base[note_of[live]] * overlap_of[live]
                        / dur[note_of[live]])

    out["grid_index"] = pd.array(grid_of, dtype="Int64")
    out[f"grid_onset_{time}"] = times[grid_of]
    out["note_id"] = pd.array(np.where(live, note_of, -1), dtype="Int64")
    out.loc[~live, "note_id"] = pd.NA
    out["weight"] = weight

    out.attrs = dict(table.attrs)
    out.attrs["granularity"] = "grid"
    out.attrs["grid_step"] = step
    out.attrs["grid_time"] = time
    return out


def _nullable(frame):
    """Integer and boolean columns that an empty grid point leaves blank.

    A missing value needs a dtype that can hold one, so the plain numpy
    integer and boolean columns become pandas' nullable equivalents.
    """
    out = frame.copy()
    for name in out.columns:
        dtype = out[name].dtype
        if pd.api.types.is_bool_dtype(dtype):
            out[name] = out[name].astype("boolean")
        elif pd.api.types.is_integer_dtype(dtype):
            out[name] = out[name].astype("Int64")
    return out
