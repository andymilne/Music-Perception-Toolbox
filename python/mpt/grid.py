"""Sampling an attribute table on a grid.

``grid_attr_table`` turns a table whose rows are notes into one whose rows are
notes at grid points, a held note replicating across the points it
occupies. The MATLAB sibling is ``gridAttrTable``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = ["grid_attr_table", "ungrid_attr_table"]

#: Occupancy is an overlap with the slice, tested against a tolerance, so
#: that a note ending exactly where a slice begins does not occupy it.
_TOL = 1e-9


def grid_attr_table(table, step, *, time="beats", duration="duration",
                weights="coverage", limits=None):
    """Sample an attribute table on a regular grid.

    The grid is a series of time points spaced ``step`` apart. Each
    point opens a *slice*, reaching from it up to the next, and a
    note occupies every slice it sounds in, so a held note occupies
    several and a note shorter than the step still occupies one. A note
    that ends exactly where a slice begins does not occupy it.

    The points are equally spaced in the unit the grid steps in --- in
    beats for a beat grid. Under a changing tempo that is not equal
    spacing in seconds, so the slices last different amounts of clock
    time.

    A slice with nothing sounding becomes one row whose note columns are
    all missing. Those empty rows are kept: they hold the
    place that makes the event index a uniform index of time, which is
    what binding and differencing read. Dropping them afterwards is one
    selection away.

    An already-gridded table regrids at a coarser step, and gridding
    composes: ``grid_attr_table(grid_attr_table(t, fine), coarse)``
    equals ``grid_attr_table(t, coarse)``. This is how a weighting the
    toolbox does not itself offer is reached --- weight the fine slices
    as the analysis requires, then coarsen, and the weights carry. The
    coarse step must be a whole multiple of the fine one, the time base
    must be the one the table was gridded over, ``weights`` must name
    the policy the table's weights already carry, since that is what
    fixes how they combine, and ``limits`` is refused, the span being
    the one the table covers.

    A grid does not refine. Coarsening combines the fine slices, which
    is determined; refining would have to divide a slice's weight among
    finer ones, which is not --- and a weighting the caller wrote into
    the grid is exactly what would have to be divided. The route is to
    grid the notes again, ``grid_attr_table(ungrid_attr_table(g),
    finer)``, which says plainly that the grid is being left behind.

    Parameters
    ----------
    table : DataFrame
        An attribute table, as :func:`mpt.read_score` returns.
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
        note fills: how the slice is filled. This is the weighting of
        Analysis 1.3 of the JMM article, "the fraction of the eighth each
        note sounds".

        ``'presence'`` takes the note's full weight in every slice it
        appears in at all, however briefly: which notes are here, rather
        than how much of the slice each occupies. It is a membership
        reading: each slice records the set of what occurs in it,
        whatever the step. It separates from coverage as the step grows
        relative to the notes -- at a bar-length step, say, a slice holds
        the set of what occurs in that bar, where coverage would hold a
        duration-weighted profile of it.

        ``'item'`` takes the fraction of the *note* in the slice, so that
        the note's weight is distributed over the slices it covers and it
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
        ``grid_onset_beats`` and ``grid_onset_seconds``
            The grid point's time in each unit. The grid steps in the
            chosen one; the other is interpolated from the table's note
            samples (one pair per onset, one per note end) and continued
            at the nearest rate beyond the first and last, so that
            metrical slices can be read on a clock -- a sigma in
            milliseconds over a grid of sixteenths. That is exact
            wherever the tempo is constant across the bracketing samples
            and approximate only across a tempo change inside a gap.
            Only the stepped unit is present where the source carries no
            second time base.
        ``note_id``
            0-based row of the source table, missing on an empty point,
            so that the grid collapses back to the table it came from
            and nothing is lost.
        ``weight``
            Under the chosen policy. Where the source carries no weight
            of its own, every note weighs one.

        Integer and boolean columns become pandas' nullable ``Int64`` and
        ``boolean``, since an empty grid point has no value for them.

    See Also
    --------
    ungrid_attr_table : The inverse, collapsing the grid back to the
        table it came from.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be an attribute table; got {type(table).__name__}.")
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

    if table.attrs.get("granularity") == "grid":
        return _coarsen_grid(table, step, time=time, weights=weights,
                             limits=limits)

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
    other = "seconds" if time == "beats" else "beats"
    other_times = _in_other_unit(table, time, other, times)
    if other_times is not None:
        out[f"grid_onset_{other}"] = other_times[grid_of]
    out["note_id"] = pd.array(np.where(live, note_of, -1), dtype="Int64")
    out.loc[~live, "note_id"] = pd.NA
    out["weight"] = weight

    out.attrs = dict(table.attrs)
    out.attrs["granularity"] = "grid"
    out.attrs["grid_step"] = step
    out.attrs["grid_time"] = time
    out.attrs["grid_weights"] = weights
    return out



def _coarsen_grid(table, step, *, time, weights, limits):
    """Regrid an already-gridded table at a coarser step.

    A gridded row records that a note sounds in a slice, with a weight;
    the note's own onset and duration are carried through unchanged, so
    reading them again would re-expand the note once per row it already
    has. Coarsening instead composes: the fine slices falling in one
    coarse slice are combined, note by note, and each policy combines in
    the way that makes ``grid_attr_table(grid_attr_table(t, fine),
    coarse)`` agree with ``grid_attr_table(t, coarse)`` exactly.

    ``coverage`` sums and rescales by ``fine / coarse``, coverage being a
    fraction of the slice and the slice having grown. ``item`` sums, its
    normalization by the note's length making it additive already.
    ``presence`` takes the maximum, an indicator over the coarse slice
    holding wherever it holds over a fine one.
    """
    fine_step = float(table.attrs["grid_step"])
    fine_time = table.attrs.get("grid_time", time)
    fine_weights = table.attrs.get("grid_weights")
    if time != fine_time:
        raise ValueError(
            f"this table was gridded over {fine_time}, so it cannot be "
            f"regridded over {time}; ungrid it first to change the time "
            f"base.")
    if fine_weights is not None and weights != fine_weights:
        raise ValueError(
            f"this table's weights are {fine_weights!r}, which is how they "
            f"combine; regridding it under {weights!r} would read them as "
            f"something they are not.")
    if limits is not None:
        raise ValueError(
            "limits cannot be given when regridding: the span is the one "
            "the table already covers.")
    ratio = step / fine_step
    if ratio < 1.0 - _TOL:
        raise ValueError(
            f"step {step:g} is finer than this table's step {fine_step:g}. "
            f"Coarsening combines the fine slices, which is determined; "
            f"refining would have to divide a slice's weight among finer "
            f"ones, which is not. Grid the notes again instead: "
            f"grid_attr_table(ungrid_attr_table(g), {step:g}).")
    if abs(ratio - round(ratio)) > _TOL:
        raise ValueError(
            f"step {step:g} is not a whole multiple of this table's step "
            f"{fine_step:g}, so the coarse slices would not align with the "
            f"fine ones.")
    ratio = int(round(ratio))

    onset_col = f"grid_onset_{time}"
    fine_index = table["grid_index"].to_numpy(dtype=float)
    n_fine = int(np.nanmax(fine_index)) + 1 if len(table) else 0
    n_coarse = -(-n_fine // ratio)
    if ratio == 1 or n_coarse == 0:
        out = table.copy()
        out.attrs = dict(table.attrs)
        out.attrs["grid_step"] = step
        return out

    lo = float(table[onset_col].min())
    other = "seconds" if time == "beats" else "beats"
    other_col = f"grid_onset_{other}"
    has_other = other_col in table.columns

    # The coarse points' times: every ratio-th fine point is one.
    keep = np.isin(fine_index, np.arange(n_coarse) * ratio)
    heads = (table.loc[keep, [onset_col] + ([other_col] if has_other else [])]
             .assign(_c=(fine_index[keep] // ratio).astype(int))
             .drop_duplicates("_c").set_index("_c").sort_index())
    coarse_time = lo + step * np.arange(n_coarse)
    coarse_other = (heads[other_col].to_numpy(dtype=float)
                    if has_other else None)

    live = table[table["note_id"].notna()].copy()
    live["_c"] = (live["grid_index"].to_numpy(dtype=int) // ratio)
    live = live.sort_values(["_c", "note_id", "grid_index"], kind="stable")
    group = live.groupby(["_c", "note_id"], sort=True, observed=True)
    if weights == "presence":
        combined = group["weight"].max()
    elif weights == "coverage":
        combined = group["weight"].sum() * (fine_step / step)
    else:
        combined = group["weight"].sum()
    rows = live.loc[group["grid_index"].idxmin()].copy()
    rows["weight"] = combined.to_numpy()

    # A coarse slice holding no note keeps its place as one blank row.
    # Reindexing an empty frame fills each column with the missing value
    # its own dtype takes, so the dtypes survive the concatenation.
    filled = np.zeros(n_coarse, dtype=bool)
    filled[rows["_c"].to_numpy(dtype=int)] = True
    blanks = np.nonzero(~filled)[0]
    if blanks.size:
        pad = rows.iloc[:0].reindex(range(blanks.size))
        pad["_c"] = blanks
        rows = pd.concat([rows.reset_index(drop=True), pad],
                         ignore_index=True)
    rows = (rows.sort_values(["_c", "note_id"], kind="stable",
                             na_position="last")
            .reset_index(drop=True))

    coarse = rows["_c"].to_numpy(dtype=int)
    out = rows.drop(columns=["_c"])
    out["grid_index"] = pd.array(coarse, dtype="Int64")
    out[onset_col] = coarse_time[coarse]
    if has_other:
        out[other_col] = coarse_other[coarse]
    out = out[list(table.columns)]
    out.attrs = dict(table.attrs)
    out.attrs["grid_step"] = step
    out.attrs["grid_weights"] = weights
    return out

#: The columns the grid writes, which ungridding removes. ``weight`` is
#: among them: the grid folds any weight the source carried into a
#: per-slice one, so the source's own weight does not survive gridding.
_GRID_COLUMNS = ("grid_index", "grid_onset_beats", "grid_onset_seconds",
                 "note_id", "weight")


def ungrid_attr_table(table):
    """Collapse a gridded attribute table back to the one it came from.

    The inverse of :func:`grid_attr_table`. ``note_id`` names the row of
    the source each grid row came from, so keeping the first row of each
    and removing the columns the grid wrote returns the source table.

    Which columns those are is the toolbox's business rather than the
    caller's, and it is not a fixed list: the grid *adds* ``weight`` to a
    table that had none and *folds* the weight of one that did into a
    per-slice one, so
    a caller comparing the two tables' columns would keep a ``weight``
    whose values are no longer the note's but the slice's.

    Parameters
    ----------
    table : DataFrame
        A gridded attribute table, as :func:`grid_attr_table` returns.

    Returns
    -------
    DataFrame
        The source table, indexed from 0. Two things do not come back: a
        note that ``limits`` cut out of the grid's span, which is a
        truncation the caller asked for, and the source's own ``weight``
        column where it had one, the grid having folded it into a
        per-slice weight. Read the source again if you need that.

    Warns
    -----
    UserWarning
        Where a kept column takes more than one value within a note. A
        column of the source repeats unchanged across a note's slices,
        so only one the caller added can vary; a per-slice quantity has
        no single value to collapse to, and the first slice's is taken.

    Notes
    -----
    Rows and columns may go before ungridding: the empty points, a run
    of slices, a column. A note whose first slice was dropped comes back
    from its next, since the source's columns repeat unchanged across
    its slices; a note whose slices were all dropped does not come back
    at all, having been selected away.

    See Also
    --------
    grid_attr_table : The inverse.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be a gridded attribute table; got "
            f"{type(table).__name__}.")
    if "note_id" not in table.columns:
        raise KeyError(
            "The table has no 'note_id' column, so it did not come from "
            "grid_attr_table and there is no grid to undo.")
    live = table.dropna(subset=["note_id"])
    keep = [c for c in live.columns if c not in _GRID_COLUMNS]
    out = (live.drop_duplicates("note_id")
               .sort_values("note_id")
               .drop(columns=[c for c in _GRID_COLUMNS if c in live.columns])
               .reset_index(drop=True))

    # A column the grid wrote is gone; a column of the source repeats
    # unchanged across a note's slices. One the caller added may not:
    # a per-slice quantity has no single value to collapse to, and the
    # first slice's is taken.
    if keep and len(live):
        counts = live.groupby("note_id", observed=True)[keep].nunique(
            dropna=False)
        varying = [c for c in keep if counts[c].max() > 1]
        if varying:
            warnings.warn(
                f"{', '.join(varying)} vary within a note, so they are not "
                "properties of the note the grid came from; the first "
                "slice's value is taken. A per-slice quantity is lost by "
                "ungridding, which is what ungridding means.",
                UserWarning, stacklevel=2)
    out = _unnullable(out)
    out.attrs = {k: v for k, v in table.attrs.items()
                 if k not in ("granularity", "grid_step", "grid_time",
                              "grid_weights")}
    return out


def _unnullable(frame):
    """Undo the nullable dtypes gridding imposed, an empty grid point
    having had no value for an integer or boolean column."""
    out = frame.copy()
    for name in out.columns:
        kind = str(out[name].dtype)
        if kind == "Int64":
            out[name] = out[name].astype("int64")
        elif kind == "boolean":
            out[name] = out[name].astype(bool)
    return out


def _in_other_unit(table, time, other, times):
    """The grid points' times in the unit the grid did not step in.

    The grid steps in one unit, but its points have a time in both, and
    an analysis may want metrical slices read on a clock -- a sigma in
    milliseconds over a grid of sixteenths. The table carries the
    correspondence only as samples, one pair per note onset and one per
    note end, so the map between the units is interpolated linearly
    between them and continued at the nearest rate beyond the first and
    last. That is exact wherever the tempo is constant across the
    bracketing samples and approximate only across a tempo change inside
    a gap.

    Returns ``None`` where the source carries only one time base, a bare
    performance having no beat map to read.
    """
    have, want = f"onset_{time}", f"onset_{other}"
    if have not in table.columns or want not in table.columns:
        return None
    x = np.concatenate([table[have].to_numpy(dtype=float),
                        (table[have].to_numpy(dtype=float)
                         + table[f"duration_{time}"].to_numpy(dtype=float))])
    y = np.concatenate([table[want].to_numpy(dtype=float),
                        (table[want].to_numpy(dtype=float)
                         + table[f"duration_{other}"].to_numpy(dtype=float))])
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0:
        return None
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x, first = np.unique(x, return_index=True)
    y = y[first]
    if x.size == 1:
        return np.full(times.shape, y[0])
    inner = np.interp(times, x, y)
    # np.interp clamps outside the samples; continue the end rates
    # instead, so that a slice before the first note or after the last
    # still has a time rather than the first or last note's.
    lo_rate = (y[1] - y[0]) / (x[1] - x[0])
    hi_rate = (y[-1] - y[-2]) / (x[-1] - x[-2])
    below, above = times < x[0], times > x[-1]
    inner[below] = y[0] + lo_rate * (times[below] - x[0])
    inner[above] = y[-1] + hi_rate * (times[above] - x[-1])
    return inner


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
