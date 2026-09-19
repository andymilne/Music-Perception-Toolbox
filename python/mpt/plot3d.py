"""Three-dimensional drawing of an expectation tensor density.

Mirror of MATLAB plotMaet3d. Two of its three methods are here,
``ellipsoids`` and ``points``; the third, ``slices``, rests on
texture-mapped surfaces, which matplotlib has no counterpart for.

matplotlib is imported when a plot is drawn rather than when the
toolbox is imported, so that the package's own dependencies stay
``numpy`` and ``scipy``.
"""

from __future__ import annotations

import numpy as np

from .tensor import eval_maet, maet_centres


__all__ = ["plot_maet_3d", "plot_maet_3d_points"]


def _unit_sphere(n_lon=16, n_lat=10):
    """A quadrilateral mesh on the unit sphere, built here rather than
    taken from the plotting library so that the two languages draw the
    same mesh."""
    lon = np.arange(n_lon) * 2.0 * np.pi / n_lon
    lat = np.linspace(0.0, np.pi, n_lat)
    t, p = np.meshgrid(lon, lat, indexing='ij')
    verts = np.column_stack([(np.sin(p) * np.cos(t)).ravel(),
                             (np.sin(p) * np.sin(t)).ravel(),
                             np.cos(p).ravel()])
    faces = []
    for i in range(n_lon):
        nxt = (i + 1) % n_lon
        for j in range(n_lat - 1):
            faces.append([i * n_lat + j, nxt * n_lat + j,
                          nxt * n_lat + j + 1, i * n_lat + j + 1])
    return verts, np.asarray(faces)


def _kernel_cov(dens, dim):
    """The kernel's covariance in the drawn coordinates.

    Spherical in absolute mode. In relative mode the quadratic form is
    ``I - J/r`` over the ``r - 1`` drawn coordinates, whose inverse is
    ``I + J``, so the kernel is elongated by ``sqrt(r)`` along the
    all-ones diagonal and circular across it.
    """
    sigma = float(np.atleast_1d(dens.sigma)[0])
    if bool(np.atleast_1d(dens.is_rel)[0]):
        return sigma ** 2 * (np.eye(dim) + np.ones((dim, dim)))
    return sigma ** 2 * np.eye(dim)


def plot_maet_3d(dens, ax=None, k_sigma=1.0, colour_gamma=0.5,
                 cmap='hot', view=(20.0, -70.0), dark=True):
    """Draw a three-dimensional density as its kernels.

    One ellipsoid per tuple centre, shaped by the kernel's covariance
    and coloured by the density there. No grid is evaluated, so the cost
    is the number of centres rather than the volume, and the result is
    ordinary geometry that rotates and occludes correctly.

    It draws the model rather than the density: where kernels overlap,
    what is shown is the kernels and not the sum they make. Colouring by
    the density at each centre keeps the heights visible, which a level
    surface of the summed density would not.

    Parameters
    ----------
    dens :
        Density from :func:`build_maet`, of one attribute and three
        drawn dimensions.
    ax :
        Target axes, which must have been made with
        ``projection='3d'``. Default: a new figure.
    k_sigma : float
        The level surface drawn, in standard deviations. Larger shows
        more of each kernel and hides more behind it.
    colour_gamma : float
        Display curve on the colour.
    cmap :
        Colour map, by name or as a matplotlib colormap.
    view : tuple
        Elevation and azimuth in degrees.
    dark : bool
        Dark panes for the cube, the ground a glow is read against.

    Returns
    -------
    The ``Poly3DCollection`` drawn.

    Examples
    --------
    >>> p_attr = [np.array([0, 200, 400, 500, 700, 900, 1100.])[:, None]]
    >>> specs = mpt.flat_specs(p_attr, r=4, rel=True, exch=True)
    >>> dens = mpt.build_maet(p_attr, None, specs=specs, sigma=[15.0],
    ...                       is_per=[True], period=[1200.0])
    >>> mpt.plot_maet_3d(dens)                        # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm as _cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            'plot_maet_3d needs matplotlib, which the toolbox does not '
            'require: pip install matplotlib.') from exc

    centres = np.asarray(maet_centres(dens)[0], dtype=float)
    dim = centres.shape[0]
    if dim != 3:
        raise ValueError('A three-dimensional plot needs a density of '
                         f'three drawn dimensions; this one has {dim}.')

    is_per = bool(np.atleast_1d(dens.is_per)[0])
    period = float(np.atleast_1d(dens.period)[0])
    if is_per:
        centres = np.mod(centres, period)
    cov = _kernel_cov(dens, dim)
    chol = np.linalg.cholesky(cov) * k_sigma
    peaks = np.asarray(eval_maet(dens, centres, 'none',
                                 verbose=False)).ravel()

    unit_v, unit_f = _unit_sphere()
    n_vert = unit_v.shape[0]
    reach = k_sigma * np.sqrt(np.diag(cov))
    # A centre whose kernel crosses a face of a periodic cube is drawn
    # again on the other side, so that the wrap is cut by the face
    # rather than missing from it.
    blocks, values = [], []
    for j in range(centres.shape[1]):
        offsets = [[0.0] for _ in range(dim)]
        if is_per:
            for i in range(dim):
                if centres[i, j] + reach[i] > period:
                    offsets[i].append(-period)
                if centres[i, j] - reach[i] < 0.0:
                    offsets[i].append(period)
        for dx in offsets[0]:
            for dy in offsets[1]:
                for dz in offsets[2]:
                    shift = centres[:, j] + np.array([dx, dy, dz])
                    blocks.append(unit_v @ chol.T + shift)
                    values.append(peaks[j])
    verts = np.concatenate(blocks)
    faces = np.concatenate([unit_f + i * n_vert
                            for i in range(len(blocks))])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    colours = _cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    shade = (np.repeat(values, unit_f.shape[0]) / peaks.max()) ** colour_gamma
    base = colours(shade)[:, :3]
    # Flat shading from a fixed direction, so that turning the cube
    # turns the object under a steady illumination.
    poly = verts[faces]
    normal = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.4, -0.7, 0.9])
    light /= np.linalg.norm(light)
    lit = np.clip(0.32 + 0.68 * np.abs(normal @ light), 0.0, 1.0)

    mesh = Poly3DCollection(poly)
    mesh.set_facecolor(np.clip(base * lit[:, None], 0.0, 1.0))
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)

    if is_per:
        lims = (0.0, period)
    else:
        pad = 3.0 * np.sqrt(np.diag(cov)).max() * max(k_sigma, 1.0)
        lims = (centres.min() - pad, centres.max() + pad)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_zlim(*lims)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    if dark:
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.06, 0.06, 0.06, 1.0))
    return mesh


def plot_maet_3d_points(dens, ax=None, step=None, thresh_frac=0.02,
                        marker_size=10.0, colour_gamma=0.5, cmap='hot',
                        view=(20.0, -70.0), dark=True):
    """Draw a three-dimensional density as its sampled values.

    One translucent mark per grid node above a threshold, coloured and
    made translucent by the value there. It shows the material between
    the peaks, which the kernels cannot, and costs the evaluation and
    the marks alone.

    The opacity is the value itself rather than an extinction per unit
    of path, so this is a translucent cloud and not a volume rendering:
    what a ray accumulates along its length is not what the picture
    shows. For that, MATLAB's ``plotMaet3d(dens, 'method', 'slices')``
    is the method; matplotlib has no counterpart.

    Parameters
    ----------
    dens :
        Density from :func:`build_maet`, of one attribute and three
        drawn dimensions.
    ax :
        Target axes, made with ``projection='3d'``. Default: a new
        figure.
    step : float, optional
        Spacing of the evaluated volume, in the density's own units.
        Default: the range over 120. Memory goes as its cube.
    thresh_frac : float
        Nodes below this fraction of the largest value are left
        undrawn.
    marker_size : float
        The mark's area in square points.
    colour_gamma : float
        Display curve on the colour, which is also the opacity.
    cmap :
        Colour map, by name or as a matplotlib colormap.
    view : tuple
        Elevation and azimuth in degrees.
    dark : bool
        Dark panes for the cube.

    Returns
    -------
    The ``Path3DCollection`` drawn.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm as _cm
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            'plot_maet_3d_points needs matplotlib, which the toolbox '
            'does not require: pip install matplotlib.') from exc

    centres = np.asarray(maet_centres(dens)[0], dtype=float)
    dim = centres.shape[0]
    if dim != 3:
        raise ValueError('A three-dimensional plot needs a density of '
                         f'three drawn dimensions; this one has {dim}.')
    is_per = bool(np.atleast_1d(dens.is_per)[0])
    period = float(np.atleast_1d(dens.period)[0])
    if is_per:
        lims = (0.0, period)
    else:
        pad = 3.0 * np.sqrt(np.diag(_kernel_cov(dens, dim))).max()
        lims = (float(centres.min()) - pad, float(centres.max()) + pad)

    if step is None:
        step = (lims[1] - lims[0]) / 120.0
    n = max(1, int(round((lims[1] - lims[0]) / step)))
    g = np.linspace(lims[0], lims[1], n + 1)
    # A slab of planes at a time: the whole cube of query points is a
    # large array to hold before any of it is used.
    kept_pts, kept_vals, peak = [], [], 0.0
    for first in range(0, g.size, 8):
        slab = g[first:first + 8]
        ga, gb, gc = np.meshgrid(slab, g, g, indexing='ij')
        pts = np.vstack([ga.ravel(), gb.ravel(), gc.ravel()])
        vals = np.asarray(eval_maet(dens, pts, 'none',
                                    verbose=False)).ravel()
        peak = max(peak, float(vals.max()))
        keep = vals > thresh_frac * peak
        kept_pts.append(pts[:, keep])
        kept_vals.append(vals[keep])
    pts = np.hstack(kept_pts)
    vals = np.concatenate(kept_vals)
    keep = vals > thresh_frac * peak
    pts, vals = pts[:, keep], vals[keep]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    colours = _cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    shade = (vals / vals.max()) ** colour_gamma
    rgba = colours(shade)
    rgba[:, 3] = shade
    handle = ax.scatter(pts[0], pts[1], pts[2], s=marker_size, c=rgba,
                        linewidths=0, depthshade=False)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_zlim(*lims)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    if dark:
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.06, 0.06, 0.06, 1.0))
    return handle
