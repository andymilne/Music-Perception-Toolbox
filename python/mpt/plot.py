"""Drawing an expectation tensor density of one, two, or three drawn
dimensions.

Mirror of MATLAB ``plotMaet``: the same three methods -- ``kernels``,
``points``, and ``density`` -- dispatching on the same rule. One
combination is missing. ``density`` at three dimensions rests on
texture-mapped surfaces, which matplotlib has no counterpart for, and
``points`` is what stands in for it there.

Appearance follows matplotlib's defaults rather than MATLAB's: no
colour-map brightening, light panes unless asked otherwise, and
matplotlib's own camera until a view is given.

matplotlib is imported when a plot is drawn rather than when the
toolbox is imported, so that the package's own dependencies stay
``numpy`` and ``scipy``.
"""

from __future__ import annotations

import numpy as np

from ._defaults import get_default, set_default
from .tensor import eval_maet, maet_centres


__all__ = ["plot_maet"]

_METHODS = ("kernels", "points", "density")


def _need_pyplot():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "plot_maet needs matplotlib, which the toolbox does not "
            "require: pip install matplotlib.") from exc


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


def _unit_circle(n=96):
    """Points on the unit circle, built here for the same reason."""
    t = np.arange(n) * 2.0 * np.pi / n
    return np.column_stack([np.cos(t), np.sin(t)])


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


def _wrap_shifts(centre, reach, is_per, period, dim):
    """The copies of a centre a periodic box calls for: zero always,
    and one period either way for each coordinate whose kernel reaches
    past a face."""
    shifts = []
    for i in range(dim):
        s = [0.0]
        if is_per:
            r = float(np.atleast_1d(reach)[min(i, np.atleast_1d(reach).size - 1)])
            if centre[i] + r > period:
                s.append(-period)
            if centre[i] - r < 0.0:
                s.append(period)
        shifts.append(s)
    return shifts


def _limits(centres, cov, is_per, period, k_sigma):
    """The box drawn in: one period on a periodic attribute, otherwise
    the centres' own extent with room for the kernels around them."""
    if is_per:
        return (0.0, float(period))
    pad = 3.0 * float(np.sqrt(np.diag(cov)).max()) * max(k_sigma, 1.0)
    return (float(centres.min()) - pad, float(centres.max()) + pad)


class _quiet:
    """Silence the dispatch messages for the duration.

    Which path an evaluation takes is the toolbox's business rather
    than the picture's, and a grid is evaluated a slab at a time, so
    routing the same way for every slab would announce itself once per
    slab.
    """

    def __enter__(self):
        self._was = get_default('show_hints')
        set_default(show_hints=False)
        return self

    def __exit__(self, *exc):
        set_default(show_hints=self._was)
        return False


def _volume(dens, lims, step, dim):
    """The density on a grid of ``dim`` dimensions."""
    n = max(1, int(round((lims[1] - lims[0]) / step)))
    g = np.linspace(lims[0], lims[1], n + 1)
    with _quiet():
        if dim == 1:
            v = np.asarray(eval_maet(dens, g[None, :], 'none',
                                     verbose=False)).ravel()
            return v, g
        if dim == 2:
            ga, gb = np.meshgrid(g, g, indexing='ij')
            v = np.asarray(eval_maet(dens,
                                     np.vstack([ga.ravel(), gb.ravel()]),
                                     'none', verbose=False)).ravel()
            return v.reshape(ga.shape), g
    # A slab at a time: the whole cube of query points is a large array
    # to hold at once.
    out = np.empty((g.size, g.size, g.size))
    with _quiet():
        for first in range(0, g.size, 8):
            slab = g[first:first + 8]
            ga, gb, gc = np.meshgrid(slab, g, g, indexing='ij')
            v = np.asarray(eval_maet(dens,
                                     np.vstack([ga.ravel(), gb.ravel(),
                                                gc.ravel()]),
                                     'none', verbose=False)).ravel()
            out[first:first + slab.size] = v.reshape(ga.shape)
    return out, g


def _alpha_curve(rel, alpha_peak, alpha_floor, alpha_gamma):
    """Opacity from the density: the curve runs from ``alpha_floor`` at
    nothing to ``alpha_peak`` at the density's own peak, so a floor of
    1 turns the scaling off and leaves the picture opaque."""
    return alpha_floor + (alpha_peak - alpha_floor) * rel ** alpha_gamma


def _frame(ax, lims, dim, view, dark, relief=False):
    """The box the density is drawn in. Square at two and three
    dimensions, the drawn coordinates covering the same range as each
    other; at one the vertical axis is the density's own, and on a
    relief surface so is the third."""
    ax.set_xlim(*lims)
    if dim == 1:
        if dark:
            ax.set_facecolor((0.06, 0.06, 0.06))
        return
    ax.set_ylim(*lims)
    if dim == 2 and not relief:
        ax.set_aspect('equal')
        if dark:
            ax.set_facecolor((0.06, 0.06, 0.06))
        return
    if relief:
        # The drawn coordinates are square to each other; the height is
        # the density's own scale and is left to the axes, which is
        # what keeps a surface of small values from being stretched
        # over the drawn range.
        ax.set_box_aspect((1, 1, 0.6))
        # Face on by default, looking down the height axis, so that a
        # relief surface opens as the image it is an alternative to and
        # is tilted from there.
        ax.view_init(elev=90, azim=-90) if view is None else \
            ax.view_init(elev=view[0], azim=view[1])
        if dark:
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.set_pane_color((0.06, 0.06, 0.06, 1.0))
        return
    ax.set_zlim(*lims)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    if dark:
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.06, 0.06, 0.06, 1.0))


def _colours(cmap):
    """The colour map as a callable, matplotlib's default when none is
    named."""
    import matplotlib as mpl
    if cmap is None:
        return mpl.colormaps[mpl.rcParams['image.cmap']]
    if isinstance(cmap, str):
        return mpl.colormaps[cmap]
    return cmap


def _new_axes(plt, dim):
    fig = plt.figure()
    if dim == 3:
        return fig.add_subplot(projection='3d')
    return fig.add_subplot()


# --------------------------------------------------------------- kernels

def _kernels_1d(ax, dens, centres, cov, is_per, period, colour_gamma, cmap):
    """One curve per centre, summing to the density.

    The density under ``normalize='none'`` is
    ``sum_j w_j exp(-Q(c_j - x) / 2 sigma^2)``, so each tuple's own
    term is a Gaussian of the kernel's width scaled by that tuple's
    weight, and the curves drawn here add up to the line the
    ``density`` method draws. Colour is the density at the curve's
    centre, the total with every neighbour counted, so two curves of
    equal height differ in colour where kernels crowd.
    """
    var = float(cov[0, 0])
    reach = 4.0 * np.sqrt(var)
    w_j = np.asarray(dens.w_j, dtype=float).ravel()
    with _quiet():
        peaks = np.asarray(eval_maet(dens, centres, 'none',
                                     verbose=False)).ravel()
    shade = (peaks / peaks.max()) ** colour_gamma
    rgba = _colours(cmap)(shade)

    t = np.linspace(-reach, reach, 129)
    bell = np.exp(-(t ** 2) / (2.0 * var))
    lines = []
    for j in range(centres.shape[1]):
        for shift in _wrap_shifts(centres[:, j], np.array([reach]),
                                  is_per, period, 1)[0]:
            lines.extend(ax.plot(centres[0, j] + shift + t, w_j[j] * bell,
                                 color=rgba[j], linewidth=1.0))
    return lines


def _kernels_2d(ax, centres, peaks, cov, is_per, period, k_sigma,
                colour_gamma, cmap):
    """One ellipse per centre, in a single collection."""
    from matplotlib.collections import PolyCollection

    chol = np.linalg.cholesky(cov) * k_sigma
    unit = _unit_circle()
    reach = k_sigma * np.sqrt(np.diag(cov))
    polys, values = [], []
    for j in range(centres.shape[1]):
        sx, sy = _wrap_shifts(centres[:, j], reach, is_per, period, 2)
        for dx in sx:
            for dy in sy:
                polys.append(unit @ chol.T + centres[:, j] + np.array([dx, dy]))
                values.append(peaks[j])
    shade = (np.asarray(values) / peaks.max()) ** colour_gamma
    patch = PolyCollection(polys, facecolors=_colours(cmap)(shade),
                           edgecolors='none')
    ax.add_collection(patch)
    return patch


def _kernels_3d(ax, centres, peaks, cov, is_per, period, k_sigma,
                colour_gamma, cmap):
    """One ellipsoid per centre, in a single collection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    chol = np.linalg.cholesky(cov) * k_sigma
    unit_v, unit_f = _unit_sphere()
    n_vert = unit_v.shape[0]
    reach = k_sigma * np.sqrt(np.diag(cov))
    blocks, values = [], []
    for j in range(centres.shape[1]):
        sx, sy, sz = _wrap_shifts(centres[:, j], reach, is_per, period, 3)
        for dx in sx:
            for dy in sy:
                for dz in sz:
                    blocks.append(unit_v @ chol.T + centres[:, j]
                                  + np.array([dx, dy, dz]))
                    values.append(peaks[j])
    verts = np.concatenate(blocks)
    faces = np.concatenate([unit_f + i * n_vert for i in range(len(blocks))])

    shade = (np.repeat(values, unit_f.shape[0]) / peaks.max()) ** colour_gamma
    base = _colours(cmap)(shade)[:, :3]
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
    return mesh


# ---------------------------------------------------------------- points

def _points_3d(ax, dens, lims, step, thresh_frac, marker_size,
               colour_gamma, cmap, alpha_peak, alpha_floor, alpha_gamma):
    """One translucent mark per grid node above the threshold.

    matplotlib depth-sorts and blends the marks, so there is no
    counterpart here to MATLAB's marker budget: overlapping marks
    combine as their opacities describe, and the size is whatever is
    asked for.
    """
    vol, g = _volume(dens, lims, step, 3)
    ga, gb, gc = np.meshgrid(g, g, g, indexing='ij')
    keep = vol > thresh_frac * vol.max()
    vals = vol[keep]
    rel = vals / vals.max()

    rgba = _colours(cmap)(rel ** colour_gamma)
    rgba[:, 3] = _alpha_curve(rel, alpha_peak, alpha_floor, alpha_gamma)
    return ax.scatter(ga[keep], gb[keep], gc[keep], s=marker_size,
                      c=rgba, linewidths=0, depthshade=False)


# --------------------------------------------------------------- density

def _density_1d(ax, dens, lims, step, cmap):
    """The density over its grid."""
    vol, g = _volume(dens, lims, step, 1)
    colour = _colours(cmap)(0.75)
    return ax.plot(g, vol, color=colour, linewidth=1.5)[0]


def _density_2d_relief(ax, dens, lims, step, colour_gamma, cmap,
                       alpha_peak, alpha_floor, alpha_gamma):
    """The density as a surface in relief, on a three-dimensional axes.

    The same values as the image, with the density as height as well as
    colour, so that the picture can be tilted. What opacity says from
    directly above, relief says from an angle -- which is why an
    ``alpha_floor`` above zero usually suits this form: the sheet stays
    visible and the blobs rise from it rather than floating.

    The grid is drawn at its own resolution rather than decimated, so
    the surface carries one quad per grid cell. matplotlib projects and
    sorts all of them in Python on every frame, at roughly 30
    microseconds each, so a grid fine enough for an image is far too
    fine to turn: 3600 quads gives about 8 frames a second and 14400
    about 2. Choose the grid for the picture wanted.
    """
    vol, g = _volume(dens, lims, step, 2)
    ga, gb = np.meshgrid(g, g, indexing='ij')
    rel = np.maximum(vol, 0.0) / max(vol.max(), np.finfo(float).tiny)
    rgba = _colours(cmap)(rel ** colour_gamma)
    rgba[..., 3] = _alpha_curve(rel, alpha_peak, alpha_floor, alpha_gamma)
    return ax.plot_surface(ga, gb, vol, facecolors=rgba, shade=False,
                           linewidth=0, antialiased=False,
                           rcount=vol.shape[0], ccount=vol.shape[1])


def _density_2d(ax, dens, lims, step, colour_gamma, cmap,
                alpha_peak, alpha_floor, alpha_gamma):
    """The density as one translucent image.

    The opacity follows the value, so the low material fades to the
    axes rather than flooring at the colour map's low colour -- which,
    painted opaque, would become the ground the density is read
    against.
    """
    vol, _ = _volume(dens, lims, step, 2)
    rel = np.maximum(vol, 0.0) / max(vol.max(), np.finfo(float).tiny)
    alpha = _alpha_curve(rel, alpha_peak, alpha_floor, alpha_gamma)
    # The grid runs along the first axis in x; an image wants rows in y.
    return ax.imshow(vol.T, origin='lower', extent=(*lims, *lims),
                     cmap=_colours(cmap), alpha=alpha.T,
                     norm=_power_norm(colour_gamma, vol), aspect='equal',
                     interpolation='nearest')


def _power_norm(colour_gamma, vol):
    """The display curve on the colour, as a norm."""
    from matplotlib.colors import PowerNorm
    return PowerNorm(colour_gamma, vmin=0.0, vmax=float(vol.max()))


def plot_maet(dens, method='kernels', ax=None, limits=None, k_sigma=1.0,
              step=None, nodes=None, thresh_frac=0.001, marker_size=10.0,
              alpha_peak=1.0, alpha_floor=0.0, alpha_gamma=1.0,
              colour_gamma=1.0, cmap=None, view=None, dark=False,
              relief=False):
    """Draw a one-, two-, or three-dimensional expectation tensor density.

    Dispatches on the dimensionality the density carries,
    ``dim = r - is_rel``. Four or more cannot be drawn and is refused.

    Three methods, each named for what it shows rather than for the
    geometry it uses, since the geometry is what changes with the
    dimensionality:

    ``kernels``
        The model rather than the density: one object per tuple centre,
        an ellipsoid at three dimensions, an ellipse at two, a curve at
        one, coloured by the density at that centre. No grid is
        evaluated, so the cost is the number of centres rather than the
        volume. Where kernels overlap it shows the kernels and not the
        sum they make -- except at one dimension, where each curve is
        one tuple's own term, its width the kernel's and its height
        that tuple's weight, so the curves do sum to the density.
    ``points``
        The density sampled: one translucent mark per grid node above a
        threshold. Three drawn dimensions only; below that it samples
        what ``density`` already draws whole. At a fine grid it is much
        the same picture as ``density`` and costs more to draw; what
        differs is that the samples stay discrete, so the grid is
        visible rather than interpolated away.
    ``density``
        The density itself: a line at one dimension and a translucent
        image at two, or a surface in relief with ``relief=True``.
        **Not available at three**, where it rests on texture-mapped
        surfaces that matplotlib has no counterpart for; use ``points``
        there. The MATLAB mirror draws all three.

    Parameters
    ----------
    dens : MaetDensity
        Density from :func:`~mpt.build_maet`, of one attribute and one
        to three drawn dimensions.
    method : {'kernels', 'points', 'density'}
        Default ``'kernels'``.
    ax : matplotlib axes, optional
        Target axes; a new figure is made when none is given. It must
        carry a 3-D projection for a three-dimensional density, and
        for a two-dimensional one drawn with ``relief``.
    limits : (lo, hi), optional
        For every drawn axis. Default: one period for a periodic
        attribute, and otherwise the centres' own extent with room for
        the kernels around them.
    k_sigma : float
        Kernels: the level surface drawn, in standard deviations. It
        sets the outline at two and three dimensions; at one the curve
        is drawn whole and this sets only how far a periodic kernel has
        to reach to be drawn again across a face.
    step, nodes : float or int, optional
        The grid ``points`` and ``density`` evaluate, as a spacing in
        the density's own units or as a count of steps across the range
        -- the same request said two ways, so give one or the other.
        The grid has one more point than ``nodes`` along each axis.
        Default: 1200 steps at one and two dimensions, 120 at three,
        the grid being ``dim``-dimensional so that the same count per
        axis costs wildly different amounts.
    thresh_frac : float
        Points: nodes below this fraction of the largest value are left
        undrawn.
    marker_size : float
        Points: the mark's area in square points. matplotlib blends
        overlapping marks correctly, so there is no counterpart here to
        the MATLAB mirror's automatic sizing.
    alpha_peak, alpha_floor : float
        Points and density: the opacity at the density's own peak and
        where there is no density, the curve running between them. A
        floor of 1 turns the fading off and draws the picture opaque.
    alpha_gamma, colour_gamma : float
        The display curves on the opacity and on the colour. Below 1
        lifts the low material and above 1 suppresses it.
    cmap : str or Colormap, optional
        Default: matplotlib's own.
    view : (elev, azim), optional
        Three dimensions: the camera, in matplotlib's own convention.
        Default: matplotlib's own. The MATLAB mirror takes ``[azimuth
        elevation]`` as MATLAB's ``view`` does, each language following
        its own plotting library.
    dark : bool
        Dark panes, the ground a glow is read against. Default False,
        matplotlib's own; the MATLAB mirror defaults to dark.
    relief : bool
        Density at two dimensions: draw it as a surface on a
        three-dimensional axes, the density as height as well as
        colour, rather than as an image. It can then be tilted, which
        an image cannot be. An ``alpha_floor`` above zero usually suits
        this form: what opacity says from directly above, relief says
        from an angle, and a floored sheet keeps the blobs from
        floating. The MATLAB mirror draws the two-dimensional density
        as a surface always, that being how it carries opacity.

    Returns
    -------
    The artist drawn: a collection for ``kernels`` at two and three
    dimensions and a list of lines at one, a path collection for
    ``points``, and for ``density`` a line or an image.

    Examples
    --------
    >>> dens = build_maet([0, 200, 400, 500, 700, 900, 1100], None,
    ...                   15, 3, True, True, 1200)   # doctest: +SKIP
    >>> plot_maet(dens)                              # doctest: +SKIP

    The MATLAB mirror is plotMaet.

    See Also
    --------
    build_maet, eval_maet, maet_centres
    """
    plt = _need_pyplot()

    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError("method must be one of 'kernels', 'points', or "
                         f"'density'; got {method!r}.")
    if getattr(dens, 'n_attrs', 1) != 1:
        raise ValueError('plot_maet draws one attribute; this density has '
                         f'{dens.n_attrs}. Draw one at a time.')

    centres = np.asarray(maet_centres(dens)[0], dtype=float)
    dim = centres.shape[0]
    if dim < 1 or dim > 3:
        raise ValueError('One to three drawn dimensions can be drawn; this '
                         f'density has {dim}.')
    if method == 'points' and dim != 3:
        raise ValueError(
            "'points' exists because a three-dimensional density is hard "
            "to draw whole. Below that it only samples what 'density' "
            f'already draws. This density has {dim}.')
    if method == 'density' and dim == 3:
        raise ValueError(
            "'density' at three dimensions rests on texture-mapped "
            'surfaces, which matplotlib has no counterpart for; use '
            "'points'. The MATLAB mirror, plotMaet, draws it.")

    is_per = bool(np.atleast_1d(dens.is_per)[0])
    period = float(np.atleast_1d(dens.period)[0])
    if is_per:
        centres = np.mod(centres, period)
    cov = _kernel_cov(dens, dim)
    lims = (_limits(centres, cov, is_per, period, k_sigma)
            if limits is None else
            (float(min(limits)), float(max(limits))))

    # 'nodes' is the same request as 'step' in the units a grid is
    # usually thought about, so it becomes a step here and nothing
    # further down knows about it.
    if nodes is not None:
        if step is not None:
            raise ValueError("'step' and 'nodes' say the same thing two "
                             'ways, so only one of them can be given.')
        step = (lims[1] - lims[0]) / round(nodes)
    if step is None:
        step = (lims[1] - lims[0]) / (1200.0 if dim < 3 else 120.0)

    relief = bool(relief) and method == 'density' and dim == 2
    if ax is None:
        ax = _new_axes(plt, 3 if relief else dim)

    if method == 'kernels':
        if dim == 1:
            handle = _kernels_1d(ax, dens, centres, cov, is_per, period,
                                 colour_gamma, cmap)
        else:
            with _quiet():
                peaks = np.asarray(eval_maet(dens, centres, 'none',
                                             verbose=False)).ravel()
            draw = _kernels_2d if dim == 2 else _kernels_3d
            handle = draw(ax, centres, peaks, cov, is_per, period, k_sigma,
                          colour_gamma, cmap)
    elif method == 'points':
        handle = _points_3d(ax, dens, lims, step, thresh_frac, marker_size,
                            colour_gamma, cmap, alpha_peak, alpha_floor,
                            alpha_gamma)
    elif dim == 1:
        handle = _density_1d(ax, dens, lims, step, cmap)
    else:
        draw = _density_2d_relief if relief else _density_2d
        handle = draw(ax, dens, lims, step, colour_gamma, cmap,
                      alpha_peak, alpha_floor, alpha_gamma)

    _frame(ax, lims, dim, view, dark, relief)
    return handle
