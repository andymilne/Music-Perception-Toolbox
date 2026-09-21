"""demo_maet_plots.py

Draws the same seven pitches as a MAET under every combination of the
four parameters that define one, and as each of the methods plot_maet
offers for drawing it.

=== The four parameters ===

The ``configs`` table below sets out the combinations. What each
parameter does, and where it shows in the pictures:

  r        raises the dimensionality, since dim = r - is_rel. Going
           from r = 2 to r = 3 turns a plane into a cube. The density
           places one kernel per r-tuple, so the number of blobs goes
           as the number of tuples.
  is_rel   absolute against relative. An absolute density lives at the
           pitches themselves; a relative one lives at the intervals
           between them, is transposition-invariant, and costs a
           dimension. Its kernels are elongated along the all-ones
           diagonal, which the 'kernels' method shows directly.
  is_per   whether the space wraps. A periodic density is drawn over
           one period, and a kernel crossing a face reappears on the
           other side; a non-periodic one runs off into silence.
  is_exch  unordered against ordered. An ordered density counts each
           arrangement of a tuple separately and is unsymmetric in its
           arguments; the unordered one is its symmetrization and so is
           mirror-symmetric about the diagonal. Each configuration is
           drawn both ways, adjacent, so the symmetrization is a
           difference between neighbouring figures.

=== The drawing ===

Every plot is drawn by plot_maet, which dispatches on the density's
drawn dimensionality and offers three methods -- 'kernels', 'points',
and 'density' -- described at PLOT_METHOD below. This script makes no
picture of its own: it builds the densities, sets the options, and
frames the result.

Each parameter block below states the choice made and what the
alternatives do.

The MATLAB mirror is demo_maetPlots. It draws 'density' at three
dimensions as well, which rests on texture-mapped surfaces that
matplotlib has no counterpart for; 'points' stands in for it here.

Uses: build_maet, plot_maet.
"""

import sys

import numpy as np

import matplotlib.pyplot as plt

import mpt


# ===================================================================
#  User-editable parameters
# ===================================================================

# The multiset to draw, and its weights. The diatonic scale in cents:
# seven pitches keeps the structure legible and r = 4 quick. None
# means all weights equal; a weight vector scales each pitch's
# contribution and shows as differing blob heights.
P = [0, 200, 400, 500, 700, 900, 1100]
W = None

# Kernel width, in the same units as P. At 15 cents the semitone
# spacings of this scale are about seven sigma apart and the blobs
# resolve separately; at 50 they merge into ridges, which shows what a
# listener might confuse rather than where the tuples are.
SIGMA = 15.0

# The period for the periodic configurations, in the same units as P.
PERIOD = 1200.0

# One row per plot: (r, is_rel, is_per, is_exch). What each does is set
# out in the header; the table is ordered so that the differences are
# adjacent.
#
# Only one to three drawn dimensions can be drawn, dim = r - is_rel, so
# r runs to 3 absolute and 4 relative.
#
# Each configuration appears twice, unordered then ordered, so that the
# symmetrization is a difference between neighbouring figures. r = 1 is
# included both ways although it has one slot and so nothing to order:
# the two come out identical. Exchangeability is a statement about the
# arrangement of a tuple's elements, and a tuple of one has only the
# one.
CONFIGS = [
    (1, False, False, True), (1, False, False, False),
    (1, False, True,  True), (1, False, True,  False),
    (2, False, False, True), (2, False, False, False),
    (2, False, True,  True), (2, False, True,  False),
    (2, True,  False, True), (2, True,  False, False),
    (2, True,  True,  True), (2, True,  True,  False),
    (3, False, False, True), (3, False, False, False),
    (3, False, True,  True), (3, False, True,  False),
    (3, True,  False, True), (3, True,  False, False),
    (3, True,  True,  True), (3, True,  True,  False),
    (4, True,  False, True), (4, True,  False, False),
    (4, True,  True,  True), (4, True,  True,  False),
]

# The grid plot_maet evaluates. It takes either nodes, a count of steps
# across whatever range is drawn, or step, a spacing in the units of P
# -- the same request said two ways.
#
# The number that matters is the grid measured against sigma rather
# than against the range: a blob is a few sigma across, so a grid
# coarser than sigma steps straight over it and the density looks as
# though it has peaks missing rather than blurred. Roughly one sample
# per sigma is the least that shows the shape.
#
# Cost goes as the count to the power of the dimensionality. The
# 'kernels' method evaluates no grid and ignores all of this. None
# leaves the choice to plot_maet, which asks for 1200 steps at one and
# two dimensions and 120 at three.
NODES_1D = 1200    # steps across the range, so periodic and
NODES_2D = 1200    # non-periodic are sampled alike
STEP_3D = 10.0     # cents per step, so the wider non-periodic cube
                   # is not left at 20 cents and blocky

# Periodic configurations are always drawn over [0, PERIOD]. The rest
# are drawn over the range set here, rather than over the extent
# plot_maet would choose from the data, so that every configuration
# shares one frame and can be read against the others. Centred on zero
# because a relative density is symmetric about the unison and this
# shows an interval beside its inversion; absolute densities take the
# same range so the two kinds are comparable.
AX_MIN_NONPER = -1200.0
AX_MAX_NONPER = 1200.0

# Where the opacity curve starts, at zero density, for the
# two-dimensional images. At 0 the empty parts are fully transparent,
# which reads well against a plain background; 1 turns the fading off
# and draws the image opaque, which is matplotlib's own look.
ALPHA_FLOOR_2D = 0.0

# Passed straight to plot_maet.
#   'kernels' - the model rather than the density: one object per tuple
#               centre, an ellipsoid, an ellipse, or a curve. Shows
#               where the kernels are and what shape they have, which
#               is where the elongation of a relative kernel becomes
#               visible. Evaluates no grid.
#
#               At one dimension each curve is one tuple's own term,
#               its width the kernel's and its height that tuple's
#               weight, so the curves sum to the density. Colour is the
#               density at that centre -- the total, neighbours
#               included -- so two curves of equal height differ in
#               colour where kernels crowd, and a peak can be seen to
#               be one kernel or several.
#   'points'  - the density sampled: one translucent mark per grid node
#               above a threshold. Three dimensions only.
#   'density' - the density itself: a line at one dimension and a
#               translucent image at two. Not available at three,
#               where MATLAB's mirror draws a stack of textured planes
#               and matplotlib has no counterpart.
#
# 'density' is the default here because the demo is about what the four
# parameters do to the density, and the kernels are a step behind that.
# It falls back to 'points' at three dimensions, that being the only
# method matplotlib can show a volume's interior with.
PLOT_METHOD = 'density'

# Draw the two-dimensional density as a surface in relief rather than
# as an image. An image is read from directly above and cannot be
# tilted; a surface can, and the height says what the opacity says from
# above -- so ALPHA_FLOOR_2D above zero usually suits it, keeping the
# sheet visible so the blobs rise from it rather than floating.
#
# A surface is drawn as one quad per grid cell, in Python, where an
# image is one array handed to the renderer. NODES_2D_RELIEF is
# therefore coarser than NODES_2D: at 1200 it would be 1.4 million
# quads and some forty seconds a frame. At 240 the spacing is 5 cents
# against a sigma of 15, which resolves the blobs, and a frame costs
# about 1.5 seconds -- slow to turn, but the surface opens face on and
# is tilted from there rather than turned freely.
RELIEF_2D = False
NODES_2D_RELIEF = 240

# Collect the figures as tabs of one window rather than opening a
# window each, as the MATLAB mirror's docked figures do, so that one
# close ends the run. It needs a Qt binding (pip install PyQt5, or
# PySide6); without one the figures open separately and a note says so.
TAB_FIGURES = True

# Ticks fall on one of these intervals, the smallest that is not
# crowded, so that every figure is read the same way.
TICK_STEPS = (100, 200, 300, 400, 600)


# ===================================================================
#  Helpers
# ===================================================================

def tick_step(avail_pts, span, min_pts):
    """The smallest tidy interval whose labels still have room."""
    for candidate in TICK_STEPS:
        if avail_pts / (span / candidate + 1) >= min_pts:
            return candidate
    return TICK_STEPS[-1]


def set_demo_ticks(ax, lims, dim):
    """Ticks at the smallest tidy interval that is not crowded.

    One interval serves every drawn axis, the largest any of them
    needs. The axes cover the same range as each other, so ticking them
    differently would make a square plot read as though they did not.
    """
    fig = ax.get_figure()
    box = ax.get_window_extent()
    scale = 72.0 / fig.dpi
    span = lims[1] - lims[0]
    if dim == 1:
        avail, room = (box.width * scale,), (50.0,)
    elif dim == 2:
        avail, room = (box.width * scale, box.height * scale), (50.0, 30.0)
    else:
        # No orientation projects the cube wider than its space
        # diagonal, and all three axes share the box.
        side = min(box.width, box.height) * scale / np.sqrt(3.0)
        avail, room = (side, side, side), (50.0, 50.0, 50.0)

    step = max(tick_step(a, span, r) for a, r in zip(avail, room))
    ticks = np.arange(lims[0], lims[1] + step / 2.0, step)
    setters = [ax.set_xticks, ax.set_yticks]
    if dim == 3:
        setters.append(ax.set_zticks)
    for setter in setters[:max(1, min(dim, 3))]:
        setter(ticks)


class Figures:
    """Where the figures go: tabs of one window, or a window each.

    matplotlib has no tabbed figure manager of its own. With a Qt
    binding installed this builds one, owning the figures outright
    rather than taking them from pyplot -- a canvas reparented out of
    pyplot leaves pyplot holding a figure it can no longer show, so
    plt.close would empty the tabs without closing the window. Without
    Qt the figures are pyplot's, one window each, which is
    matplotlib's own behaviour.

    Either way, closing the window or windows ends the run, and escape
    closes the lot from any figure.
    """

    def __init__(self, tabbed):
        self.qt = None
        if not tabbed:
            print('Figures: separate windows (TAB_FIGURES is False).')
            return
        try:
            self.qt = self._find_qt()
            print('Figures: tabs of one window.')
        except Exception as exc:
            # Say which import failed, and how: 'Qt is missing' and
            # 'Qt is there but this matplotlib names it differently'
            # want different fixes.
            print(f'Figures: separate windows -- tabs need Qt, and '
                  f'importing it raised {type(exc).__name__}: {exc}. '
                  f'pip install PyQt5 into {sys.executable} if it is '
                  'not there.')
            return
        if self.qt is not None:
            QtWidgets = self.qt[1]
            self.app = (QtWidgets.QApplication.instance()
                        or QtWidgets.QApplication([]))
            self.window = QtWidgets.QMainWindow()
            self.window.setWindowTitle('demo_maet_plots')
            self.tabs = QtWidgets.QTabWidget()
            self.window.setCentralWidget(self.tabs)
            self.window.resize(900, 780)

    @staticmethod
    def _find_qt():
        """The Qt widgets and matplotlib's Qt canvas, however they are
        named here.

        matplotlib's qt_compat picks a binding for itself and raises
        when it finds none; a binding imported directly is the fallback
        for the case where it looks in the wrong place.
        """
        try:
            from matplotlib.backends.qt_compat import QtCore, QtWidgets
        except Exception:
            from PyQt5 import QtCore, QtWidgets          # noqa: F401
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
        except ImportError:
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
        return QtCore, QtWidgets, FigureCanvasQTAgg, NavigationToolbar2QT

    def new(self, label, three_d, figsize=(7.5, 6.0)):
        """A figure and its axes, added to the window as it is made."""
        if self.qt is None:
            fig = plt.figure(figsize=figsize)
        else:
            from matplotlib.figure import Figure
            QtCore, QtWidgets, FigureCanvas, NavToolbar = self.qt
            fig = Figure(figsize=figsize)
            canvas = FigureCanvas(fig)
            page = QtWidgets.QWidget()
            box = QtWidgets.QVBoxLayout(page)
            box.setContentsMargins(0, 0, 0, 0)
            toolbar = NavToolbar(canvas, page)
            # On a high-resolution display the toolbar sizes itself to
            # the raw pixmap rather than to the pixmap's device ratio,
            # so the icons come out at twice their intended size unless
            # the size is set.
            toolbar.setIconSize(QtCore.QSize(20, 20))
            box.addWidget(toolbar)
            box.addWidget(canvas)
            self.tabs.addTab(page, label)
        ax = (fig.add_subplot(projection='3d') if three_d
              else fig.add_subplot())
        # Escape closes everything from any figure. A script blocked in
        # show() leaves no prompt to close from, and matplotlib's own
        # 'q' closes one figure at a time.
        fig.canvas.mpl_connect(
            'key_press_event',
            lambda evt: self.close_all() if evt.key == 'escape' else None)
        return fig, ax

    def close_all(self):
        """Close every figure, however they are held."""
        if self.qt is None:
            plt.close('all')
        else:
            self.window.close()

    def run(self):
        if self.qt is None:
            plt.show()
        else:
            self.window.show()
            self.app.exec() if hasattr(self.app, 'exec') else self.app.exec_()


def method_for(dim):
    """The method to draw this dimensionality with.

    'density' has no three-dimensional form here, so the default falls
    back to 'points', which is what matplotlib can show a volume's
    interior with.
    """
    if PLOT_METHOD == 'density' and dim == 3:
        return 'points'
    return PLOT_METHOD


def grid_args(dim, lims, relief=False):
    """The grid request, and the node count it comes to."""
    if dim == 1:
        nodes = NODES_1D
    elif dim == 2:
        nodes = NODES_2D_RELIEF if relief else NODES_2D
    else:
        nodes = None
    if dim == 3:
        if STEP_3D is None:
            return {}, 120
        return {'step': STEP_3D}, int(round((lims[1] - lims[0]) / STEP_3D))
    if nodes is None:
        return {}, 1200
    return {'nodes': nodes}, int(nodes)


# ===================================================================
#  Main loop
# ===================================================================

def main():
    # Anything left over from a previous run, as the MATLAB mirror's
    # close all does.
    plt.close('all')

    labels = {True: ('relative', 'Interval'), False: ('absolute', 'Pitch')}
    figures = Figures(TAB_FIGURES)

    print('\n--- Plot summary ---')
    for ci, (r, is_rel, is_per, is_exch) in enumerate(CONFIGS, start=1):
        dim = r - int(is_rel)
        if r > len(P) or (is_rel and r < 2) or dim > 3:
            print(f'  Config {ci}: r={r}, dim={dim} -- not drawn')
            continue

        lims = ((0.0, PERIOD) if is_per
                else (AX_MIN_NONPER, AX_MAX_NONPER))
        method = method_for(dim)
        relief = RELIEF_2D and dim == 2 and method == 'density'
        args, nodes = grid_args(dim, lims, relief)
        mode_str, ax_label = labels[bool(is_rel)]
        per_str = 'periodic' if is_per else 'non-periodic'
        ord_str = 'unordered' if is_exch else 'ordered'
        print(f'  Config {ci}: r={r}, {mode_str}, {per_str}, {ord_str}, '
              f'dim={dim}, nodes={nodes}, method={method}')

        dens = mpt.build_maet(P, W, SIGMA, r, is_rel, is_per, PERIOD,
                              is_exch, verbose=False)

        fig, ax = figures.new(f'{ci}: r={r} dim={dim}',
                              three_d=(dim == 3 or relief))
        extra = ({'alpha_floor': ALPHA_FLOOR_2D, 'relief': relief}
                 if dim == 2 and method == 'density' else {})
        mpt.plot_maet(dens, method=method, ax=ax, limits=lims,
                      **args, **extra)

        ax.set_xlabel(f'{ax_label} 1')
        if dim == 1:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel(f'{ax_label} 2')
        if dim == 3:
            ax.set_zlabel(f'{ax_label} 3')
        elif relief:
            ax.set_zlabel('Density')
        ax.set_title(f'r = {r}, {mode_str}, {per_str}, {ord_str}, '
                     f'$\\sigma$ = {SIGMA:g} — {method}')
        fig.canvas.draw()
        # Two axes to tick even in relief: the height is the
        # density's own scale, and ticking it over the drawn range
        # would stretch the surface flat.
        set_demo_ticks(ax, lims, dim)

    print('All plots complete. Close the window to finish.')
    figures.run()


if __name__ == '__main__':
    main()
