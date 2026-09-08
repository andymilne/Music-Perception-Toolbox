"""test_show_pre_maet.py -- the pre-MAET table renderer.

The rendering is a fixed string, so most assertions here are on exact
output: the markdown table is cross-language (its column widths must be
identical in Python and MATLAB, which is why it is plain ASCII), and the
LaTeX table is pasted into a manuscript. Anything that changes these
strings changes a published artefact, so the tests state them in full
rather than probing for substrings.
"""
import numpy as np
import pytest

import mpt


def _cells(out, row=-1):
    """The stripped cells of one body row, past the attribute stub."""
    return [c.strip() for c in out.splitlines()[row].split("|")[2:-1]]


def _p():
    return [np.array([[69.0, 69, 69, 71, 67, 66, 64]]),
            np.array([[1.0, 2, 3, 4, 5, 6, 7]])]


def _w():
    m = np.array([[1.0, 0.5, 0.75, 0.5, 1.0, 0.5, 0.75]])
    return [m.copy(), m.copy()]


KW = dict(names=['pitch', 'time'], sigma=[0.5, 0.25],
          is_per=[True, False], period=[12.0, 0.0], verbose=False)


class TestMarkdown:

    def test_flat_table(self):
        out = mpt.show_pre_maet(_p(), _w(), **KW)
        assert out.splitlines() == [
            "| attribute                                               "
            "| n = 1  |  n = 2   |   n = 3   |  n = 4   | n = 5  "
            "|  n = 6   |   n = 7   |",
            "|:--------------------------------------------------------"
            "|:------:|:--------:|:---------:|:--------:|:------:"
            "|:--------:|:---------:|",
            "| pitch: sigma = 0.5, r = 1, [rel] = 0, [per] = 1, P = 12 "
            "| 69^(1) | 69^(0.5) | 69^(0.75) | 71^(0.5) | 67^(1) "
            "| 66^(0.5) | 64^(0.75) |",
            "| time: sigma = 0.25, r = 1, [rel], [per] = 0             "
            "| 1^(1)  | 2^(0.5)  | 3^(0.75)  | 4^(0.5)  | 5^(1)  "
            "| 6^(0.5)  | 7^(0.75)  |",
        ]

    def test_rows_are_equal_width(self):
        """The padding must survive into MATLAB, whose char is bytewise."""
        for kwargs in ({}, dict(max_events=4), dict(max_elements=3)):
            out = mpt.show_pre_maet(_p(), _w(), **KW, **kwargs)
            widths = {len(line) for line in out.splitlines()}
            assert len(widths) == 1

    def test_is_ascii(self):
        """A multi-byte glyph would count as several characters in
        Octave and one in MATLAB, so the column widths would diverge."""
        out = mpt.show_pre_maet(_p(), _w(), max_events=4, **KW)
        out.encode("ascii")

    def test_uniform_weights_are_not_shown(self):
        out = mpt.show_pre_maet(_p(), None, **KW)
        assert "^(" not in out
        out = mpt.show_pre_maet(_p(), [np.ones((1, 7))] * 2, **KW)
        assert "^(" not in out

    def test_weights_forced_and_suppressed(self):
        assert "^(1)" in mpt.show_pre_maet(
            _p(), [np.ones((1, 7))] * 2, weights=True, **KW)
        assert "^(" not in mpt.show_pre_maet(
            _p(), _w(), weights=False, **KW)

    def test_event_elision(self):
        out = mpt.show_pre_maet(_p(), _w(), max_events=4, **KW)
        assert "n = 1" in out and "n = 7" in out
        assert "n = 4" not in out
        assert " ... " in out

    def test_element_elision(self):
        out = mpt.show_pre_maet([np.arange(60.0, 72.0).reshape(12, 1)],
                                names=['pitch'], sigma=0.1,
                                max_elements=5, verbose=False)
        assert "{60, 61, 62, 63, ...}" in out


class TestCells:

    def test_unordered_takes_braces_ordered_takes_parens(self):
        P = [np.array([[36.0], [55], [60], [64]])]
        sym = mpt.show_pre_maet(P, specs=[{'r': 2, 'rel': False,
                                           'sym': True}],
                                names=['pitch'], sigma=0.15, verbose=False)
        ord_ = mpt.show_pre_maet(P, specs=[{'r': 2, 'rel': False,
                                            'sym': False}],
                                 names=['pitch'], sigma=0.15, verbose=False)
        assert "{36, 55, 60, 64}" in sym
        assert "(36, 55, 60, 64)" in ord_

    def test_single_element_is_bare_at_top_level(self):
        out = mpt.show_pre_maet([np.array([[69.0, 66, 64]])],
                                names=['pitch'], sigma=0.5, verbose=False)
        assert _cells(out) == ["69", "66", "64"]
        assert "{69}" not in out

    def test_nested_brackets_outermost_first(self):
        """The outermost level takes the outermost bracket, as the
        article writes it: an ordered run of unordered chords."""
        P = np.array([[36.0, 43], [55, 55], [60, 59], [64, 62]])
        pb, wb, sb = mpt.unpack_pre_maet(mpt.bind_events([P], None, [2]))
        out = mpt.show_pre_maet(pb, wb, sb, names=['pitch'], sigma=0.15,
                                is_per=True, period=12.0, verbose=False)
        assert "({36, 55, 60, 64}, {43, 55, 59, 62})" in out

    def test_nested_keeps_brackets_at_one_element(self):
        """An inner level of a nest stays bracketed, so the level is
        visible even where it holds a single value."""
        pb, wb, sb = mpt.unpack_pre_maet(mpt.bind_events([np.array([[69.0, 66, 64]])], None, [2]))
        out = mpt.show_pre_maet(pb, wb, sb, names=['pitch'], sigma=0.5,
                                verbose=False)
        assert "({69}, {66})" in out

    def test_nan_is_absent_not_an_element(self):
        P = [np.array([[60.0, 60], [64, 64], [67, np.nan]])]
        out = mpt.show_pre_maet(P, specs=[{'r': 1, 'rel': False,
                                           'sym': True}],
                                names=['pitch'], sigma=0.5, verbose=False)
        assert "{60, 64, 67}" in out and "{60, 64}" in out


class TestNumbers:

    def test_rounded_away_value_is_not_shown_as_zero(self):
        """A table is read for which entries vanish; a rounded-away tail
        is not one of them."""
        out = mpt.show_pre_maet([np.array([[1.0, 2.0]])],
                                [np.array([[1.0, 3.7e-6]])],
                                names=['x'], sigma=1.0, decimals=4,
                                verbose=False)
        assert "^(0)" not in out
        assert "e-06" in out

    def test_floating_point_residue_is_zero(self):
        """Below the dirt floor the value is residue, not a small
        number: a simplex vertex coordinate must not read -1.96e-17."""
        out = mpt.show_pre_maet([np.array([[0.0, -1.96e-17]])],
                                names=['x'], sigma=1.0, verbose=False)
        assert "e-17" not in out
        assert _cells(out) == ["0", "0"]

    def test_exact_zero_is_zero(self):
        out = mpt.show_pre_maet([np.array([[0.0, 1.0]])], names=['x'],
                                sigma=1.0, verbose=False)
        assert _cells(out) == ["0", "1"]


class TestStub:

    def test_flags_collapse_when_both_are_scalar_zero(self):
        out = mpt.show_pre_maet([np.array([[1.0]])], names=['x'],
                                sigma=1.0, verbose=False)
        assert "[rel], [per] = 0" in out

    def test_period_shown_only_when_periodic(self):
        out = mpt.show_pre_maet(_p(), _w(), **KW)
        assert "[per] = 1, P = 12" in out
        assert "P = 0" not in out

    def test_kernel_covariance_named_by_shape(self):
        """An attribute may carry a covariance where the others carry a
        width; the row states its shape rather than printing a matrix."""
        C = np.eye(3) * 0.04
        pb, wb, sb = mpt.unpack_pre_maet(mpt.bind_events([np.array([[1.0, 2, 3, 4]])], None, [3]))
        out = mpt.show_pre_maet(pb, wb, sb, names=['trigram'],
                                sigma=[C], verbose=False)
        assert "sigma = 3x3 covariance" in out

    def test_names_default_to_specs_then_to_index(self):
        out = mpt.show_pre_maet([np.array([[1.0]]), np.array([[2.0]])],
                                specs=[{'r': 1, 'rel': False, 'sym': True,
                                        'name': 'pitch'},
                                       {'r': 1, 'rel': False, 'sym': True}],
                                sigma=1.0, verbose=False)
        assert "| pitch: " in out and "| a_2: " in out


class TestDensityInput:

    def test_density_and_raw_agree(self):
        """A built density carries every field the raw triple supplies,
        so the two inputs must render the same table."""
        dens = mpt.build_exp_tens(_p(), _w(), [0.5, 0.25], [1, 1],
                                  [False, False], [True, False],
                                  [12.0, 0.0], verbose=False)
        from_dens = mpt.show_pre_maet(dens, names=['pitch', 'time'],
                                      verbose=False)
        assert from_dens == mpt.show_pre_maet(_p(), _w(), **KW)


class TestLatex:

    def test_latex_table(self):
        out = mpt.show_pre_maet(
            [np.array([[36.0], [55], [60], [64]])], None,
            [{'r': 2, 'rel': False, 'sym': True}], names=['pitch'],
            sigma=0.15, is_per=True, period=12.0, format='latex',
            caption='A caption.', label='tab:x', verbose=False)
        assert out.splitlines() == [
            r"\begin{table}[]",
            r"\centering",
            r"\footnotesize",
            r"\caption{A caption.}",
            r"\label{tab:x}",
            r"\smallskip",
            r"\begin{tabular}{@{}cc@{}}",
            r"\toprule",
            r"attribute & $n = 1$ \\",
            r"\midrule",
            r"$\begin{array}{@{}c@{}} \text{pitch} \\ \sigma = 0.15, "
            r"r = 2, {[\mathrm{rel}]} = 0, {[\mathrm{per}]} = 1, P = 12 "
            r"\end{array}$ & $\{36, 55, 60, 64\}$ \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

    def test_latex_has_no_thin_space_after_commas(self):
        out = mpt.show_pre_maet(_p(), _w(), format='latex', **KW)
        assert r",\ " not in out

    def test_latex_elision_and_caption_optional(self):
        out = mpt.show_pre_maet(_p(), _w(), format='latex', max_events=4,
                                **KW)
        assert r"$\cdots$" in out
        assert r"\caption" not in out and r"\label" not in out


class TestErrors:

    def test_bad_format(self):
        with pytest.raises(ValueError, match="format must be"):
            mpt.show_pre_maet(_p(), verbose=False, format='html')

    def test_ragged_passage(self):
        with pytest.raises(ValueError, match="every attribute must span"):
            mpt.show_pre_maet([np.array([[1.0, 2.0]]), np.array([[1.0]])],
                              sigma=1.0, verbose=False)

    def test_weight_attribute_count(self):
        with pytest.raises(ValueError, match="w has 1 attributes"):
            mpt.show_pre_maet(_p(), [np.ones((1, 7))], verbose=False)

    def test_returns_string_when_silent(self):
        out = mpt.show_pre_maet(_p(), _w(), **KW)
        assert isinstance(out, str) and out
