"""Tests for the ``kernel_threads`` default and the threaded kernel paths.

The elementwise Gaussian work is split over contiguous spans of the
evaluation points. Every point keeps the arithmetic it has serially --
the image sum inside the wrapped kernel runs within a point, and a
query's accumulation runs over its own centres in an order set by the
bucket lattice, which is pitched on the centres alone -- so the split
changes the schedule and nothing else. What is pinned here is that
identity, across the routes that carry it, together with the resolution
rules for the default.

The bit-identity assertions pin ``kernel_chunk_bytes`` to a fixed value.
That is not a threading matter: the factory ``'auto'`` resolves against
currently available memory, so two runs of the same call can chunk
differently and sum in a different order, which already perturbs the
last bits on the relative periodic route. Fixing the budget removes
that variable so the comparison is about threading alone.

There is no MATLAB twin. MATLAB threads elementwise arithmetic in the
runtime and gives each parfor worker a single computational thread, so
it needs neither the default nor the machinery under test.
"""

import os

import numpy as np
import pytest

import mpt
from mpt._utils import (_KERNEL_THREAD_CAP, kernel_thread_count,
                        kernel_threads_resolved, split_ranges)


PERIOD = 1200.0
SIGMA = 15.0
SCALE = np.array([0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0])
CHUNK_BYTES = 1 << 30


@pytest.fixture
def fixed_chunk():
    """Pin the chunk budget so only the thread count varies."""
    prev = mpt.set_default(kernel_chunk_bytes=CHUNK_BYTES)
    yield
    mpt.set_default(**prev)


@pytest.fixture
def threads():
    """Restore whatever thread setting the caller had."""
    prev = mpt.get_default('kernel_threads')
    yield
    mpt.set_default(kernel_threads=prev)


def density(r, is_rel, is_per, is_exch=True):
    p_attr = [SCALE[:, None]]
    specs = mpt.flat_specs(p_attr, r=r, rel=is_rel, exch=is_exch)
    return mpt.build_maet(p_attr, None, specs=specs, sigma=[SIGMA],
                          is_per=[is_per], period=[PERIOD], verbose=False)


def evaluated(dens, points, n_threads):
    mpt.set_default(kernel_threads=n_threads)
    return np.asarray(mpt.eval_maet(dens, points, 'none',
                                    verbose=False)).ravel()


class TestResolution:

    def test_an_integer_is_taken_as_given(self, threads):
        mpt.set_default(kernel_threads=3)
        assert kernel_threads_resolved() == 3

    def test_auto_is_capped(self, threads, monkeypatch):
        monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
        mpt.set_default(kernel_threads='auto')
        assert 1 <= kernel_threads_resolved() <= _KERNEL_THREAD_CAP

    def test_auto_follows_the_scheduler_variable(self, threads, monkeypatch):
        """A batch scheduler communicates its allocation this way, and a
        library that ignores it oversubscribes the node."""
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
        mpt.set_default(kernel_threads='auto')
        assert kernel_threads_resolved() == 2

    def test_an_unreadable_scheduler_variable_is_ignored(self, threads,
                                                        monkeypatch):
        monkeypatch.setenv('OMP_NUM_THREADS', 'all of them')
        mpt.set_default(kernel_threads='auto')
        assert 1 <= kernel_threads_resolved() <= _KERNEL_THREAD_CAP

    def test_an_explicit_setting_overrides_the_environment(self, threads,
                                                           monkeypatch):
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
        mpt.set_default(kernel_threads=5)
        assert kernel_threads_resolved() == 5

    @pytest.mark.parametrize('bad', [0, -1, 2.5, 'two', True, None])
    def test_rejected_values(self, bad):
        with pytest.raises(ValueError):
            mpt.set_default(kernel_threads=bad)

    def test_it_is_reported_by_show_defaults(self, capsys):
        mpt.show_defaults()
        assert 'kernel_threads' in capsys.readouterr().out

    def test_the_factory_value_is_auto(self):
        assert mpt.get_defaults()['kernel_threads'] == 'auto'


class TestWorkThreshold:

    def test_small_work_stays_serial(self, threads):
        """Below the threshold the dispatch costs more than it saves."""
        mpt.set_default(kernel_threads=8)
        assert kernel_thread_count(1000) == 1

    def test_large_work_is_divided(self, threads):
        mpt.set_default(kernel_threads=8)
        assert kernel_thread_count(10 ** 8) == 8

    def test_the_division_is_bounded_by_the_work(self, threads):
        """Enough for two threads is not enough for eight."""
        mpt.set_default(kernel_threads=8)
        assert kernel_thread_count(600_000) == 2

    def test_one_thread_means_serial(self, threads):
        mpt.set_default(kernel_threads=1)
        assert kernel_thread_count(10 ** 9) == 1


class TestSplitRanges:

    def test_the_spans_cover_the_range_without_overlap(self):
        spans = split_ranges(1000, 7)
        assert spans[0][0] == 0 and spans[-1][1] == 1000
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))

    def test_no_span_is_empty(self):
        assert all(b > a for a, b in split_ranges(5, 8))

    def test_the_lengths_differ_by_at_most_one(self):
        lengths = [b - a for a, b in split_ranges(1000, 7)]
        assert max(lengths) - min(lengths) <= 1


class TestBitIdentity:
    """Threading changes the schedule, not the arithmetic."""

    @pytest.mark.parametrize('r,is_rel,is_per', [
        (2, False, False),      # centres, bucketed, non-periodic
        (2, False, True),       # centres, periodic, wrapped kernel
        (3, True, False),       # relative
        (2, True, True),        # relative periodic
    ])
    def test_eval_agrees_with_the_serial_path(self, fixed_chunk, threads,
                                              r, is_rel, is_per):
        dens = density(r, is_rel, is_per)
        dim = r - int(is_rel)
        rng = np.random.default_rng(0)
        points = rng.uniform(-PERIOD, PERIOD, (dim, 200_000))
        serial = evaluated(dens, points, 1)
        assert np.array_equal(serial, evaluated(dens, points, 4))

    def test_the_wrapped_kernel_agrees_elementwise(self, threads):
        from mpt._wrapped_kernel import wrapped_gaussian_1d
        d = np.random.default_rng(1).uniform(-3 * PERIOD, 3 * PERIOD,
                                             2_000_000)
        mpt.set_default(kernel_threads=1)
        serial = wrapped_gaussian_1d(d, SIGMA, PERIOD, 6,
                                     exponent_denominator=2)
        mpt.set_default(kernel_threads=4)
        threaded = wrapped_gaussian_1d(d, SIGMA, PERIOD, 6,
                                       exponent_denominator=2)
        assert np.array_equal(serial, threaded)

    def test_the_fourier_branch_agrees_elementwise(self, threads):
        """A sigma wide relative to the period takes the Poisson-summed
        branch, which sums over harmonics rather than images."""
        from mpt._wrapped_kernel import wrapped_gaussian_1d
        d = np.random.default_rng(2).uniform(-PERIOD, PERIOD, 2_000_000)
        mpt.set_default(kernel_threads=1)
        serial = wrapped_gaussian_1d(d, 300.0, PERIOD, 6,
                                     exponent_denominator=2)
        mpt.set_default(kernel_threads=4)
        threaded = wrapped_gaussian_1d(d, 300.0, PERIOD, 6,
                                       exponent_denominator=2)
        assert np.array_equal(serial, threaded)

    def test_the_shape_and_dtype_are_preserved(self, threads):
        from mpt._wrapped_kernel import wrapped_gaussian_1d
        d = np.random.default_rng(3).uniform(-PERIOD, PERIOD,
                                             (4, 500_000)).astype(np.float32)
        mpt.set_default(kernel_threads=4)
        out = wrapped_gaussian_1d(d, SIGMA, PERIOD, 6,
                                  exponent_denominator=2)
        assert out.shape == d.shape and out.dtype == np.float32

    def test_similarity_agrees_with_the_serial_path(self, fixed_chunk,
                                                    threads):
        rng = np.random.default_rng(4)
        context = np.tile(SCALE, (40, 1))
        probes = rng.uniform(0.0, PERIOD, (40, 3))
        args = (context, None, probes, None, SIGMA, 2, False, True, PERIOD)
        mpt.set_default(kernel_threads=1)
        serial = np.asarray(mpt.sim_maet(*args, verbose=False))
        mpt.set_default(kernel_threads=4)
        assert np.array_equal(serial,
                              np.asarray(mpt.sim_maet(*args, verbose=False)))


class TestNesting:

    def test_a_call_from_inside_the_pool_does_not_fan_out_again(self,
                                                                threads):
        """Otherwise a fan-out inside a fan-out would multiply the
        thread count by itself."""
        from concurrent.futures import ThreadPoolExecutor
        from mpt._utils import run_in_kernel_threads
        mpt.set_default(kernel_threads=8)
        seen = []

        def inner(_):
            seen.append(kernel_thread_count(10 ** 8))

        run_in_kernel_threads(inner, [0, 1, 2, 3])
        assert seen == [1, 1, 1, 1]

    def test_the_guard_is_lifted_afterwards(self, threads):
        from mpt._utils import run_in_kernel_threads
        mpt.set_default(kernel_threads=8)
        run_in_kernel_threads(lambda _: None, [0, 1])
        assert kernel_thread_count(10 ** 8) == 8

    def test_results_are_still_correct_from_a_user_thread(self, fixed_chunk,
                                                          threads):
        """A caller parallelising at a higher level is the case the
        default exists for; the values must not depend on it."""
        from concurrent.futures import ThreadPoolExecutor
        dens = density(2, False, True)
        rng = np.random.default_rng(5)
        points = rng.uniform(0.0, PERIOD, (2, 60_000))
        serial = evaluated(dens, points, 1)
        mpt.set_default(kernel_threads=4)
        with ThreadPoolExecutor(3) as pool:
            outs = list(pool.map(
                lambda _: np.asarray(mpt.eval_maet(dens, points, 'none',
                                                   verbose=False)).ravel(),
                range(3)))
        assert all(np.array_equal(serial, o) for o in outs)


class TestPoolLifecycle:

    def test_resetting_the_defaults_discards_the_pool(self, threads):
        import mpt._utils as u
        mpt.set_default(kernel_threads=2)
        u.run_in_kernel_threads(lambda _: None, [0, 1])
        assert u._kernel_pool is not None
        mpt.reset_defaults()
        assert u._kernel_pool is None

    def test_raising_the_count_rebuilds_the_pool(self, threads):
        import mpt._utils as u
        mpt.set_default(kernel_threads=2)
        u.run_in_kernel_threads(lambda _: None, [0, 1])
        first = u._kernel_pool
        mpt.set_default(kernel_threads=6)
        assert u._kernel_pool is None or u._kernel_pool is not first

    def test_an_exception_in_a_task_reaches_the_caller(self, threads):
        from mpt._utils import run_in_kernel_threads
        mpt.set_default(kernel_threads=4)

        def boom(task):
            if task == 2:
                raise ValueError('from a worker')
            return task

        with pytest.raises(ValueError, match='from a worker'):
            run_in_kernel_threads(boom, [0, 1, 2, 3])
