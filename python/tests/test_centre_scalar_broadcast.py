"""Tests for global scalar broadcasting of ``window_spec['centre']``.

When ``centre`` is a size-1 input (Python scalar, 0-D ndarray,
length-1 sequence, or 1x1 ndarray), :func:`window_tensor` broadcasts
the value across every per-attribute slot, equivalent to passing a
length-d_a array of identical values per attribute.

Per-attribute scalars in list form (e.g. ``[5.0, 10.0]`` for two
attributes with d_a > 1 each) are NOT broadcast — that input is
interpreted as flat form by the existing dispatcher and rejected on
length mismatch. Users wanting per-attribute uniform centres at
distinct values per attribute pass the explicit list-of-arrays form.
"""
import numpy as np
import pytest

from mpt.tensor import build_exp_tens, cos_sim_exp_tens, window_tensor


def _build(seed, A, K, N, r, sigma=20.0, is_rel=0):
    rng = np.random.default_rng(seed)
    P = [rng.uniform(-50, 50, size=(K, N)) for _ in range(A)]
    W = [rng.uniform(0.5, 1.5, size=(K, N)) for _ in range(A)]
    sigma_vec = np.array([sigma], dtype=np.float64)
    r_vec = np.asarray(r, dtype=np.intp)
    groups = np.zeros(A, dtype=np.intp)
    is_rel_vec = np.array([is_rel], dtype=np.intp)
    is_per_vec = np.array([0], dtype=np.intp)
    period_vec = np.array([0.0])
    return build_exp_tens(P, W, sigma_vec, r_vec, groups,
                          is_rel_vec, is_per_vec, period_vec,
                          verbose=False)


@pytest.mark.parametrize("scalar_form", [
    5.0,
    5,
    np.float64(5.0),
    np.array(5.0),         # 0-D
    np.array([5.0]),       # length-1 array
    np.array([[5.0]]),     # 1x1 array
])
def test_global_scalar_broadcast_sa(scalar_form):
    """Each numeric size-1 input form produces the same wmd and
    cosine as the explicit length-d_a uniform reference. List/tuple
    inputs do NOT take the broadcast path — see
    ``test_list_form_never_broadcasts``."""
    dens_c = _build(seed=1, A=1, K=4, N=4, r=[3])
    dens_q = _build(seed=2, A=1, K=4, N=3, r=[3])
    d_a = int(dens_c.dim_per_attr[0])
    assert d_a == 3

    ref_centre = [np.full(d_a, 5.0)]
    wmd_ref = window_tensor(dens_c, dict(size=1.5, mix=0.5,
                                          centre=ref_centre))
    cos_ref = cos_sim_exp_tens(dens_q, wmd_ref, verbose=False)

    wmd = window_tensor(dens_c, dict(size=1.5, mix=0.5,
                                      centre=scalar_form))
    # Resulting centre should be byte-identical to the reference.
    assert len(wmd.centre) == len(wmd_ref.centre)
    np.testing.assert_array_equal(wmd.centre[0], wmd_ref.centre[0])

    cos = cos_sim_exp_tens(dens_q, wmd, verbose=False)
    assert cos == cos_ref


def test_list_form_never_broadcasts():
    """List/tuple inputs are interpreted structurally and never
    broadcast, even when size-1. A length-1 list on A > 1 raises;
    a length-1 list of length-1 arrays on A=1 with d_a > 1 raises
    on inner-length mismatch. Preserves the contract that a
    wrong-length cell is a user error."""
    # A=2 case: length-1 list raises on length mismatch.
    rng = np.random.default_rng(101)
    P_list = [rng.uniform(-50, 50, size=(4, 5)) for _ in range(2)]
    W_list = [rng.uniform(0.5, 1.5, size=(4, 5)) for _ in range(2)]
    sigma_vec = np.array([20.0, 25.0])
    r_vec = np.array([1, 1], dtype=np.intp)
    groups = np.array([0, 1], dtype=np.intp)
    is_rel = np.array([0, 0], dtype=np.intp)
    is_per = np.array([0, 0], dtype=np.intp)
    period = np.array([0.0, 0.0])
    dens_c = build_exp_tens(P_list, W_list, sigma_vec, r_vec, groups,
                             is_rel, is_per, period, verbose=False)
    with pytest.raises(ValueError, match="length A = 2; got length 1"):
        window_tensor(dens_c, dict(size=[1, 1], mix=[0, 0],
                                    centre=[np.zeros(1)]))
    # Length-1 list whose inner is a length-1 array.
    with pytest.raises(ValueError, match="length A = 2; got length 1"):
        window_tensor(dens_c, dict(size=[1, 1], mix=[0, 0],
                                    centre=[[0.0]]))

    # A=1, d_a=3 case: length-1 list of length-1 array raises on
    # inner-length mismatch (cell-form).
    dens = _build(seed=102, A=1, K=4, N=4, r=[3])
    with pytest.raises(ValueError, match=r"\[0\] must have length 3"):
        window_tensor(dens, dict(size=1.0, mix=0.0,
                                  centre=[np.zeros(1)]))


def test_global_scalar_broadcast_ma_two_groups():
    """With A=2 attributes in different groups, a single scalar fills
    every slot of both attributes."""
    rng = np.random.default_rng(42)
    P_list = [rng.uniform(-50, 50, size=(4, 5)) for _ in range(2)]
    W_list = [rng.uniform(0.5, 1.5, size=(4, 5)) for _ in range(2)]
    P_q = [rng.uniform(-50, 50, size=(4, 3)) for _ in range(2)]
    W_q = [rng.uniform(0.5, 1.5, size=(4, 3)) for _ in range(2)]

    sigma_vec = np.array([20.0, 25.0])
    r_vec = np.array([2, 3], dtype=np.intp)
    groups = np.array([0, 1], dtype=np.intp)
    is_rel = np.array([0, 0], dtype=np.intp)
    is_per = np.array([0, 0], dtype=np.intp)
    period = np.array([0.0, 0.0])
    dens_c = build_exp_tens(P_list, W_list, sigma_vec, r_vec, groups,
                             is_rel, is_per, period, verbose=False)
    dens_q = build_exp_tens(P_q, W_q, sigma_vec, r_vec, groups,
                             is_rel, is_per, period, verbose=False)
    # d_a per attribute = r_a since absolute: [2, 3]; dim_total = 5.
    assert list(dens_c.dim_per_attr) == [2, 3]

    # Reference: explicit uniform per attribute.
    ref_centre = [np.full(2, 7.0), np.full(3, 7.0)]
    wmd_ref = window_tensor(dens_c, dict(size=[2.0, 2.0], mix=[0.5, 0.5],
                                          centre=ref_centre))
    cos_ref = cos_sim_exp_tens(dens_q, wmd_ref, verbose=False)

    # Scalar broadcast.
    wmd = window_tensor(dens_c, dict(size=[2.0, 2.0], mix=[0.5, 0.5],
                                      centre=7.0))
    np.testing.assert_array_equal(wmd.centre[0], np.full(2, 7.0))
    np.testing.assert_array_equal(wmd.centre[1], np.full(3, 7.0))
    cos = cos_sim_exp_tens(dens_q, wmd, verbose=False)
    assert cos == cos_ref


def test_per_attribute_scalar_list_not_broadcast():
    """A length-A list of scalars is interpreted as flat-form and
    rejected when length != dim_total — this preserves backward
    compatibility with the existing flat-form behaviour."""
    rng = np.random.default_rng(43)
    P_list = [rng.uniform(-50, 50, size=(4, 5)) for _ in range(2)]
    W_list = [rng.uniform(0.5, 1.5, size=(4, 5)) for _ in range(2)]
    sigma_vec = np.array([20.0, 25.0])
    r_vec = np.array([2, 3], dtype=np.intp)
    groups = np.array([0, 1], dtype=np.intp)
    is_rel = np.array([0, 0], dtype=np.intp)
    is_per = np.array([0, 0], dtype=np.intp)
    period = np.array([0.0, 0.0])
    dens_c = build_exp_tens(P_list, W_list, sigma_vec, r_vec, groups,
                             is_rel, is_per, period, verbose=False)
    # dim_total = 2 + 3 = 5; a length-2 list-of-scalars is not auto-
    # broadcast to per-attribute uniform centres.
    with pytest.raises(ValueError, match="dim = 5; got length 2"):
        window_tensor(dens_c, dict(size=[2.0, 2.0], mix=[0.5, 0.5],
                                    centre=[5.0, 10.0]))


@pytest.mark.parametrize("bad_centre,match", [
    ([np.array([5.0, 5.0])],         r"centre'\]\[0\] must have length 3"),
    (np.array([1.0, 2.0, 3.0, 4.0]), r"centre'\] \(flat form\) must have length dim = 3"),
    (np.array([1.0, 2.0]),           r"centre'\] \(flat form\) must have length dim = 3"),
])
def test_wrong_length_still_errors_informatively(bad_centre, match):
    """Wrong-length inputs that are NOT size-1 still raise informative
    ValueError. The scalar-broadcast bypass should not catch them."""
    dens = _build(seed=44, A=1, K=4, N=4, r=[3])
    with pytest.raises(ValueError, match=match):
        window_tensor(dens, dict(size=1.5, mix=0.5, centre=bad_centre))


def test_scalar_broadcast_matches_flat_uniform():
    """Scalar 5.0 and flat-form np.array([5.0]*dim) produce identical
    results."""
    dens = _build(seed=45, A=1, K=4, N=4, r=[3])
    d_a = int(dens.dim_per_attr[0])
    spec_scalar = dict(size=2.0, mix=0.3, centre=5.0)
    spec_flat = dict(size=2.0, mix=0.3, centre=np.full(d_a, 5.0))
    wmd_s = window_tensor(dens, spec_scalar)
    wmd_f = window_tensor(dens, spec_flat)
    np.testing.assert_array_equal(wmd_s.centre[0], wmd_f.centre[0])


def test_scalar_zero_broadcast():
    """Edge case: scalar 0.0 should still broadcast."""
    dens = _build(seed=46, A=1, K=4, N=4, r=[3])
    wmd = window_tensor(dens, dict(size=1.5, mix=0.5, centre=0.0))
    np.testing.assert_array_equal(wmd.centre[0], np.zeros(3))


def test_scalar_negative_broadcast():
    """Negative scalar values broadcast correctly."""
    dens = _build(seed=47, A=1, K=4, N=4, r=[3])
    wmd = window_tensor(dens, dict(size=1.5, mix=0.5, centre=-12.5))
    np.testing.assert_array_equal(wmd.centre[0], np.full(3, -12.5))
