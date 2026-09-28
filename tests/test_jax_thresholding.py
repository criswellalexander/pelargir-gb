"""
Parity tests for the JAX thresholder (SNR_Threshold.jax_array_sort / jax_thresholding.py)
against serial_array_sort and block_array_sort on identical inputs, for the unfiltered
kernel and the per-bin pre-filter at cuts of 1 and snr_thresh.

Runs under any backend: `PELARGIR_BACKEND=cupy pytest tests/test_jax_thresholding.py`
compares against the cupy reference (and feeds JAX through DLPack); under `jax` it
also checks PopModel.run_model's JAX path.
"""
import os
import sys

os.environ.setdefault("PELARGIR_BACKEND", "numpy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import backend
from thresholding import SNR_Threshold
from models import PopModel
from utils import get_amp_freq, to_numpy

pytest.importorskip("jax")
import jax_thresholding

xp = backend.xp

FS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
NF = len(FS)
ONES = np.ones(NF)


def make_thresher(noisePSD=None, rx=None, block_after=2, duration=1.0):
    n = ONES if noisePSD is None else np.asarray(noisePSD, dtype=float)
    r = ONES if rx is None else np.asarray(rx, dtype=float)
    return SNR_Threshold(xp.asarray(FS), xp.asarray(n), xp.asarray(r), duration=duration, block_after=block_after)


def make_binaries(freqs, amps):
    return xp.asarray(np.array([np.asarray(freqs, dtype=float), np.asarray(amps, dtype=float)]))


## prefilter_snr: None (unfiltered), 1 (the default), and snr_thresh, the most aggressive exact cut
with_cuts = pytest.mark.parametrize("cut", [None, 1.0, 7.0], ids=["unfiltered", "cut-1", "cut-7"])


def assert_matches_reference(th, binaries, fs, cut=None):
    """JAX output equals serial and block output (and serial's res_idx set, for one galaxy)."""
    j_Nres, j_fg, j_mask = th.jax_array_sort(binaries, fs, get_mask=True, prefilter_snr=cut)
    for sort in (th.serial_array_sort, th.block_array_sort):
        r_Nres, r_fg = sort(binaries, fs)
        assert_array_equal(to_numpy(j_Nres), to_numpy(r_Nres))
        assert_allclose(to_numpy(j_fg), to_numpy(r_fg), rtol=1e-12, atol=0.0)
    if binaries.ndim == 2:
        _, _, res_idx = th.serial_array_sort(binaries, fs, get_indices=True)
        assert set(np.flatnonzero(to_numpy(j_mask)).tolist()) == {int(i) for i in res_idx}
    return j_Nres, j_fg, j_mask


# =============================================================================
# Unit grid: every case from test_thresholding.py, against both references
# =============================================================================

UNIT_CASES = {
    "single unresolved":        (dict(), [3.0], [1.0]),
    "one source per bin":       (dict(), FS, np.ones(NF)),
    "top bin":                  (dict(), [5.0], [1.0]),
    "response power":           (dict(rx=[1.0, 4.0, 0.25, 9.0, 16.0]), FS, [0.1, 0.2, 0.3, 0.4, 0.5]),
    "noise spike elsewhere":    (dict(noisePSD=[1.0, 1.0, 1e12, 1.0, 1.0]), [2.0], [10.0]),
    "noise spike own bin":      (dict(noisePSD=[1.0, 1e12, 1.0, 1.0, 1.0]), [2.0], [10.0]),
    "response elsewhere":       (dict(rx=[1.0, 1.0, 0.0625, 1.0, 1.0]), [2.0], [10.0]),
    "response own bin":         (dict(rx=[1.0, 0.0625, 1.0, 1.0, 1.0]), [2.0], [10.0]),
    "bin 0 resolved":           (dict(), [1.0], [20.0]),
    "out of band":              (dict(), [0.1, 3.0, 10.0], [2.0, 1.0, 3.0]),
    "below-band contamination": (dict(), [0.1, 3.0], [1e6, 1.0]),
    "confusion cumsum":         (dict(), np.full(4, 3.0), [3.0, 4.0, 12.0, 100.0]),
    "last sub-threshold":       (dict(), np.full(3, 5.0), [8.0, 9.0, 1000.0]),
    "top bin confusion":        (dict(), np.full(2, 5.0), [0.5, 10.0]),
    "boundary case":            (dict(), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
                                 [0.5, 20, 0.5, 20, 0.5, 20, 0.5, 20, 0.5, 20, 30]),
    "all resolved [T,T]":       (dict(), [3, 3, 5, 5, 5], [10.0, 100.0, 0.5, 2.0, 3.0]),
    "loudest buried [F,T,F]":   (dict(), [3, 3, 3], [1.0, 10.0, 10.5]),
    ## pre-filter cases (bare-noise SNR = amplitude on this grid): ten 0.9s are dropped at
    ## cut 1 but their confusion still buries the 8 (8/sqrt(1 + 8.1) < 7)
    "dropped confusion buries": (dict(), np.full(11, 3.0), [0.9]*10 + [8.0]),
    ## every survivor passes, so the last sub-threshold source is a dropped one
    "all survivors resolved":   (dict(), np.full(4, 4.0), [0.5, 0.5, 50.0, 1000.0]),
    "only dropped":             (dict(), [2.0, 2.0], [0.3, 0.6]),
    "dropped with bin 0 / oob": (dict(), [0.1, 1.0, 10.0, 3.0, 3.0], [0.5, 20.0, 30.0, 0.2, 9.0]),
}


@with_cuts
@pytest.mark.parametrize("name", list(UNIT_CASES))
def test_unit_grid_matches_reference(name, cut):
    kwargs, freqs, amps = UNIT_CASES[name]
    th = make_thresher(**kwargs)
    assert_matches_reference(th, make_binaries(freqs, amps), xp.asarray(FS), cut)


@with_cuts
def test_exact_values_for_the_eq17_edge_cases(cut):
    th = make_thresher()
    sort = lambda b: th.jax_array_sort(b, xp.asarray(FS), get_mask=True, prefilter_snr=cut)
    Nres, fg, mask = sort(make_binaries([3, 3, 3], [1.0, 10.0, 10.5]))
    assert int(Nres) == 0
    assert_allclose(to_numpy(fg), [0.0, 0.0, 211.25, 0.0, 0.0], rtol=1e-15, atol=0.0)
    assert not to_numpy(mask).any()

    Nres, fg, mask = sort(make_binaries([3, 3], [10.0, 100.0]))
    assert int(Nres) == 2
    assert_array_equal(to_numpy(fg), np.zeros(NF))
    assert to_numpy(mask).all()


@with_cuts
def test_realization_and_parallel_axes_are_independent(cut):
    ## trailing f = 100 entries are out-of-band filler with huge amplitudes
    b = np.zeros((2, 2, 2, 2))
    b[:, :, 0, 0] = [[2.0, 100.0], [20.0, 1e9]]
    b[:, :, 0, 1] = [[3.0, 100.0], [2.0, 1e9]]
    b[:, :, 1, 0] = [[5.0, 100.0], [3.0, 1e9]]
    b[:, :, 1, 1] = [[1.0, 5.0], [4.0, 5.0]]
    th = make_thresher()
    Nres, fg, mask = assert_matches_reference(th, xp.asarray(b), xp.asarray(FS), cut)
    assert_array_equal(to_numpy(Nres), [[1, 0], [0, 0]])
    assert to_numpy(mask).shape == (2, 2, 2)


# =============================================================================
# Realistic draws: Nreal = 2, Nparallel = 3
# =============================================================================

@pytest.fixture(scope="module")
def realistic():
    pm = PopModel(int(1e5), xp.random.default_rng(11), Nreal=2, block_after=4)
    pm.gbprior.condition(pm.hyperprior.sample(3))
    A, f = get_amp_freq(pm.gbprior.sample_conditional(pm.N))
    return pm, xp.array([f, A])


@with_cuts
def test_realistic_draws_match_block_and_serial(realistic, cut):
    pm, obs = realistic
    th = pm.thresher
    assert obs.shape[2:] == (2, 3)
    j_Nres, j_fg, j_mask = th.jax_array_sort(obs, pm.fbins, snr_thresh=pm.thresh_val, get_mask=True,
                                             prefilter_snr=cut)
    for sort in (th.serial_array_sort, th.block_array_sort):
        r_Nres, r_fg = sort(obs, pm.fbins, snr_thresh=pm.thresh_val)
        assert_array_equal(to_numpy(j_Nres), to_numpy(r_Nres))
        assert_allclose(to_numpy(j_fg), to_numpy(r_fg), rtol=1e-12, atol=0.0)
    ## resolved sets, one galaxy at a time (serial's res_idx needs Nrealz == Nparallel == 1)
    for r in range(2):
        for p in range(3):
            _, _, res_idx = th.serial_array_sort(xp.ascontiguousarray(obs[:, :, r, p]), pm.fbins,
                                                 snr_thresh=pm.thresh_val, get_indices=True)
            assert set(np.flatnonzero(to_numpy(j_mask[:, r, p])).tolist()) == {int(i) for i in res_idx}


@with_cuts
@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_results_do_not_depend_on_batch_size(realistic, batch_size, cut):
    pm, obs = realistic
    th = pm.thresher
    ref = th.jax_array_sort(obs, pm.fbins, snr_thresh=pm.thresh_val, get_mask=True, prefilter_snr=None)
    out = th.jax_array_sort(obs, pm.fbins, snr_thresh=pm.thresh_val, get_mask=True, batch_size=batch_size,
                            prefilter_snr=cut)
    assert_array_equal(to_numpy(out[0]), to_numpy(ref[0]))
    assert_allclose(to_numpy(out[1]), to_numpy(ref[1]), rtol=1e-12, atol=0.0)
    assert_array_equal(to_numpy(out[2]), to_numpy(ref[2]))


@with_cuts
def test_same_shape_call_does_not_recompile(realistic, cut):
    pm, obs = realistic
    th = pm.thresher
    ## the second call differs only in traced scalars (threshold and cut)
    calls = [dict(snr_thresh=pm.thresh_val, prefilter_snr=cut),
             dict(snr_thresh=6.5, prefilter_snr=None if cut is None else min(cut, 6.5) - 0.25)]
    for kw in calls:   ## settles the capacity bucket
        th.jax_array_sort(obs, pm.fbins, batch_size=4, **kw)
    n_compiled = jax_thresholding._threshold_batch._cache_size()
    n_counted = jax_thresholding._count_survivors._cache_size()
    for kw in calls:
        th.jax_array_sort(obs, pm.fbins, batch_size=4, **kw)
    assert jax_thresholding._threshold_batch._cache_size() == n_compiled
    assert jax_thresholding._count_survivors._cache_size() == n_counted


def test_capacity_overflow_is_rerun(realistic):
    """A capacity below the survivor count is detected, grown, and gives the unfiltered result."""
    pm, obs = realistic
    th = pm.thresher
    N, G = obs.shape[1], obs.shape[2]*obs.shape[3]
    edges = pm.fbins + 0.5*th.delf
    args = (obs, edges, th.noisePSD, th.LISA_rx, th.duration, th.duration_eff)
    cache = {(N, G): jax_thresholding.MIN_CAPACITY}
    out = jax_thresholding.jax_threshold(*args, snr_thresh=pm.thresh_val, return_mask=True,
                                         prefilter_snr=1.0, capacity_cache=cache)
    ref = jax_thresholding.jax_threshold(*args, snr_thresh=pm.thresh_val, return_mask=True, prefilter_snr=None)
    n_surv = int(jax_thresholding.jax.numpy.max(jax_thresholding._count_survivors(
        jax_thresholding._to_jax(xp.ascontiguousarray(xp.moveaxis(obs.reshape(2, N, G), -1, 0))),
        *[jax_thresholding._to_jax(xp.asarray(c, dtype=xp.float64)) for c in (edges, th.noisePSD, th.LISA_rx)],
        float(th.duration), 1.0)))
    assert n_surv > jax_thresholding.MIN_CAPACITY
    assert cache[(N, G)] >= n_surv
    assert_array_equal(to_numpy(out[0]), to_numpy(ref[0]))
    assert_allclose(to_numpy(out[1]), to_numpy(ref[1]), rtol=1e-12, atol=0.0)
    assert_array_equal(to_numpy(out[2]), to_numpy(ref[2]))


def test_prefilter_above_threshold_is_rejected():
    th = make_thresher()
    with pytest.raises(ValueError):
        th.jax_array_sort(make_binaries([3.0], [1.0]), xp.asarray(FS), snr_thresh=7, prefilter_snr=7.5)


# =============================================================================
# PopModel.run_model on the jax backend
# =============================================================================

jax_backend_only = pytest.mark.skipif(backend.BACKEND != "jax", reason="needs PELARGIR_BACKEND=jax")


@jax_backend_only
def test_run_model_jax_path_matches_serial_reference():
    pm = PopModel(int(1e5), xp.random.default_rng(3), Nreal=1, block_after=4)
    fs, fg, Nres, res_idx, galaxy_draw = pm.run_model(return_extras=True)
    A, f = get_amp_freq(galaxy_draw)
    r_Nres, r_fg, r_idx = pm.thresher.serial_array_sort(xp.array([f, A]), pm.fbins,
                                                        snr_thresh=pm.thresh_val, get_indices=True)
    assert int(Nres) == int(r_Nres)
    assert_allclose(to_numpy(fg), to_numpy(pm.reweight_foreground(r_fg)[1:, ...]), rtol=1e-12, atol=0.0)
    assert set(res_idx) == {int(i) for i in r_idx}


@jax_backend_only
def test_run_model_jax_path_with_realizations():
    pm = PopModel(int(1e4), xp.random.default_rng(4), Nreal=2, block_after=4, jax_batch_size=1)
    fs, fg, Nres = pm.run_model()
    assert fg.shape == (len(pm.fbins) - 1, 2, 1)
    assert Nres.shape == (2, 1)
    assert np.isfinite(to_numpy(fg)).all()
