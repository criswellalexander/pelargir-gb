"""
Tests for the Stage-1 naive-SNR pre-filter (SNR_Threshold.naive_snr_max /
naive_snr_survives / prefilter_and_partial_foreground, and its wiring into
PopModel via use_naive_prefilter).

Conventions follow tests/test_thresholding.py: a unit frequency grid
(fs = [1,2,3,4,5], delf = 1) with duration = 1, noisePSD = rx = 1 everywhere unless
overridden, so every expected value below is exact arithmetic.
"""
import os
import sys

os.environ["PELARGIR_GPU"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from thresholding import SNR_Threshold
from models import PopModel


FS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
NF = len(FS)
DELF = FS[1] - FS[0]
ONES = np.ones(NF)


def make_thresher(noisePSD=None, rx=None, block_after=None, fs=FS, duration=1.0):
    """Build an SNR_Threshold on the unit test grid."""
    n = ONES if noisePSD is None else np.asarray(noisePSD, dtype=float)
    r = ONES if rx is None else np.asarray(rx, dtype=float)
    return SNR_Threshold(fs, n, r, duration=duration, block_after=block_after)


def make_binaries(freqs, amps):
    """(2, Ndraws) array of [frequency, amplitude]."""
    return np.array([np.asarray(freqs, dtype=float), np.asarray(amps, dtype=float)])


# =============================================================================
# Unit tests: min_sens / naive_snr_max / naive_snr_survives
# =============================================================================

def test_min_sens_matches_manual_computation():
    ## noisePSD/rx = [2, 0.5, 3, 1, 5] -> min = 0.5
    th = make_thresher(noisePSD=[2, 1, 3, 1, 5], rx=[1, 2, 1, 1, 1])
    assert_allclose(th.min_sens, 0.5)


@pytest.mark.parametrize("snr_thresh", [1.0, 7.0, 12.5])
def test_naive_snr_max_matches_threshold_at_analytic_cutoff(snr_thresh):
    th = make_thresher()  # min_sens = 1, duration = 1 -> naive_snr_max(amp) == amp
    amp_cutoff = snr_thresh * np.sqrt(th.min_sens / th.duration)
    assert_allclose(th.naive_snr_max(np.array([amp_cutoff])), [snr_thresh])
    assert th.naive_snr_survives(np.array([amp_cutoff * 1.0001]), snr_thresh=snr_thresh)[0]
    assert not th.naive_snr_survives(np.array([amp_cutoff * 0.9999]), snr_thresh=snr_thresh)[0]


def test_naive_snr_max_scales_with_response_weighted_noise_floor():
    ## min_sens = 0.5 here -> naive_snr_max(amp) == amp / sqrt(0.5) == amp*sqrt(2)
    th = make_thresher(noisePSD=[2, 1, 3, 1, 5], rx=[1, 2, 1, 1, 1])
    amp = np.array([3.0, 7.0])
    assert_allclose(th.naive_snr_max(amp), amp * np.sqrt(2))


# =============================================================================
# Soundness property: the filter must never reject a true positive
# =============================================================================

@pytest.mark.parametrize("seed", range(15))
def test_naive_filter_never_rejects_a_true_positive(seed):
    rng = np.random.default_rng(seed)
    noisePSD = rng.uniform(0.2, 5.0, NF)
    rx = rng.uniform(0.2, 5.0, NF)
    duration = rng.uniform(0.5, 3.0)
    th = SNR_Threshold(FS, noisePSD, rx, duration=duration)

    n = int(rng.integers(5, 40))
    freqs = rng.uniform(FS[0] - 0.5 * DELF, FS[-1] + 0.5 * DELF, n)
    amps = rng.uniform(0.0, 20.0, n)
    binaries = make_binaries(freqs, amps)

    Nres, fg, res_idx = th.serial_array_sort(binaries, FS, get_indices=True)
    survive_mask = th.naive_snr_survives(amps, snr_thresh=7)

    for idx in res_idx:
        idx = int(idx)
        assert survive_mask[idx], (
            f"resolved binary {idx} (amp={amps[idx]}) was rejected by the naive-SNR "
            "pre-filter; naive_snr_max should always upper-bound the true per-bin SNR"
        )


# =============================================================================
# extra_confusion_psd: the core exact-match deliverable
# =============================================================================

def _mixed_case():
    """
    Per bin: naive-filtered-out companions mixed with survivors.
      bin0 (f=1): amp=8            -- lone source, resolved.
      bin1 (f=2): amp=1,2          -- both naive-dropped, both unresolved.
      bin2 (f=3): amp=1,1,1,1,10   -- four dropped companions, one survivor
                                       (amp=10) that is unresolved once their
                                       confusion is accounted for.
      bin3 (f=4): empty.
      bin4 (f=5): amp=100          -- lone source, resolved.
    """
    freqs = np.array([FS[0], FS[1], FS[1], FS[2], FS[2], FS[2], FS[2], FS[2], FS[4]])
    amps = np.array([8.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 10.0, 100.0])
    return make_binaries(freqs, amps)


EXPECTED_MIXED_NRES = 1  # bin4 only; bin0's resolved source is excluded (bin 0 convention)
EXPECTED_MIXED_FG = np.array([0.0, 5.0, 104.0, 0.0, 0.0])


def test_reference_mixed_case_matches_hand_computation():
    """Sanity check on the fixture itself, independent of the pre-filter."""
    th = make_thresher()
    Nres, fg = th.serial_array_sort(_mixed_case(), FS)
    assert int(Nres) == EXPECTED_MIXED_NRES
    assert_array_equal(fg, EXPECTED_MIXED_FG)


@pytest.mark.parametrize("sorter_name", ["serial_array_sort", "block_array_sort"])
def test_prefilter_with_extra_confusion_psd_matches_reference(sorter_name):
    th = make_thresher(block_after=2)
    binaries = _mixed_case()

    survive_mask, fg_partial = th.prefilter_and_partial_foreground(binaries, FS, snr_thresh=7)
    filtered = binaries[:, survive_mask]

    sorter = getattr(th, sorter_name)
    Nres, fg = sorter(filtered, FS, extra_confusion_psd=fg_partial)
    fg_total = fg + fg_partial

    assert int(Nres) == EXPECTED_MIXED_NRES
    assert_allclose(fg_total, EXPECTED_MIXED_FG)


def test_prefilter_without_extra_confusion_psd_diverges():
    """Without extra_confusion_psd, the amp=10 survivor in bin 2 wrongly flips to
    resolved once its companions are removed."""
    th = make_thresher()
    binaries = _mixed_case()

    survive_mask, fg_partial = th.prefilter_and_partial_foreground(binaries, FS, snr_thresh=7)
    filtered = binaries[:, survive_mask]

    Nres, fg = th.serial_array_sort(filtered, FS)  # extra_confusion_psd omitted
    fg_total = fg + fg_partial

    assert int(Nres) == 2  # bin2's amp=10 source is now wrongly resolved too
    assert_allclose(fg_total, [0.0, 5.0, 4.0, 0.0, 0.0])
    assert not np.allclose(fg_total, EXPECTED_MIXED_FG)


# =============================================================================
# Interaction with the pre-existing serial_array_sort xfail bug
# (tests/test_thresholding.py::test_all_resolved_bin_agrees_between_serial_and_block)
# =============================================================================

def test_prefilter_can_newly_expose_the_all_resolved_bin_bug():
    """
    Known, accepted interaction with the pre-existing serial_array_sort
    "all-resolved bin" bug (see test_thresholding.py). Naive-filtered-out binaries
    are always the sub-threshold anchor in their bin, so removing them can turn a
    bin into an all-True array, which the pre-existing bug misclassifies. Not
    fixed here.
    """
    th = make_thresher(block_after=2)
    binaries = make_binaries([FS[2], FS[2], FS[2]], [1.0, 10.0, 100.0])

    ref_Nres, ref_fg = th.serial_array_sort(binaries, FS)
    assert int(ref_Nres) == 2
    assert_allclose(ref_fg, [0.0, 0.0, 1.0, 0.0, 0.0])  # only the anchor unresolved

    survive_mask, fg_partial = th.prefilter_and_partial_foreground(binaries, FS, snr_thresh=7)
    filtered = binaries[:, survive_mask]
    new_Nres, new_fg = th.serial_array_sort(filtered, FS, extra_confusion_psd=fg_partial)
    new_fg_total = new_fg + fg_partial

    ## the bug manifests: amp=10 is wrongly demoted to unresolved
    assert int(new_Nres) == 1
    assert_allclose(new_fg_total, [0.0, 0.0, 101.0, 0.0, 0.0])
    assert int(new_Nres) != int(ref_Nres)


# =============================================================================
# Full pipeline: PopModel with use_naive_prefilter on vs. off
# =============================================================================

def test_popmodel_prefilter_reproduces_reference_spectrum_and_res_idx():
    ntot = 400
    fbins = np.arange(1e-4, 6e-4, 2e-5)

    pm_ref = PopModel(ntot, np.random.default_rng(2024), fbins=fbins, Nreal=1,
                       block_after=2, use_naive_prefilter=False)
    pm_new = PopModel(ntot, np.random.default_rng(2024), fbins=fbins, Nreal=1,
                       block_after=2, use_naive_prefilter=True)

    fs_ref, fg_ref, Nres_ref, idx_ref, draw_ref = pm_ref.run_model(return_extras=True)
    fs_new, fg_new, Nres_new, idx_new, draw_new = pm_new.run_model(return_extras=True)

    ## sanity check: same draw, so this is a fair comparison
    assert_allclose(draw_ref, draw_new)

    assert_allclose(fs_ref, fs_new)
    assert_allclose(fg_ref, fg_new)
    assert int(Nres_ref) == int(Nres_new)

    idx_ref_set = {int(i) for i in idx_ref}
    idx_new_set = {int(i) for i in idx_new}
    assert idx_ref_set == idx_new_set

    ## remapped indices must be valid positions into the ORIGINAL, full-Ntot draw
    assert all(0 <= int(i) < ntot for i in idx_new)


def test_popmodel_prefilter_reproduces_reference_spectrum_without_indices():
    """Same check via the block_array_sort (return_extras=False) path, with Nreal>1."""
    ntot = 400
    fbins = np.arange(1e-4, 6e-4, 2e-5)

    pm_ref = PopModel(ntot, np.random.default_rng(9), fbins=fbins, Nreal=2,
                       block_after=2, use_naive_prefilter=False)
    pm_new = PopModel(ntot, np.random.default_rng(9), fbins=fbins, Nreal=2,
                       block_after=2, use_naive_prefilter=True)

    fs_ref, fg_ref, Nres_ref = pm_ref.run_model()
    fs_new, fg_new, Nres_new = pm_new.run_model()

    assert_allclose(fs_ref, fs_new)
    assert_allclose(fg_ref, fg_new)
    assert_array_equal(Nres_ref, Nres_new)
