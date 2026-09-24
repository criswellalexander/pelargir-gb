"""
Tests for pelargir.thresholding.SNR_Threshold, with a focus on the frequency-bin
alignment between the binaries and the noise PSD / LISA response function they are
thresholded against.

The convention under test: coarsegrain_bin returns f_idx such that a binary anywhere
in [fs[k] - 0.5*delf, fs[k] + 0.5*delf) gets f_idx == k. That makes f_idx a direct
index into fs, and hence into noisePSD and LISA_rx, which is what the sort loops
assume when they apply self.noisePSD[ii] / sqrt(self.LISA_rx[ii]) to the binaries
selected by f_idx == ii.

Most tests use a unit frequency grid (fs = [1,2,3,4,5], delf = 1) with duration = 1.
Under those values duration_eff = 1/delf = 1 and calc_Nij collapses to

    N_i = A_i_eff / sqrt(Sn_bin + sum_{j<i} A_j_eff**2),   A_eff = A * sqrt(LISA_rx[bin])

so every expected value below is exact arithmetic rather than a recorded output.
A unit grid also removes any floating-point ambiguity at the bin edges.
"""
import os
import sys

## thresholding.py reads PELARGIR_GPU at module scope, so this must precede the import
os.environ["PELARGIR_GPU"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from thresholding import SNR_Threshold
from utils import lisa_noise_psd


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
# T1 -- the index mapping itself
# =============================================================================

@pytest.mark.parametrize(
    "freq, expected",
    [
        (1e-5, 0),                  # far below the band
        (FS[0] - 0.5 * DELF, 0),    # lower edge of bin 0 (left-closed)
        (FS[0], 0),                 # centre of bin 0
        (FS[0] + 0.5 * DELF, 1),    # edge bin0/bin1 belongs to the upper bin
        (FS[1] - 0.3 * DELF, 1),
        (FS[1], 1),                 # centre of bin 1
        (FS[1] + 0.3 * DELF, 1),
        (FS[2], 2),
        (FS[3], 3),
        (FS[3] + 0.5 * DELF, 4),
        (FS[-1], NF - 1),           # centre of the TOP bin
        (FS[-1] + 0.5 * DELF, NF),  # top edge -> out of band
        (10.0, NF),
        (100.0, NF),
    ],
)
def test_coarsegrain_bin_index_mapping(freq, expected):
    th = make_thresher()
    amps, f_idx = th.coarsegrain_bin(make_binaries([freq], [1.0]), FS)
    assert int(f_idx[0]) == expected


def test_coarsegrain_bin_passes_amplitudes_through():
    th = make_thresher()
    freqs = np.array([1.0, 2.5, 4.0])
    in_amps = np.array([3.0, 7.0, 11.0])
    amps, f_idx = th.coarsegrain_bin(make_binaries(freqs, in_amps), FS)
    assert_array_equal(amps, in_amps)
    assert f_idx.shape == freqs.shape


# =============================================================================
# T2 / T7 / T6 -- placement of unresolved sources
# =============================================================================

def test_single_unresolved_source_lands_in_its_own_bin():
    ## A = 1, Sn = 1 -> N = 1 < 7, so the only thing under test is placement
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries([3.0], [1.0]), FS)
    assert int(Nres) == 0
    assert_array_equal(fg, [0.0, 0.0, 1.0, 0.0, 0.0])


def test_every_bin_receives_its_own_source():
    ## one sub-threshold source per bin -> every bin must be populated
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries(FS, np.ones(NF)), FS)
    assert int(Nres) == 0
    assert_array_equal(fg, np.ones(NF))
    assert np.all(fg > 0)


def test_top_bin_is_not_dropped():
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries([FS[-1]], [1.0]), FS)
    assert int(Nres) == 0
    assert_array_equal(fg, [0.0, 0.0, 0.0, 0.0, 1.0])


def test_foreground_conserves_power_for_in_band_unresolved_sources():
    rx = np.array([1.0, 4.0, 0.25, 9.0, 16.0])
    th = make_thresher(rx=rx)
    amps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Nres, fg = th.serial_array_sort(make_binaries(FS, amps), FS)
    assert int(Nres) == 0
    assert_allclose(fg.sum(), np.sum(amps**2 * rx), rtol=1e-12, atol=0.0)


# =============================================================================
# T3 / T4 / T5 -- the noise PSD and response are read at the binary's own bin
# =============================================================================

def test_noise_psd_is_read_at_the_binaries_own_bin():
    ## spike in bin 2; the binary is in bin 1 and must not see it
    th = make_thresher(noisePSD=[1.0, 1.0, 1e12, 1.0, 1.0])
    Nres, fg = th.serial_array_sort(make_binaries([FS[1]], [10.0]), FS)
    assert int(Nres) == 1          # N = 10/sqrt(1) = 10 >= 7
    assert_array_equal(fg, np.zeros(NF))


def test_noise_psd_spike_in_the_binaries_own_bin_does_suppress_it():
    ## same source, spike moved onto bin 1; pins the sign of the mapping
    th = make_thresher(noisePSD=[1.0, 1e12, 1.0, 1.0, 1.0])
    Nres, fg = th.serial_array_sort(make_binaries([FS[1]], [10.0]), FS)
    assert int(Nres) == 0          # N = 10/1e6 << 7
    assert_array_equal(fg, [0.0, 100.0, 0.0, 0.0, 0.0])


def test_lisa_response_is_read_at_the_binaries_own_bin():
    ## 0.0625 = 1/16 so sqrt is exactly 0.25
    th = make_thresher(rx=[1.0, 1.0, 0.0625, 1.0, 1.0])
    Nres, fg = th.serial_array_sort(make_binaries([FS[1]], [10.0]), FS)
    assert int(Nres) == 1          # A_eff = 10, N = 10 >= 7
    assert_array_equal(fg, np.zeros(NF))


def test_lisa_response_suppression_in_the_binaries_own_bin():
    th = make_thresher(rx=[1.0, 0.0625, 1.0, 1.0, 1.0])
    Nres, fg = th.serial_array_sort(make_binaries([FS[1]], [10.0]), FS)
    assert int(Nres) == 0          # A_eff = 2.5, N = 2.5 < 7
    assert_array_equal(fg, [0.0, 6.25, 0.0, 0.0, 0.0])


# =============================================================================
# T8 / T9 -- bin 0 and out-of-band handling
# =============================================================================

def test_lowest_bin_is_excluded_from_Nres():
    ## Nres = sum(Nres_f[1:]); a resolved source in bin 0 is deliberately not counted
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries([FS[0]], [20.0]), FS)
    assert int(Nres) == 0
    assert_array_equal(fg, np.zeros(NF))


def test_out_of_band_sources():
    ## below band -> quarantined in bin 0; above band -> dropped entirely
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries([0.1, 3.0, 10.0], [2.0, 1.0, 3.0]), FS)
    assert int(Nres) == 0
    assert_array_equal(fg, [4.0, 0.0, 1.0, 0.0, 0.0])


def test_below_band_contamination_is_quarantined_in_bin_zero():
    th = make_thresher()
    ref_Nres, ref_fg = th.serial_array_sort(make_binaries([3.0], [1.0]), FS)
    Nres, fg = th.serial_array_sort(make_binaries([0.1, 3.0], [1e6, 1.0]), FS)
    assert int(Nres) == int(ref_Nres)
    assert_array_equal(fg[1:], ref_fg[1:])


# =============================================================================
# T10 / T11 / T12 -- the cumsum confusion term and the tilt/argmax rule
# =============================================================================

def test_confusion_noise_cumsum_path():
    ## A = [3,4,12,100]: A^2 = [9,16,144,1e4], confusion = [0,9,25,169],
    ## denom = [1,10,26,170], N = [3, 1.265, 2.353, 7.670] -> only the last is resolved.
    ## Note A=12 is buried by the A=3 source despite its larger raw amplitude.
    th = make_thresher()
    freqs = np.full(4, FS[2])
    Nres, fg = th.serial_array_sort(make_binaries(freqs, [3.0, 4.0, 12.0, 100.0]), FS)
    assert int(Nres) == 1
    assert_array_equal(fg, [0.0, 0.0, 169.0, 0.0, 0.0])


def test_only_sources_above_the_last_sub_threshold_source_are_resolved():
    ## A = [8,9,1000] in the TOP bin: N = [8, 1.116, 82.8] -> snr_filt = [T,F,T],
    ## but only entries after the final False count, so res_filt = [F,F,T].
    th = make_thresher()
    freqs = np.full(3, FS[-1])
    Nres, fg = th.serial_array_sort(make_binaries(freqs, [8.0, 9.0, 1000.0]), FS)
    assert int(Nres) == 1
    assert_array_equal(fg, [0.0, 0.0, 0.0, 0.0, 145.0])


def test_top_bin_with_confusion():
    ## A = [0.5, 10]: N = [0.5, 8.944] -> res_filt = [F, T]
    th = make_thresher()
    freqs = np.full(2, FS[-1])
    Nres, fg = th.serial_array_sort(make_binaries(freqs, [0.5, 10.0]), FS)
    assert int(Nres) == 1
    assert_array_equal(fg, [0.0, 0.0, 0.0, 0.0, 0.25])


# =============================================================================
# T13 -- serial vs block
# =============================================================================

def _boundary_case():
    ## every populated bin carries a sub-threshold companion; see the xfail below for why
    freqs = np.array([FS[0], FS[0], FS[1], FS[1], FS[2], FS[2],
                      FS[3], FS[3], FS[4], FS[4], FS[4]])
    amps = np.array([0.5, 20.0, 0.5, 20.0, 0.5, 20.0,
                     0.5, 20.0, 0.5, 20.0, 30.0])
    return make_binaries(freqs, amps)


EXPECTED_BOUNDARY_FG = np.array([0.25, 0.25, 0.25, 0.25, 900.25])


def test_serial_sort_across_all_bins():
    th = make_thresher()
    Nres, fg = th.serial_array_sort(_boundary_case(), FS)
    assert int(Nres) == 4          # bins 1-4; bin 0's resolved source is excluded
    assert_array_equal(fg, EXPECTED_BOUNDARY_FG)


@pytest.mark.parametrize("block_after", [0, 1, 2, 3, 4])
def test_block_sort_matches_serial_across_the_boundary(block_after):
    th = make_thresher(block_after=block_after)
    s_Nres, s_fg = th.serial_array_sort(_boundary_case(), FS)
    b_Nres, b_fg = th.block_array_sort(_boundary_case(), FS)
    assert int(b_Nres) == int(s_Nres) == 4
    assert_array_equal(b_fg, s_fg)
    assert_array_equal(b_fg, EXPECTED_BOUNDARY_FG)


@pytest.mark.xfail(strict=True, reason="pre-existing: a bin whose sources are ALL above "
                                       "threshold loses its faintest source (tilt_filt[0] "
                                       "is always 0), and the block path's zero-padding "
                                       "supplies the missing False, so serial and block "
                                       "disagree. Out of scope for the alignment fix.")
def test_all_resolved_bin_agrees_between_serial_and_block():
    th = make_thresher(block_after=2)
    binaries = make_binaries([FS[2], FS[2], FS[4], FS[4], FS[4]],
                             [10.0, 100.0, 0.5, 2.0, 3.0])
    s_Nres, s_fg = th.serial_array_sort(binaries, FS)
    b_Nres, b_fg = th.block_array_sort(binaries, FS)
    assert int(s_Nres) == int(b_Nres)
    assert_array_equal(s_fg, b_fg)


# =============================================================================
# T14 / T15 -- realization and parallel axes
# =============================================================================

def test_realization_and_parallel_axes_are_independent():
    ## trailing f = 100 entries are ragged-column filler: out of band in both codes.
    ## their A = 1e9 makes any regression in out-of-band handling explode loudly.
    binaries = np.zeros((2, 2, 2, 2))
    binaries[:, :, 0, 0] = [[2.0, 100.0], [20.0, 1e9]]
    binaries[:, :, 0, 1] = [[3.0, 100.0], [2.0, 1e9]]
    binaries[:, :, 1, 0] = [[5.0, 100.0], [3.0, 1e9]]
    binaries[:, :, 1, 1] = [[1.0, 5.0], [4.0, 5.0]]

    th = make_thresher()
    Nres, fg = th.serial_array_sort(binaries, FS)

    assert_array_equal(Nres, [[1, 0], [0, 0]])
    assert fg.shape == (NF, 2, 2)
    assert_array_equal(fg[:, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert_array_equal(fg[:, 0, 1], [0.0, 0.0, 4.0, 0.0, 0.0])
    assert_array_equal(fg[:, 1, 0], [0.0, 0.0, 0.0, 0.0, 9.0])
    assert_array_equal(fg[:, 1, 1], [16.0, 0.0, 0.0, 0.0, 25.0])

    th_block = make_thresher(block_after=2)
    b_Nres, b_fg = th_block.block_array_sort(binaries, FS)
    assert_array_equal(b_Nres, Nres)
    assert_array_equal(b_fg, fg)


def test_force_shape_required_when_realizations_exceed_draws():
    binaries = np.zeros((2, 1, 3, 1))
    binaries[0, 0, :, 0] = [1.0, 3.0, 5.0]
    binaries[1, 0, :, 0] = [1.0, 2.0, 3.0]
    th = make_thresher()
    with pytest.raises(RuntimeError):
        th.serial_array_sort(binaries, FS)

    Nres, fg = th.serial_array_sort(binaries, FS, force_shape=True)
    assert_array_equal(Nres, [[0], [0], [0]])
    assert_array_equal(fg[:, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert_array_equal(fg[:, 1, 0], [0.0, 0.0, 4.0, 0.0, 0.0])
    assert_array_equal(fg[:, 2, 0], [0.0, 0.0, 0.0, 0.0, 9.0])


# =============================================================================
# T16 / T17 / T18 -- contract guards
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_shape_contract_and_squeeze(ndim):
    freqs, amps = [3.0], [1.0]
    binaries = make_binaries(freqs, amps)
    if ndim >= 3:
        binaries = binaries[:, :, np.newaxis]
    if ndim == 4:
        binaries = binaries[..., np.newaxis]
    th = make_thresher()
    Nres, fg = th.serial_array_sort(binaries, FS)
    assert np.ndim(Nres) == 0
    assert fg.shape == (NF,)


def test_invalid_ndim_raises():
    th = make_thresher()
    with pytest.raises(ValueError):
        th.serial_array_sort(np.array([1.0, 2.0]), FS)


def test_all_out_of_band_returns_zeros():
    th = make_thresher(block_after=2)
    binaries = make_binaries([100.0, 200.0], [1.0, 2.0])
    s_Nres, s_fg = th.serial_array_sort(binaries, FS)
    b_Nres, b_fg = th.block_array_sort(binaries, FS)
    assert int(s_Nres) == 0
    assert int(b_Nres) == 0
    assert_array_equal(s_fg, np.zeros(NF))
    assert_array_equal(b_fg, np.zeros(NF))


def test_zero_amplitude_padding_is_inert():
    ## a literal A = 0 source is indistinguishable from zero-padding: it must be
    ## neither counted as resolved nor added to the foreground
    th = make_thresher()
    Nres, fg = th.serial_array_sort(make_binaries([3.0, 3.0], [0.0, 20.0]), FS)
    assert int(Nres) == 1
    assert_array_equal(fg, np.zeros(NF))


def test_block_after_boundaries():
    with pytest.raises(ValueError):
        make_thresher(block_after=NF).block_array_sort(make_binaries([3.0], [1.0]), FS)
    with pytest.raises(TypeError):
        make_thresher(block_after=None).block_array_sort(make_binaries([3.0], [1.0]), FS)


# =============================================================================
# T19 -- realistic grid
# =============================================================================

def test_realistic_grid_alignment():
    fs = np.arange(1e-4, 5e-3, 1e-5)      # the PopModel default grid
    delf = fs[1] - fs[0]
    nf = len(fs)
    Sn = lisa_noise_psd(fs)
    rx = np.full(nf, 0.3)
    duration = 1.262e8

    ## digitize must be robust to np.arange's accumulated float error on the real grid
    assert_array_equal(np.digitize(fs, fs + 0.5 * delf), np.arange(nf))
    assert_array_equal(np.digitize(fs + 0.3 * delf, fs + 0.5 * delf), np.arange(nf))
    assert_array_equal(np.digitize(fs - 0.3 * delf, fs + 0.5 * delf), np.arange(nf))

    ## one source per bin, an order of magnitude below threshold everywhere
    A0 = 0.1 * 7 * np.sqrt(Sn.min() / (duration * rx[0]))
    binaries = np.array([fs, np.full(nf, A0)])

    th = SNR_Threshold(fs, Sn, rx, duration=duration, block_after=nf - 25)
    Nres, fg = th.serial_array_sort(binaries, fs)
    assert int(Nres) == 0
    assert np.all(fg > 0)
    assert_allclose(fg, A0**2 * rx[0], rtol=1e-12, atol=0.0)

    b_Nres, b_fg = th.block_array_sort(binaries, fs)
    assert int(b_Nres) == 0
    assert_array_equal(b_fg, fg)
