"""
Tests that Res_Astro_Likelihood bins resolved binaries on the same frequency
convention as SNR_Threshold, and reads the response / noise / foreground at the
binary's own bin.

Grids in play:
  self.fbins, self.lisa_rx, Sn   -- full length Nf, index k <-> fbins[k]
  Sgw (from PopModel.run_model)  -- length Nf-1,  index j <-> fbins[j+1]
so a binary in bin k must read lisa_rx[k], Sn[k] and Sgw[k-1].
"""
import os
import sys

os.environ["PELARGIR_BACKEND"] = "numpy"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from inference import Res_Astro_Likelihood
from thresholding import SNR_Threshold
from utils import get_amp_freq

G = 6.6743e-11
MSUN_KG = 1.98840987e30
KPC_M = 3.0856775814913673e19
AU_M = 1.495978707e11

FBINS = np.arange(1e-3, 3e-3, 1e-4)
NF = len(FBINS)
DELF = FBINS[1] - FBINS[0]
DURATION = 1.262e8


def thetas_at_frequencies(freqs, m1=0.6, m2=0.6, d_L=8.0):
    """Build a (Nres, 4) [m1, m2, d_L, a] array whose GW frequencies are `freqs`.

    Inverts fgw = (1/pi) * sqrt(G*M/a**3) from utils.get_amp_freq.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    M = (m1 + m2) * MSUN_KG
    a_m = (G * M / (np.pi * freqs) ** 2) ** (1.0 / 3.0)
    return np.column_stack([np.full(freqs.shape, m1),
                            np.full(freqs.shape, m2),
                            np.full(freqs.shape, d_L),
                            a_m / AU_M])


def make_likelihood(freqs, lisa_rx=None):
    rx = np.ones(NF) if lisa_rx is None else np.asarray(lisa_rx, dtype=float)
    return Res_Astro_Likelihood(np.random.default_rng(1),
                                thetas_at_frequencies(freqs),
                                FBINS, rx, duration=DURATION,
                                scatter=False, dynamic_scatter=False)


def test_get_phenom_matches_the_thresher_binning():
    ## bin centres and centre +/- 0.3*delf, plus below- and above-band sources.
    ## exact bin edges are deliberately avoided: the a <-> fgw round trip in
    ## get_amp_freq perturbs the frequency at the 1e-16 level.
    freqs = np.concatenate([FBINS,
                            FBINS + 0.3 * DELF,
                            FBINS - 0.3 * DELF,
                            [FBINS[0] - 3 * DELF, FBINS[-1] + 3 * DELF]])
    like = make_likelihood(freqs)

    ## the frequencies the likelihood actually sees, binned by the thresher
    th = SNR_Threshold(FBINS, np.ones(NF), np.ones(NF), duration=DURATION)
    phenom_fs = like.current_phenom[1, :]
    _, expected_idx = th.coarsegrain_bin(np.array([phenom_fs, np.ones_like(phenom_fs)]), FBINS)

    assert_array_equal(like.current_phenom_idx, expected_idx)
    ## sanity: the in-band members really do span bins 0..NF-1
    assert set(np.asarray(expected_idx[:NF]).tolist()) == set(range(NF))


def _p_res(like, Sn, Sgw, rho_thresh=7.0):
    return like.res_prob(Sgw[:, None, None], Sn, rho_thresh)


@pytest.mark.parametrize("k", [1, 3, NF - 1])
def test_res_prob_reads_the_foreground_at_the_binaries_own_bin(k):
    like = make_likelihood([FBINS[k]])
    assert int(like.current_phenom_idx[0]) == k

    A = like.current_phenom[0, 0]
    base = 1e-3 * DURATION * A**2 / 49.0     # rho >> 7 with this denominator
    Sn = np.full(NF, base)
    quiet = np.zeros(NF - 1)

    ## no spike -> resolved
    assert _p_res(like, Sn, quiet)[0, 0] == 1.0

    ## spike at Sgw[k-1] (the binary's own bin) -> suppressed
    loud = quiet.copy()
    loud[k - 1] = 1e30
    assert _p_res(like, Sn, loud)[0, 0] == 0.0

    ## spike at Sgw[k] (one bin up) -> must NOT touch it
    if k < NF - 1:
        off = quiet.copy()
        off[k] = 1e30
        assert _p_res(like, Sn, off)[0, 0] == 1.0


@pytest.mark.parametrize("k", [1, 3, NF - 1])
def test_res_prob_reads_the_noise_at_the_binaries_own_bin(k):
    like = make_likelihood([FBINS[k]])
    A = like.current_phenom[0, 0]
    base = 1e-3 * DURATION * A**2 / 49.0
    quiet = np.zeros(NF - 1)

    Sn = np.full(NF, base)
    Sn[k] = 1e30
    assert _p_res(like, Sn, quiet)[0, 0] == 0.0

    if k < NF - 1:
        Sn = np.full(NF, base)
        Sn[k + 1] = 1e30
        assert _p_res(like, Sn, quiet)[0, 0] == 1.0


@pytest.mark.parametrize("k", [1, 3, NF - 1])
def test_res_prob_reads_the_response_at_the_binaries_own_bin(k):
    freqs = [FBINS[k]]
    A = make_likelihood(freqs).current_phenom[0, 0]
    base = 1e-3 * DURATION * A**2 / 49.0
    Sn = np.full(NF, base)
    quiet = np.zeros(NF - 1)

    rx = np.ones(NF)
    rx[k] = 1e-30
    assert _p_res(make_likelihood(freqs, lisa_rx=rx), Sn, quiet)[0, 0] == 0.0

    if k < NF - 1:
        rx = np.ones(NF)
        rx[k + 1] = 1e-30
        assert _p_res(make_likelihood(freqs, lisa_rx=rx), Sn, quiet)[0, 0] == 1.0


def test_out_of_range_bins_are_clipped_not_wrapped():
    ## bin 0 is discarded by run_model and anything above the band has no model bin;
    ## both must give a finite probability rather than wrapping or raising
    like = make_likelihood([FBINS[0], FBINS[-1] + 5 * DELF])
    assert int(like.current_phenom_idx[0]) == 0
    assert int(like.current_phenom_idx[1]) == NF

    Sn = np.full(NF, 1e-40)
    Sgw = np.linspace(1e-40, 2e-40, NF - 1)
    p = _p_res(like, Sn, Sgw)
    assert p.shape == (2, 1)
    assert np.all(np.isfinite(p))
