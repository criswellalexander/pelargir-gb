"""
Grid/shape guards for PopModel. These pin down the invariant that the foreground
spectrum, its frequency array and the noise PSD it is added to all live on
fbins[1:], without running an inference.
"""
import os
import sys

os.environ["PELARGIR_BACKEND"] = "numpy"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from models import PopModel
from utils import lisa_noise_psd

FBINS = np.arange(1e-4, 5e-4, 2e-5)
NF = len(FBINS)
NTOT = int(1e4)


@pytest.fixture(scope="module")
def popmodel():
    return PopModel(NTOT, np.random.default_rng(170817), fbins=FBINS, Nreal=2, block_after=4)


@pytest.fixture(scope="module")
def model_draw(popmodel):
    return popmodel.run_model()


@pytest.fixture(scope="module")
def data_draw():
    """A single-realization draw standing in for the observed dataset (1-D spectrum)."""
    datamodel = PopModel(NTOT, np.random.default_rng(150914), fbins=FBINS, Nreal=1, block_after=4)
    fs, fg_psd, N_res = datamodel.run_model()
    return fs, fg_psd, N_res


def test_run_model_returns_matching_frequency_and_spectrum_lengths(model_draw):
    fs, fg_psd, N_res = model_draw
    assert len(fs) == NF - 1
    assert fg_psd.shape[0] == NF - 1
    assert_allclose(fs, FBINS[1:], rtol=0, atol=0)


def test_return_spec_astro_info_is_on_the_same_grid(popmodel, data_draw):
    fs, fg_data, N_res_data = data_draw
    popmodel.construct_fg_likelihood(fg_data, np.array(0.1))
    popmodel.construct_Nres_likelihood(N_res_data)

    ln_p, astro_info = popmodel.fg_N_ln_prob(popmodel.hyperprior.sample(1), return_spec=True)
    assert np.isfinite(ln_p).all()
    assert len(astro_info[0]) == NF - 1
    assert astro_info[1].shape[0] == NF - 1
    assert len(astro_info[0]) == astro_info[1].shape[0]


def test_default_noise_psd_is_on_the_spectrum_grid(popmodel, data_draw):
    fs, fg_data, N_res_data = data_draw
    popmodel.construct_fg_likelihood(fg_data, np.array(0.1))
    assert popmodel.fg_like.noise_psd.shape[0] == NF - 1
    assert_allclose(popmodel.fg_like.noise_psd, lisa_noise_psd(FBINS[1:]), rtol=0, atol=0)


def test_naive_prefilter_does_not_change_output_shapes():
    """Light shape-contract guard: turning use_naive_prefilter on/off must not change
    fs/fg_psd shapes (full numeric exact-match is covered in
    tests/test_naive_snr_prefilter.py)."""
    pm_off = PopModel(NTOT, np.random.default_rng(170817), fbins=FBINS, Nreal=2,
                       block_after=4, use_naive_prefilter=False)
    pm_on = PopModel(NTOT, np.random.default_rng(170817), fbins=FBINS, Nreal=2,
                      block_after=4, use_naive_prefilter=True)

    fs_off, fg_off, Nres_off = pm_off.run_model()
    fs_on, fg_on, Nres_on = pm_on.run_model()

    assert fs_on.shape == fs_off.shape == (NF - 1,)
    assert fg_on.shape == fg_off.shape
    assert Nres_on.shape == Nres_off.shape


def test_explicit_noise_psd_length_is_checked(popmodel, data_draw):
    fs, fg_data, N_res_data = data_draw
    ## the full-length grid is exactly the mistake the guard is there to catch
    ## match the message: without the guard this still raises, but as an opaque
    ## broadcast error from deep inside FG_Likelihood
    with pytest.raises(ValueError, match="noise_psd has leading dimension"):
        popmodel.construct_fg_likelihood(fg_data, np.array(0.1), noise_psd=lisa_noise_psd(FBINS))
    ## the correct grid must still be accepted
    popmodel.construct_fg_likelihood(fg_data, np.array(0.1), noise_psd=lisa_noise_psd(FBINS[1:]))
    assert popmodel.fg_like.noise_psd.shape[0] == NF - 1
