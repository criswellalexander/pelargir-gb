"""
Tests for the flow emulator (flows.py) and the per-galaxy rho_thresh of the JAX forward model:
per-galaxy thresholds against the thresholder on the materialized draws, hyperpriors, bands,
training sets, transforms, the Gauss-Legendre count marginalization, and save/load.

Runs under any backend (the forward-model reference is jax_thresholding, itself tested against
serial_array_sort); needs torch and zuko.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("jax")
pytest.importorskip("zuko")
import torch
import jax
import jax.numpy as jnp

from pelargir import backend
from pelargir import flows
from pelargir import jax_population as jp
from pelargir import jax_thresholding as jt
from pelargir.inference import GalacticBinaryPrior, PopulationHyperPrior
from pelargir.utils import get_amp_freq, lisa_noise_psd

xp = backend.xp
FIDUCIAL = np.array([0.6, 0.15, 3.31, 0.75, 0.33, 0.5])
FBINS = flows.model_fbins(1e-4, 6e-4, 2e-5)
DELF = FBINS[1] - FBINS[0]
BOUNDS = jp.prior_bounds(GalacticBinaryPrior(np.random.default_rng(0)))


def forward_consts():
    import legwork as lw
    import astropy.units as u
    rx = lw.psd.approximate_response_function(FBINS*u.Hz, 19.09*u.mHz).value
    return FBINS + 0.5*DELF, lisa_noise_psd(FBINS, cpu=True), rx


# =============================================================================
# Per-galaxy rho_thresh in the forward model
# =============================================================================

@pytest.mark.parametrize("cut", [None, 1.0])
def test_per_galaxy_rho_matches_the_thresholder_on_the_same_draws(cut):
    edges, Sn, rx = forward_consts()
    thetas = np.stack([FIDUCIAL, [0.8, 0.1, 5.0, 0.5, 0.6, 1.0], [0.4, 0.2, 2.0, 1.0, 0.2, 0.0]])
    rhos = np.array([5.0, 7.0, 9.5])
    Ns = np.array([[30000, 20000, 25000], [22000, 31000, 18000]])
    key = jax.random.key(3)
    Nres, fg, mask, Nres_f = jp.jax_forward_model(key, thetas, Ns, BOUNDS, edges, Sn, rx, flows.DURATION, 1/DELF,
                                                  snr_thresh=rhos, batch_size=4, prefilter_snr=cut,
                                                  return_mask=True, return_nres_f=True)
    assert_array_equal(Nres_f[1:].sum(axis=0), Nres)
    Nr, Np = Ns.shape
    for r in range(Nr):
        for p in range(Np):
            g = r*Np + p
            draw = jp.sample_galaxy_draw(key, g, thetas[p], Ns[r, p], BOUNDS, out_module=xp)
            A, f = get_amp_freq(draw)
            r_N, r_fg, r_mask = jt.jax_threshold(xp.array([f, A]), edges, Sn, rx, flows.DURATION, 1/DELF,
                                                 snr_thresh=rhos[p], return_mask=True, prefilter_snr=None)
            assert int(Nres[r, p]) == int(jp._host(r_N))
            assert_allclose(fg[:, r, p], jp._host(r_fg), rtol=1e-12, atol=0)
            assert_array_equal(mask[:Ns[r, p], r, p], jp._host(r_mask))


def test_prefilter_above_the_smallest_rho_raises():
    edges, Sn, rx = forward_consts()
    with pytest.raises(ValueError, match="must not exceed"):
        jp.jax_forward_model(jax.random.key(0), np.stack([FIDUCIAL]*2), np.full((1, 2), 1000), BOUNDS, edges, Sn, rx,
                             flows.DURATION, 1/DELF, snr_thresh=np.array([7.0, 0.5]), prefilter_snr=1.0)


# =============================================================================
# Hyperpriors, bands
# =============================================================================

def test_hyperprior_matches_PopulationHyperPrior():
    ref = PopulationHyperPrior(xp.random.default_rng(0)).hyperprior_dict
    assert flows.CONTEXT_NAMES[:flows.N_POP] == GalacticBinaryPrior(None).pop_params == list(ref)
    for k, d in ref.items():
        mine = flows.HYPERPRIOR[k]
        if hasattr(d, 'loc'):
            assert (float(d.loc), float(d.scale)) == (mine.kwds['loc'], mine.kwds['scale'])
        else:
            assert float(d.a) == mine.args[0] and mine.kwds.get('scale', 1.0) == 1.0


def test_new_priors():
    c = flows.sample_context(np.random.default_rng(1), 20000)
    rho, loglam = c[:, 6], c[:, 7]
    assert rho.min() >= flows.RHO_MIN
    assert abs(np.median(rho) - 8) < 0.1
    assert np.log10(5e5) <= loglam.min() and loglam.max() <= np.log10(5e7)


def test_bands_partition_the_grid():
    fs = flows.model_fbins()[1:]
    bands = flows.make_bands(fs)
    assert [b.nf for b in bands] == [5]*9
    assert np.concatenate([np.arange(fs.size)[b.slice] for b in bands]).tolist() == list(range(fs.size))
    bands = flows.make_bands(fs, edges=[1e-4, 3e-4, 1e-3])
    assert [b.nf for b in bands] == [10, 35]
    assert_allclose(np.concatenate([b.fs for b in bands]), fs)
    with pytest.raises(ValueError, match="no bins"):
        flows.make_bands(fs, edges=[1e-4, 1.05e-4, 1e-3])


# =============================================================================
# Training sets
# =============================================================================

@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    ## lambda_tot ~ 1e5-2e5 keeps the set cheap while every bin still has unresolved binaries
    saved = flows.HYPERPRIOR['log10_lambda_tot']
    import scipy.stats as ss
    flows.HYPERPRIOR['log10_lambda_tot'] = ss.uniform(loc=5.0, scale=0.3)
    try:
        ts = flows.draw_training_set(24, 3, FBINS, seed=5, chunk=10, progress=False)
        ts2 = flows.draw_training_set(24, 3, FBINS, seed=5, chunk=10, progress=False)
    finally:
        flows.HYPERPRIOR['log10_lambda_tot'] = saved
    path = tmp_path_factory.mktemp("ts")/"ts.npz"
    ts.save(path)
    return ts, ts2, flows.TrainingSet.load(path)


def test_training_set_shapes_reproducibility_and_bands(small_set):
    ts, ts2, loaded = small_set
    Nf = FBINS.size - 1
    assert ts.context.shape == (72, 8) and ts.nres_f.shape == (72, Nf) and ts.psd.shape == (72, Nf)
    assert_array_equal(ts.context[::3], ts.context[1::3])      ## realizations share their draw
    assert_array_equal(ts.nres_f, ts2.nres_f)
    ## foreground sums use an atomic segment_sum (~1e-16 run to run)
    assert_allclose(ts.psd, ts2.psd, rtol=1e-12, atol=0)
    assert np.all(ts.psd > 0)
    assert abs(ts.ntot.mean()/np.mean(10**ts.context[:, 7]) - 1) < 0.01
    for k in ('context', 'nres_f', 'psd', 'ntot', 'fbins'):
        assert_array_equal(getattr(loaded, k), getattr(ts, k))
    for b in flows.make_bands(ts.fs, bins_per_band=4):
        c, N, S = flows.band_view(ts, b)
        assert_array_equal(N, ts.nres_f[:, b.start:b.stop].sum(axis=1))
        assert S.shape == (72, b.nf)


def test_chunked_runs_resume_to_the_same_set(tmp_path, monkeypatch):
    import scipy.stats as ss
    monkeypatch.setitem(flows.HYPERPRIOR, 'log10_lambda_tot', ss.uniform(loc=5.0, scale=0.3))
    kw = dict(chunk=4, progress=False, chunk_dir=tmp_path/"chunks")
    full = flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    os.remove(tmp_path/"chunks"/"chunk_00001.npz")
    resumed = flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    assert resumed.meta['simulated_draws'] == 4
    assert_array_equal(resumed.nres_f, full.nres_f)
    assert_allclose(resumed.psd, full.psd, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="different settings"):
        flows.draw_training_set(12, 2, FBINS, seed=8, **kw)


# =============================================================================
# Transforms
# =============================================================================

def test_band_transform_round_trip_and_jacobian():
    rng = np.random.default_rng(2)
    N = rng.integers(0, 500, 200)
    S = 10**rng.uniform(-41, -37, (200, 3))
    t = flows.BandTransform(3).fit(N, S, rng)
    u = rng.uniform(size=200)
    z, logdet = t(torch.as_tensor(N + u), torch.as_tensor(S))
    N_back, S_back = t.inverse(z)
    assert_array_equal(N_back.numpy(), N)
    assert_allclose(S_back.numpy(), S, rtol=1e-12)
    f = lambda x: t(x[:1], x[1:][None])[0][0]
    for i in range(3):
        x = torch.cat([torch.as_tensor([N[i] + u[i]]), torch.as_tensor(S[i])]).to(torch.float64)
        J = torch.autograd.functional.jacobian(f, x)
        assert_allclose(logdet[i].item(), torch.logdet(J).item(), rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        flows.BandTransform(3).fit(N, np.where(S > 1e-40, S, 0.0), rng)


# =============================================================================
# Count marginalization and save/load
# =============================================================================

@pytest.fixture(scope="module")
def toy_flow():
    '''A flow fitted to N ~ Poisson(lam(c)), log10 S ~ N(-38 + 0.3 c, 0.05), c ~ U(0,1).'''
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 20000
    c = rng.uniform(size=(n, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    S = 10**(-38 + 0.3*c + 0.05*rng.standard_normal((n, 1)))
    band = flows.FrequencyBand(0, 0, 1, np.array([1e-4]))
    flow = flows.BandFlow(band, n_context=1)
    flows.train_band_flow(flow, c, N, S, n_epochs=6, batch_size=256, lr=3e-3, seed=0, progress=False)
    return flow.eval()


def test_count_quadrature_converges_and_recovers_the_joint_density(toy_flow):
    import scipy.stats as ss
    rng = np.random.default_rng(9)
    c = rng.uniform(size=(300, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    logS = -38 + 0.3*c[:, 0] + 0.05*rng.standard_normal(300)
    S = 10**logS[:, None]
    with torch.no_grad():
        lp = {n: toy_flow.log_prob(c, N, S, n_quad=n).numpy() for n in (8, 32, 64, 512)}
    ## the spline flow's integrand has kinks, so Gauss-Legendre converges only algebraically
    ## (~n^-2 in the worst rows); most rows converge to float32 precision by 8 nodes
    err = {n: np.max(np.abs(lp[n] - lp[512])) for n in (8, 32, 64)}
    assert err[8] > err[32] > err[64]
    assert err[32] < 1e-4 and err[64] < 2e-5
    assert np.median(np.abs(lp[8] - lp[512])) < 1e-5
    ## the analytic joint density of (N, S), S in Hz^-1: Poisson x log-normal
    exact = ss.poisson.logpmf(N, lam) + ss.norm.logpdf(logS, -38 + 0.3*c[:, 0], 0.05) - np.log(S[:, 0]*np.log(10))
    assert abs(np.mean(lp[64] - exact)) < 0.05
    assert np.std(lp[64] - exact) < 0.2


def test_emulator_save_load_reproduces_log_prob(tmp_path, small_set):
    ts = small_set[0]
    em, _ = flows.train_emulator(ts, bands=flows.make_bands(ts.fs, bins_per_band=10), n_epochs=1, progress=False)
    N = np.stack([flows.band_view(ts, b)[1] for b in em.bands], axis=1)
    lp = em.log_prob(ts.context[:5], N[:5], ts.psd[:5])
    em.save(tmp_path)
    em2 = flows.BandedFlowEmulator.load(tmp_path)
    assert_array_equal(em2.log_prob(ts.context[:5], N[:5], ts.psd[:5]), lp)
    Ns, Ss = em2.sample(ts.context[:4])
    assert Ns.shape == (4, len(em.bands)) and np.all(np.isfinite(Ss)) and np.all(Ss > 0)
