"""
Tests for the flow toy-model inference layer (flow_inference.py): the toy dataset, the JAX hyperprior,
each log-posterior term against an independent reference (in y = log10(S_gw + C), with the pre-Stage-3
C = 0 posterior as the stage-2 formula), zero-foreground bins below log10 C, gradients against finite
differences, and batching. Uses a small toy dataset and a small randomized flax flow (not a trained
production flow).
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("flax")
pytest.importorskip("distrax")

from pelargir import backend
jax = backend.import_jax()
import jax.numpy as jnp
from flax import nnx

from pelargir.flows import data as fd
from pelargir.flows import flax_base as ff
from pelargir.inference import flow_inference as fi
from pelargir.inference import jax_likelihood as jl

from conftest import FBINS, LAMBDA, near_truth_points as near_truth

# =============================================================================
# Toy dataset
# =============================================================================

def test_toy_data_is_consistent_reproducible_and_round_trips(toy, tmp_path):
    assert_allclose(toy.data_psd, toy.truth_S + toy.noise_psd, rtol=1e-15)
    assert np.all(toy.truth_S > 0)
    ## the fixed state is every resolved binary in a modelled bin (bin 0 excluded)
    assert toy.theta_res.shape == (toy.nres_f.sum(), 4)
    consts, _ = fd._forward_consts(FBINS)
    A, f = (np.asarray(x) for x in fi.jp.amp_freq(jnp.asarray(toy.theta_res.T)))
    k = np.digitize(f, consts[0])
    assert np.all((k >= 1) & (k < FBINS.size))
    assert_array_equal(np.bincount(k - 1, minlength=toy.fs.size), toy.nres_f)
    assert_array_equal(toy.truth_theta, list(fi.FIDUCIAL) + [7.0, np.log10(LAMBDA)])
    again = fi.simulate_toy_data(fi.FIDUCIAL, 7.0, LAMBDA, FBINS, seed=3)
    assert_array_equal(again.nres_f, toy.nres_f)
    assert_array_equal(again.theta_res, toy.theta_res)
    assert_allclose(again.truth_S, toy.truth_S, rtol=1e-12)
    toy.save(tmp_path/"toy.npz")
    loaded = fi.ToyData.load(tmp_path/"toy.npz")
    for name in ('fbins', 'data_psd', 'noise_psd', 'nres_f', 'theta_res', 'truth_theta', 'truth_S'):
        assert_array_equal(getattr(loaded, name), getattr(toy, name))
    assert loaded.sigma == toy.sigma and loaded.meta == toy.meta


# =============================================================================
# Terms against references
# =============================================================================

def test_log_hyperprior_matches_scipy():
    rng = np.random.default_rng(2)
    theta = fd.sample_context(rng, 200)
    ## rows outside the support, one parameter at a time
    out = np.repeat(theta[:1], 4, axis=0)
    out[0, 0], out[1, 1], out[2, 6], out[3, 7] = 1.2, -0.1, 0.5, 8.0
    theta = np.concatenate([theta, out])
    ref = sum(fd.HYPERPRIOR[k].logpdf(theta[:, j]) for j, k in enumerate(fd.CONTEXT_NAMES))
    lp = np.asarray(fi.log_hyperprior(jnp.asarray(theta)))
    assert np.all(np.isneginf(lp[-4:])) and np.all(np.isneginf(ref[-4:]))
    assert_allclose(lp[:-4], ref[:-4], rtol=1e-12)


def test_terms_match_independent_references(post, toy):
    assert post.floor == fd.NOISE_FLOOR
    th, y = near_truth(post, 6)
    t = {k: np.asarray(v) for k, v in post.terms(th, y).items()}
    S = fd.unshift(y, post.floor)
    em = post.emulator
    ## flow density of the observed counts and S_gw, plus the Jacobian of y = log10(S_gw + C)
    flow = em.log_prob(th, toy.N_bands(em.bands), S) + np.sum(np.log((S + post.floor)*np.log(10)), axis=1)
    assert_allclose(t['flow'], flow, rtol=1e-10)
    data = -np.sum((np.log(toy.data_psd) - np.log(S + toy.noise_psd))**2, axis=1)/(2*toy.sigma**2)
    assert_allclose(t['data'], data, rtol=1e-12)
    assert_allclose(t['prior'], np.asarray(fi.log_hyperprior(jnp.asarray(th))), rtol=1e-12)
    assert_allclose(np.asarray(post.ln_post(th, y)), t['prior'] + t['flow'] + t['data'] + t['res'], rtol=1e-12)

    ## the step version is jax_likelihood.res_ln_like with one realization
    consts, bounds = fd._forward_consts(FBINS)
    edges, noise, rx = consts[:3]
    step = np.asarray(post.terms(th, y, width=None)['res'])
    for i in range(th.shape[0]):
        ref = jl.res_ln_like(jnp.asarray(toy.theta_res), jnp.asarray(th[i:i+1, :6]), jnp.asarray(S[i])[:, None, None],
                             jnp.asarray(noise), jnp.asarray(rx), jnp.asarray(edges), fd.DURATION, th[i, 6], bounds)
        assert_allclose(step[i], float(ref[0]), rtol=1e-12)
    ## the smoothed term approaches the step as the width shrinks, from below
    gaps = [np.max(np.abs(np.asarray(post.terms(th, y, width=w)['res']) - step)) for w in (0.3, 0.03, 0.003)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-2
    assert np.all(np.asarray(post.terms(th, y, width=0.1)['res']) <= step + 1e-12)


def test_v2_posterior_is_the_stage2_formula(post_v2, toy):
    '''With C = 0 the latent is log10 S_gw, and the flow term is the stage-2 density of S_gw times its Jacobian.'''
    assert post_v2.floor == 0.0
    th, y = near_truth(post_v2, 6)
    t = {k: np.asarray(v) for k, v in post_v2.terms(th, y).items()}
    S = 10**y
    em = post_v2.emulator
    flow = em.log_prob(th, toy.N_bands(em.bands), S) + np.sum(np.log(S*np.log(10)), axis=1)
    assert_allclose(t['flow'], flow, rtol=1e-12)
    data = -np.sum((np.log(toy.data_psd) - np.log(S + toy.noise_psd))**2, axis=1)/(2*toy.sigma**2)
    assert_allclose(t['data'], data, rtol=1e-12)
    assert_allclose(np.asarray(post_v2.truth[1]), np.log10(toy.truth_S)[None], rtol=1e-15)
    assert_allclose(post_v2.data_spectrum(), np.log10(toy.data_psd - toy.noise_psd), rtol=1e-12)


def test_bins_below_log10_C_are_zero_foreground(post, toy):
    '''y below log10 C is S_gw = 0: the data and resolved terms stop depending on it, and ln_post stays smooth there.'''
    th, y = near_truth(post, 1)
    lc = np.log10(post.floor)
    ## the highest-frequency bins, where the toy's resolved binaries are rare, pushed into and under the slab
    ys = np.repeat(y, 3, axis=0)
    ys[0, -3:] = lc - 0.1
    ys[1, -3:] = lc - 0.3
    ys[2, -3:] = lc - 2.0
    th3 = np.repeat(th, 3, axis=0)
    t = {k: np.asarray(v) for k, v in post.terms(th3, ys).items()}
    S0 = fd.unshift(ys, post.floor)
    assert np.all(S0[:, -3:] == 0)
    for k in ('data', 'res'):
        assert_allclose(t[k], t[k][0], rtol=1e-13)
    data = -np.sum((np.log(toy.data_psd) - np.log(S0[0] + toy.noise_psd))**2)/(2*toy.sigma**2)
    assert_allclose(t['data'][0], data, rtol=1e-12)
    ## the flow still prices the y below log10 C (that is how P(S_gw = 0) enters), and everything is finite
    assert np.all(np.isfinite(t['flow'])) and len(set(np.round(t['flow'], 8))) == 3
    v, (g_th, g_y) = post.value_and_grad(th3[1], ys[1])
    assert np.isfinite(float(v)) and np.all(np.isfinite(np.asarray(g_th))) and np.all(np.isfinite(np.asarray(g_y)))
    ## below log10 C only the flow term depends on y
    _, g_dr = jax.value_and_grad(lambda yy: fi._ln_data_res(post.consts, jnp.asarray(th3[1:2]), yy[None],
                                                            post.width)[0])(jnp.asarray(ys[1]))
    assert np.all(np.asarray(g_dr)[-3:] == 0) and np.any(np.asarray(g_dr)[:-3] != 0)


def test_truth_and_data_spectrum_in_y(post, toy):
    assert_allclose(np.asarray(post.truth[1])[0], np.log10(toy.truth_S + post.floor), rtol=1e-15)
    assert_allclose(post.data_spectrum(), np.log10(np.maximum(toy.data_psd - toy.noise_psd, 0) + post.floor),
                    rtol=1e-12)
    ## where the data sit at the noise (no foreground), the start spectrum is log10 C, i.e. S_gw = 0
    import dataclasses
    quiet = dataclasses.replace(toy, data_psd=np.where(np.arange(toy.fs.size) >= 8, toy.noise_psd, toy.data_psd))
    yq = fi.FlowPosterior(post.emulator, quiet).data_spectrum()
    assert_allclose(yq[8:], np.log10(post.floor), rtol=1e-15)
    assert np.all(fd.unshift(yq[8:], post.floor) == 0)


def test_outside_the_prior_the_posterior_is_minus_infinity(post):
    th, y = near_truth(post, 2)
    th[0, 1] = -0.05     ## m_sigma < 0
    th[1, 6] = 0.5       ## rho_thresh below RHO_MIN
    assert np.all(np.isneginf(np.asarray(post.ln_post(th, y))))


# =============================================================================
# Gradients and batching
# =============================================================================

def test_gradient_matches_central_differences(post):
    th, y = near_truth(post, 1)
    _, (g_th, g_y) = post.value_and_grad(th[0], y[0])
    x0 = np.concatenate([th[0], y[0]])
    h = np.concatenate([1e-6*np.maximum(np.abs(th[0]), 1e-2), np.full(y.shape[1], 1e-6)])
    X = np.concatenate([x0 + np.diag(h), x0 - np.diag(h)])
    lp = np.asarray(post.ln_post(X[:, :8], X[:, 8:]))
    fd_grad = (lp[:x0.size] - lp[x0.size:])/(2*h)
    ad = np.concatenate([np.asarray(g_th), np.asarray(g_y)])
    assert np.all(np.isfinite(ad)) and np.any(ad[:8] != 0)
    ## errors relative to the largest component (some components are near zero)
    assert np.max(np.abs(ad - fd_grad)) < 1e-5*np.max(np.abs(ad))


def test_batched_rows_match_single_rows(post):
    th, y = near_truth(post, 5)
    batch = np.asarray(post.ln_post(th, y))
    single = [float(post.ln_post(th[i:i+1], y[i:i+1])[0]) for i in range(5)]
    assert_allclose(batch, single, rtol=1e-12)
    v, _ = post.value_and_grad(th[2], y[2])
    assert_allclose(float(v), batch[2], rtol=1e-12)


def test_spectrum_plots_handle_zero_foreground_bins(post, toy, tmp_path):
    '''plot_spectrum (in S_gw) and plot_spectrum_x run with posterior zeros and zero-truth bins, and write their PNGs.'''
    import matplotlib
    matplotlib.use('Agg')
    from pelargir import plotting
    rng = np.random.default_rng(4)
    th, y = near_truth(post, 300)
    y[:150, -3:] = np.log10(post.floor) - 0.3*rng.uniform(size=(150, 3))
    S = fd.unshift(y, post.floor)
    truth_S = toy.truth_S.copy()
    truth_S[-2:] = 0.0
    plotting.plot_spectrum(tmp_path/'spectrum.png', toy.fs, S, truth_S, toy.data_psd, toy.noise_psd)
    plotting.plot_spectrum_x(tmp_path/'spectrum_x.png', toy.fs, y, fd.log10_shifted(truth_S, post.floor), post.floor)
    assert (tmp_path/'spectrum.png').stat().st_size > 0 and (tmp_path/'spectrum_x.png').stat().st_size > 0
    s = fi.spectrum_summary(S, truth_S)
    assert s['zero_truth_bins'] == 2 and 0.4 < s['p_zero_at_zero_truth'] <= 0.6
