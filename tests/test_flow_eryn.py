"""
Tests for joint Eryn PTMCMC on the flow toy model (flow_eryn.py) and the pieces it adds to
flow_inference.py: the split prior/likelihood, the Hessian and Laplace covariance, the Eryn wiring
in both tempering modes against FlowPosterior.ln_post, the moves, and short runs. Uses the small
posterior of tests/conftest.py.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("flax")
pytest.importorskip("eryn")

from pelargir import backend
jax = backend.import_jax()
import jax.numpy as jnp

from pelargir import flow_data as fd
from pelargir import flow_inference as fi
from pelargir import flow_eryn as fe

from conftest import near_truth_points


def test_prior_and_likelihood_split_sum_to_ln_post(post):
    th, y = near_truth_points(post, 4)
    total = np.asarray(post.ln_flow_prior(th, y)) + np.asarray(post.ln_data_res(th, y))
    assert_allclose(total, np.asarray(post.ln_post(th, y)), rtol=1e-12)
    t = post.terms(th, y)
    assert_allclose(np.asarray(post.ln_data_res(th, y)), np.asarray(t['data'] + t['res']), rtol=1e-12)


def test_hessian_and_laplace_covariance(post):
    th, y = (np.asarray(a)[0] for a in post.truth)
    H = post.hessian(th, y)
    n = 8 + y.size
    assert H.shape == (n, n) and np.allclose(H, H.T, rtol=1e-8, atol=1e-8)
    ## against central differences of the gradient
    x0 = np.concatenate([th, y])
    h = 1e-5*np.maximum(np.abs(x0), 1e-2)
    grad = lambda x: np.concatenate([np.asarray(g) for g in post.value_and_grad(x[:8], x[8:])[1]])
    for i in (0, 3, 6, 8, n - 1):
        e = np.zeros(n)
        e[i] = h[i]
        assert_allclose(H[i], (grad(x0 + e) - grad(x0 - e))/(2*h[i]), rtol=1e-3, atol=1e-3*np.abs(H[i]).max())
    cov, cov_th, cov_y = fi.laplace_covariance(post, th, y)
    for c in (cov, cov_th, cov_y):
        assert np.all(np.linalg.eigvalsh(c) > 0)
    ## no coordinate is wider than its cap
    cap = np.concatenate([[np.diff(fd.HYPERPRIOR[k].ppf([0.01, 0.99]))[0]/2 for k in fd.CONTEXT_NAMES], np.ones(y.size)])
    for c, sl in ((cov, slice(None)), (cov_th, slice(0, 8)), (cov_y, slice(8, None))):
        assert np.all(np.sqrt(np.diag(c)) <= cap[sl]*(1 + 1e-9))
    ## with caps far wider than the posterior, it is the plain inverse of -H (and of its blocks)
    P = -(H + H.T)/2
    wide = np.full(n, 1e6)
    if np.all(np.linalg.eigvalsh(P) > 1e-6):
        cov_w, cov_th_w, cov_y_w = fi.laplace_covariance(post, th, y, max_std=wide)
        assert_allclose(cov_w, np.linalg.inv(P), rtol=1e-6)
        assert_allclose(cov_y_w, np.linalg.inv(P[8:, 8:]), rtol=1e-6)


@pytest.mark.parametrize("tempering", fe.TEMPERINGS)
def test_eryn_target_at_beta_one_is_ln_post(post, tempering):
    model = fe.ErynFlowModel(post, tempering)
    th, y = near_truth_points(post, 5)
    offset = 0.0 if tempering == 'flow_prior' else model.box_log_density()
    assert_allclose(model.log_post(th, y), np.asarray(post.ln_post(th, y)) + offset, rtol=1e-12)
    ## the vectorized likelihood Eryn calls, and the prior it uses
    ll = model.log_like([th, y])
    assert ll.shape == (5,)
    if tempering == 'flow_prior':
        lp = model.priors['all_models_together'].logpdf({'pop': th[None, :, None], 'spectrum': y[None, :, None]}, None)
        assert lp.shape == (1, 5)


def test_hyperprior_container_matches_log_hyperprior(post):
    model = fe.ErynFlowModel(post)
    theta = fd.sample_context(np.random.default_rng(5), 30)
    assert_allclose(model.hyperprior_container().logpdf(theta), np.asarray(fi.log_hyperprior(jnp.asarray(theta))),
                    rtol=1e-10)
    draws = model.hyperprior_container().rvs(size=20)
    assert draws.shape == (20, 8) and np.all(np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(draws)))))


def test_initial_walkers_lie_in_the_support(post):
    start = tuple(np.asarray(a)[0] for a in post.truth)
    cov = fi.laplace_covariance(post, *start)[0]
    for strategy in ('truth', 'prior'):
        c = fe.initial_walkers(post, cov, start, 2, 6, strategy, seed=1)
        assert c['pop'].shape == (2, 6, 1, 8) and c['spectrum'].shape == (2, 6, 1, post.data.fs.size)
        assert np.all(np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(c['pop'].reshape(-1, 8))))))


@pytest.mark.parametrize("tempering", fe.TEMPERINGS)
def test_short_run(post, tempering, tmp_path):
    res = fe.run_eryn(post, 20, ntemps=2, nwalkers=8, tempering=tempering, init='truth', seed=4, progress=False)
    assert res.theta.shape == (20, 8, 8) and res.y.shape == (20, 8, post.data.fs.size)
    assert res.log_like.shape == (20, 2, 8) and res.betas.shape == (20, 2)
    assert np.all(np.isfinite(res.theta)) and np.all(np.isfinite(res.log_like))
    assert np.all(np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(res.theta.reshape(-1, 8))))))
    assert set(res.meta['move_acceptance_T1']) == {'joint', 'pop', 'spectrum', 'prior'}
    ## the chain moved
    assert np.any(res.theta[-1] != res.theta[0])
    res.save(tmp_path/"eryn.npz")
    loaded = fe.ErynResult.load(tmp_path/"eryn.npz")
    assert_array_equal(loaded.theta, res.theta)
    assert_array_equal(loaded.betas, res.betas)
    assert loaded.meta == res.meta
