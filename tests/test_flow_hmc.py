"""
Tests for joint NUTS on the flow toy model (flow_hmc.py): the numpyro model's log joint against
FlowPosterior.ln_post, the hyperprior distributions, the three initialization strategies, a short
reproducible NUTS run, HMCResult save/load, and flow_posterior.quadrature_errors. Uses the small
posterior of tests/conftest.py.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("flax")
pytest.importorskip("numpyro")

from pelargir import backend
jax = backend.import_jax()
import jax.numpy as jnp
from numpyro.infer.util import log_density, potential_energy, constrain_fn

from pelargir.flows import data as fd
from pelargir.inference import flow_posterior as fi
from pelargir.inference import flow_hmc as fh

from conftest import near_truth_points


def test_prior_distributions_match_log_hyperprior():
    theta = fd.sample_context(np.random.default_rng(4), 50)
    priors = fh.prior_distributions()
    lp = sum(np.asarray(priors[name].log_prob(jnp.asarray(theta[:, j]))) for j, name in enumerate(fd.CONTEXT_NAMES))
    assert_allclose(lp, np.asarray(fi.log_hyperprior(jnp.asarray(theta))), rtol=1e-12)


def test_model_log_joint_is_ln_post(post):
    model = fh.make_model(post)
    th, y = near_truth_points(post, 3)
    for i in range(3):
        lj, _ = log_density(model, (fh.model_args(post),), {}, fh._as_sites(th[i], y[i]))
        assert_allclose(float(lj), float(post.ln_post(th[i:i+1], y[i:i+1])[0]), rtol=1e-12)


@pytest.mark.parametrize("strategy", fh.INIT_STRATEGIES)
def test_initial_values(post, strategy):
    model = fh.make_model(post)
    init, info = fh.initial_values(post, model, strategy, num_chains=3, seed=2, map_steps=100)
    assert set(init) == set(fd.CONTEXT_NAMES) | {'y'}
    assert all(v.shape[0] == 3 for v in init.values()) and init['y'].shape == (3, post.data.fs.size)
    ## the jitter makes the chains distinct
    assert np.unique(np.asarray(init['m_mu'])).size == 3
    for c in range(3):
        u = {k: v[c] for k, v in init.items()}
        assert np.isfinite(float(potential_energy(model, (fh.model_args(post),), {}, u)))
        theta, _ = fh._from_sites(constrain_fn(model, (fh.model_args(post),), {}, u))
        assert np.isfinite(float(fi.log_hyperprior(jnp.asarray(theta[None]))[0]))
    if strategy == 'map':
        start = float(post.ln_post(np.asarray(info['start_theta'])[None], np.asarray(info['start_y'])[None])[0])
        found = float(post.ln_post(np.asarray(info['theta'][0])[None], np.asarray(info['y'][0])[None])[0])
        assert found > start


def test_short_nuts_run_is_reproducible_and_round_trips(post, tmp_path):
    kw = dict(num_warmup=15, num_samples=10, num_chains=2, init='truth', seed=3, progress=False)
    res = fh.run_nuts(post, **kw)
    again = fh.run_nuts(post, **kw)
    assert res.theta.shape == (2, 10, 8) and res.y.shape == (2, 10, post.data.fs.size)
    assert np.all(np.isfinite(res.theta)) and np.all(np.isfinite(res.y))
    assert set(res.extra) == set(fh.EXTRA_FIELDS) and res.extra['num_steps'].shape == (2, 10)
    assert {'n_eff', 'r_hat', 'mean'} <= set(res.meta['summary']['m_mu'])
    ## every draw is inside the prior's support
    assert np.all(np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(res.theta.reshape(-1, 8))))))
    assert_allclose(again.theta, res.theta, rtol=1e-10)
    res.save(tmp_path/"hmc.npz")
    loaded = fh.HMCResult.load(tmp_path/"hmc.npz")
    assert_array_equal(loaded.theta, res.theta)
    assert_array_equal(loaded.y, res.y)
    assert all(np.array_equal(loaded.extra[k], res.extra[k]) for k in res.extra)
    assert loaded.meta == res.meta


def test_quadrature_errors_vanish_at_the_reference(post):
    th, y = near_truth_points(post, 5)
    err = fi.quadrature_errors(post, th, y, n_quads=(2, 8, 64), ref=64, n_grad=2)
    assert err[64]['value'] == 0.0 and err[64]['grad'] == 0.0
    assert err[2]['value'] >= err[8]['value'] > 0
