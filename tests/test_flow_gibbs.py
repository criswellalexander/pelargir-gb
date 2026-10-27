"""
Tests for HMC-within-Gibbs on the flow toy model (flow_gibbs.py, flow_eryn.run_eryn with hmc): the
tempered conditional against Eryn's tempered posterior, the split y-Hessians and the Laplace mass
matrices, the numpyro move's bookkeeping, cache refresh and freeze, and short runs. Uses the small
posterior of tests/conftest.py.
"""
import os
import json

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("flax")
pytest.importorskip("eryn")
pytest.importorskip("numpyro")

from pelargir import backend
jax = backend.import_jax()
import jax.numpy as jnp

from pelargir import flow_inference as fi
from pelargir import flow_eryn as fe
from pelargir import flow_gibbs as fg
from eryn.ensemble import EnsembleSampler
from eryn.state import State

from conftest import near_truth_points

## bound the NUTS trees of the small random flow (its Laplace mass is only approximate)
SMALL = dict(max_tree_depth=5)


def _ln_cond(post, theta, y, beta, tempering, y_box=3.0):
    yd = fe.data_spectrum(post)
    box = (jnp.asarray(yd - y_box), jnp.asarray(yd + y_box))
    return float(fg._ln_conditional(*post._args(), jnp.asarray(theta), jnp.asarray(y), beta, box, tempering,
                                    post.slices, post.n_quad, post.width))


def _move(post, tempering='flow_prior', **kw):
    th, y = (np.asarray(a)[0] for a in post.truth)
    return fg.HMCSpectrumMove(post, *post.conditional_hessians(th, y), tempering=tempering, **{**SMALL, **kw})


def _sampler_state(post, move, tempering, ntemps, nwalkers, seed=0):
    '''An EnsembleSampler wired as run_eryn's, and a State near the truth with Eryn's prior and likelihood.'''
    model = fe.ErynFlowModel(post, tempering)
    nf = post.data.fs.size
    sampler = EnsembleSampler(nwalkers, {'pop': 8, 'spectrum': nf}, model.log_like, model.priors,
                              branch_names=list(fe.BRANCHES), nleaves_max={b: 1 for b in fe.BRANCHES},
                              tempering_kwargs=dict(ntemps=ntemps), moves=[move], vectorize=True)
    th, y = near_truth_points(post, ntemps*nwalkers, seed=seed, scale=0.01)
    state = State({'pop': th.reshape(ntemps, nwalkers, 1, 8), 'spectrum': y.reshape(ntemps, nwalkers, 1, nf)})
    m = sampler.get_model()
    return sampler, m, _evaluated(m, state)


def _evaluated(m, state):
    '''The state with Eryn's log prior and log likelihood at its coordinates.'''
    logp = m.compute_log_prior_fn(state.branches_coords, inds=state.branches_inds)
    logl, _ = m.compute_log_like_fn(state.branches_coords, inds=state.branches_inds, logp=logp)
    return State(state.branches_coords, log_like=logl, log_prior=logp, inds=state.branches_inds)


@pytest.mark.parametrize("tempering", fe.TEMPERINGS)
@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_conditional_is_eryns_tempered_posterior(post, tempering, beta):
    model = fe.ErynFlowModel(post, tempering)
    th, y = near_truth_points(post, 3)
    th = np.repeat(th[:1], 3, axis=0)
    if tempering == 'flow_prior':
        lp = model.priors['all_models_together'].logpdf({'pop': th[None, :, None], 'spectrum': y[None, :, None]}, None)[0]
    else:
        lp = model.hyperprior_container().logpdf(th) + model.box_container().logpdf(y)
    eryn = beta*model.log_like([th, y]) + lp
    ours = np.array([_ln_cond(post, th[i], y[i], beta, tempering) for i in range(3)])
    ## equal up to a theta-only constant
    assert_allclose(ours - ours[0], eryn - eryn[0], rtol=1e-9, atol=1e-8)


def test_conditional_hessians(post):
    th, y = (np.asarray(a)[0] for a in post.truth)
    Hp, Hl = post.conditional_hessians(th, y)
    assert_allclose(Hp + Hl, post.hessian(th, y)[8:, 8:], rtol=1e-8, atol=1e-8*np.abs(Hp + Hl).max())
    ## each part against central differences of its own gradient
    for H, fn in ((Hp, lambda v: post.ln_flow_prior(th[None], v[None])[0]),
                  (Hl, lambda v: post.ln_data_res(th[None], v[None])[0])):
        grad = jax.grad(fn)
        h = 1e-5
        for i in (0, y.size - 1):
            e = np.zeros(y.size)
            e[i] = h
            fd_row = (np.asarray(grad(jnp.asarray(y + e))) - np.asarray(grad(jnp.asarray(y - e))))/(2*h)
            assert_allclose(H[i], fd_row, rtol=1e-3, atol=1e-3*np.abs(H[i]).max())


def test_laplace_inverse_mass(post):
    th, y = (np.asarray(a)[0] for a in post.truth)
    Hp, Hl = post.conditional_hessians(th, y)
    betas = np.array([1.0, 0.5, 0.1, 0.0])
    for tempering in fe.TEMPERINGS:
        M = fg.laplace_inverse_mass(Hp, Hl, betas, tempering)
        assert M.shape == (4, y.size, y.size)
        assert all(np.all(np.linalg.eigvalsh(m) > 0) for m in M)
        ## no bin wider than the 1 dex cap
        assert np.all(np.sqrt(np.einsum('tii->ti', M)) <= 1 + 1e-9)
    ## 'all' at beta = 0 is flat: the cap alone
    assert_allclose(fg.laplace_inverse_mass(Hp, Hl, [0.0], 'all')[0], np.eye(y.size), atol=1e-12)
    ## with wide caps, the plain inverse of the tempered -H
    P = -(Hp + Hl)
    if np.all(np.linalg.eigvalsh((P + P.T)/2) > 1e-6):
        wide = lambda b, t: fg.laplace_inverse_mass(Hp, Hl, [b], t, cap=1e6)[0]
        assert_allclose(wide(1.0, 'flow_prior'), np.linalg.inv(P), rtol=1e-6)
        assert_allclose(wide(0.5, 'all'), 2*np.linalg.inv(P), rtol=1e-6)


@pytest.mark.parametrize("kernel", fg.KERNELS)
def test_move_bookkeeping(post, kernel):
    move = _move(post, kernel=kernel)
    _, m, state = _sampler_state(post, move, 'flow_prior', 1, 8)
    th0, y0 = state.branches_coords['pop'].copy(), state.branches_coords['spectrum'].copy()
    new, moved = move.propose(m, state)
    ## theta untouched, most spectra moved, and Eryn's prior and likelihood up to date
    assert_array_equal(new.branches_coords['pop'], th0)
    assert moved.shape == (1, 8) and moved.any()
    assert_array_equal(moved, np.any(new.branches_coords['spectrum'] != y0, axis=(2, 3)))
    fresh = _evaluated(m, new)
    assert_allclose(new.log_like, fresh.log_like, rtol=1e-12)
    assert_allclose(new.log_prior, fresh.log_prior, rtol=1e-12)
    d = move.diagnostics()
    assert d['calls'] == 1 and len(d['accept_prob']) == 1 and np.all(np.isfinite(d['step_size_median']))


def test_mass_matrix_and_cache_follow_the_state(post):
    move = _move(post, 'all')
    _, m, state = _sampler_state(post, move, 'all', 2, 4)
    state, _ = move.propose(m, state)
    ## the overwritten Laplace matrices and numpyro's square roots agree (inverse mass = U U^T, sqrt = U^-T)
    a = move.states.adapt_state
    inv, sq, sq_inv = (np.asarray(x) for x in (a.inverse_mass_matrix, a.mass_matrix_sqrt, a.mass_matrix_sqrt_inv))
    betas = np.asarray(move.temperature_control.betas)
    assert_allclose(inv, np.repeat(move.inverse_mass(betas), 4, axis=0), rtol=1e-12)
    assert_allclose(np.swapaxes(sq_inv, 1, 2) @ sq_inv, inv, rtol=1e-8, atol=1e-12)
    assert_allclose(sq @ np.swapaxes(sq, 1, 2), np.linalg.inv(inv), rtol=1e-6, atol=1e-8)
    ## change theta behind the move's back: the next call's potential is that of the new theta
    th = state.branches_coords['pop']*(1 + 0.01*np.random.default_rng(3).standard_normal(state.branches_coords['pop'].shape))
    state = _evaluated(m, State({'pop': th, 'spectrum': state.branches_coords['spectrum']}, inds=state.branches_inds))
    ## copies: Eryn's temperature swaps rearrange the state's arrays in place
    th_f = np.array(state.branches_coords['pop'][:, :, 0].reshape(-1, 8))
    b_f = np.repeat(np.asarray(move.temperature_control.betas), 4)
    move.propose(m, state)
    z = np.asarray(move.states.z)
    pe = np.asarray(move.states.potential_energy)
    expected = np.array([-_ln_cond(post, th_f[i], z[i], b_f[i], 'all') for i in range(8)])
    assert_allclose(pe, expected, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("adapt", fg.ADAPTS)
def test_freeze_stops_adaptation(post, adapt):
    move = _move(post, adapt=adapt, num_warmup=6)
    _, m, state = _sampler_state(post, move, 'flow_prior', 1, 4)
    for _ in range(3):
        state, _ = move.propose(m, state)
    step = np.asarray(move.states.adapt_state.step_size)
    assert not np.allclose(step, fg.INITIAL_STEP)
    move.freeze()
    frozen = np.asarray(move.states.adapt_state.step_size).copy()
    inv = np.asarray(move.states.adapt_state.inverse_mass_matrix).copy()
    for _ in range(2):
        state, _ = move.propose(m, state)
    assert_array_equal(np.asarray(move.states.adapt_state.step_size), frozen)
    if adapt == 'numpyro':
        assert_array_equal(np.asarray(move.states.adapt_state.inverse_mass_matrix), inv)
    assert move.burn_diagnostics['calls'] == 3 and move.diagnostics()['calls'] == 2


@pytest.mark.parametrize("tempering,hmc", [('flow_prior', {}), ('all', {}),
                                           ('flow_prior', dict(kernel='hmc', adapt='numpyro'))])
def test_short_gibbs_run(post, tempering, hmc, tmp_path):
    res = fe.run_eryn(post, 10, ntemps=2, nwalkers=8, tempering=tempering, init='truth', burn=5, seed=2,
                      progress=False, hmc={**SMALL, **hmc})
    assert res.theta.shape == (10, 8, 8) and res.y.shape == (10, 8, post.data.fs.size)
    assert np.all(np.isfinite(res.theta)) and np.all(np.isfinite(res.y)) and np.all(np.isfinite(res.log_like))
    assert np.all(np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(res.theta.reshape(-1, 8))))))
    assert set(res.meta['move_acceptance_T1']) == {'joint', 'pop', 'prior', 'hmc'}
    assert res.meta['settings']['move_weights'] == fg.GIBBS_WEIGHTS
    h = res.meta['hmc']
    assert h['sampling']['frozen'] and h['sampling']['settings']['kernel'] == hmc.get('kernel', 'nuts')
    assert len(h['sampling']['step_size_median']) == 2
    res.save(tmp_path/"gibbs.npz")
    loaded = fe.ErynResult.load(tmp_path/"gibbs.npz")
    assert_array_equal(loaded.y, res.y)
    ## (a move never proposed at T = 1 has a NaN acceptance, so compare the serialized form)
    assert json.dumps(loaded.meta) == json.dumps(res.meta)
