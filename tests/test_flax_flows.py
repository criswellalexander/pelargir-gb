"""
Tests for the flax flow base (flows/flax_base.py): the exact density of an untrained flow (which is its
base distribution, so log P(N, S) is analytic, including the mass of a zero-foreground bin's slab),
transforms and Jacobians in x = log10(S_gw + C) with zeros, the count marginalization on a toy
problem, the scrambled-Sobol integral for rows with zeros, the density in x for inference, training
with zero bins, autodiff of the conditional density (against finite differences, and under jit /
vmap / an outer jit), save/load (v3, and v2 as C = 0), and API parity with the zuko base. Runs under
any backend; needs flax, distrax and optax.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
import scipy.stats as ss
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("flax")
pytest.importorskip("distrax")
pytest.importorskip("optax")

from pelargir import backend
jax = backend.import_jax()
import jax.numpy as jnp
from flax import nnx

from pelargir.flows import data as fd
from pelargir.flows import flax_base as ff

FBINS = fd.model_fbins(1e-4, 3e-4, 2e-5)   ## 10 modelled bins
NF = FBINS.size - 1
## small architectures for tests that train or differentiate (the defaults are production-sized)
SMALL_FLAX = dict(flow_num_layers=4, hidden_size=16)
SMALL_ZUKO = dict(transforms=3, hidden_size=16)


def synthetic_set(n=2000, seed=0):
    '''A TrainingSet with Poisson counts and log-normal spectra that depend on the context.'''
    rng = np.random.default_rng(seed)
    c = fd.sample_context(rng, n)
    lam = 5 + 40*(c[:, 0] - 0.2)/0.9                               ## depends on m_mu
    nres_f = rng.poisson(lam[:, None]*np.ones(NF)/NF + 1)
    psd = 10**(-38 + 0.3*c[:, 5:6] + 0.05*rng.standard_normal((n, NF)))
    return fd.TrainingSet(c, nres_f.astype(np.int32), psd, np.full(n, 10**6), FBINS, {})


def zero_set(n=600, seed=0, frac=0.3):
    '''synthetic_set with spectra near the noise floor and a fraction frac of the S_gw values exactly 0 (high-f-like).'''
    ts = synthetic_set(n, seed)
    rng = np.random.default_rng(seed + 100)
    ts.psd[:] = 10**(-40.6 + 0.5*rng.standard_normal(ts.psd.shape))
    ts.psd[rng.uniform(size=ts.psd.shape) < frac] = 0.0
    return ts


def fitted_emulator(ts, bins_per_band=5, floor=fd.NOISE_FLOOR, **kw):
    '''An untrained emulator whose transforms (floor C) are fitted to ts.'''
    em = ff.BandedFlowEmulator(fd.make_bands(ts.fs, bins_per_band=bins_per_band), ts.fbins, **kw)
    rng = np.random.default_rng(1)
    for f in em.flows:
        c, N, S = fd.band_view(ts, f.band)
        f.fit_transforms(c, N, S, rng, floor=floor)
    return em


def randomize(model, key, scale=0.05):
    '''Replace every parameter with small random values, so the flow is not the identity.'''
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(key, len(leaves))
    new = [scale*jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, new))


def band_arrays(ts, em, rows):
    N = np.stack([fd.band_view(ts, b)[1] for b in em.bands], axis=1)[rows].astype(np.float64)
    return jnp.asarray(ts.context[rows]), jnp.asarray(N), jnp.asarray(ts.psd[rows])


# =============================================================================
# Exactness of the untrained flow, transforms
# =============================================================================

def _y(t, x, i):
    '''Flow-space coordinate i (0 = the count) of log10-space value x under transform t.'''
    return (x - float(t.loc[i]))/float(t.scale[i])*float(t.spline_scale[i])


@pytest.mark.parametrize("floor", [0.0, fd.NOISE_FLOOR])
def test_untrained_flow_log_prob_is_analytic(floor):
    '''
    An untrained flow is its N(0, I) base, so P(N, S) = [Phi(y0(N+1)) - Phi(y0(N))] prod_i phi(y_i) |dy_i/dS_i|,
    y_i from x_i = log10(S_i + C).
    '''
    ts = synthetic_set()
    em = fitted_emulator(ts, floor=floor)
    rows = np.arange(40)
    c, N, S = band_arrays(ts, em, rows)
    for j, f in enumerate(em.flows):
        t = f.transform
        assert float(t.floor) == floor
        Nj, Sj = np.asarray(N[:, j]), np.asarray(S[:, f.band.slice])
        assert Nj.min() >= 1
        y0 = lambda n: _y(t, np.log10(n), 0)
        y = (np.log10(Sj + floor) - t.loc[1:])/t.scale[1:]*t.spline_scale[1:]
        exact = (np.log(ss.norm.cdf(y0(Nj + 1)) - ss.norm.cdf(y0(Nj)))
                 + np.sum(ss.norm.logpdf(y) + np.log(t.spline_scale[1:]/((Sj + floor)*np.log(10)*t.scale[1:])), axis=1))
        assert_allclose(np.asarray(f.log_prob(c, Nj, Sj, n_quad=64)), exact, rtol=1e-10)
        ## the density of (N, x) for inference: the same, without the Jacobian of x = log10(S + C)
        x = np.log10(Sj + floor)
        assert_allclose(np.asarray(f.log_prob_x(c, Nj, x, n_quad=64)),
                        np.asarray(f.log_prob(c, Nj, Sj, n_quad=64)) + np.sum(np.log((Sj + floor)*np.log(10)), axis=1),
                        rtol=1e-12, atol=1e-12)


def test_untrained_flow_zero_bin_mass_is_the_slab_mass():
    '''
    For a zero bin the density is the base's mass over the slab below log10 C, Phi(y(log10 C)) - Phi(y(log10 C - slab)):
    the scrambled-Sobol integral converges to that analytic value. The transforms are fitted to a set with zeros
    (as in training), so the slab lies inside the fitted range.
    '''
    ts = zero_set(2000)
    em = fitted_emulator(ts)
    f = em.flows[0]
    t = f.transform
    c, N, S = band_arrays(ts, em, np.arange(10))
    Nj, Sj = np.asarray(N[:, 0]), np.array(S[:, f.band.slice])
    zero = Sj == 0
    assert zero.any() and (~zero).any()
    lc, slab = np.log10(fd.NOISE_FLOOR), fd.SLAB_DEX
    y0 = lambda n: _y(t, np.log10(n), 0)
    exact = np.log(ss.norm.cdf(y0(Nj + 1)) - ss.norm.cdf(y0(Nj)))
    for i in range(Sj.shape[1]):
        top, bot = _y(t, lc, i + 1), _y(t, lc - slab, i + 1)
        yi = _y(t, np.log10(Sj[:, i] + fd.NOISE_FLOOR), i + 1)
        dense = ss.norm.logpdf(yi) + np.log(float(t.spline_scale[i + 1])/((Sj[:, i] + fd.NOISE_FLOOR)*np.log(10)*float(t.scale[i + 1])))
        exact = exact + np.where(zero[:, i], np.log(ss.norm.cdf(top) - ss.norm.cdf(bot)), dense)
    errs = {k: np.max(np.abs(np.asarray(f.log_prob(c, Nj, Sj, n_qmc=k)) - exact)) for k in (64, 1024, 8192)}
    assert errs[8192] < 2e-3 and errs[8192] < errs[64]
    ## rows without zeros keep the exact Gauss-Legendre path
    pos = np.where(zero, 1e-40, Sj)
    assert_allclose(np.asarray(f.log_prob(c, Nj, pos, n_quad=64)), np.asarray(ff._log_prob_band(
        f.model, t, f.context_transform, c, jnp.asarray(Nj), jnp.asarray(pos), 64)), rtol=1e-14)


def test_band_transform_round_trip_and_logdet():
    rng = np.random.default_rng(2)
    N = rng.integers(0, 500, 200)
    S = 10**rng.uniform(-41, -37, (200, 3))
    S[rng.uniform(size=S.shape) < 0.3] = 0.0
    t = ff.fit_band_transform(N, S, rng, B=4.0)
    assert float(t.floor) == fd.NOISE_FLOOR and float(t.slab) == fd.SLAB_DEX
    u, v = rng.uniform(size=200), rng.uniform(size=(200, 3))
    y, logdet = ff.band_forward(t, N + u, S, v)
    ## the dequantized spectra fill [-B, B] (a fresh draw can land slightly past); everything stays
    ## inside the spline domain [-B-1, B+1]
    assert float(jnp.max(jnp.abs(y))) < 5.0
    ## zero bins land in the slab below log10 C, nonzero bins above it
    x = np.asarray(y[:, 1:])/np.asarray(t.spline_scale[1:])*np.asarray(t.scale[1:]) + np.asarray(t.loc[1:])
    lc = np.log10(fd.NOISE_FLOOR)
    assert np.all((x[S == 0] <= lc) & (x[S == 0] >= lc - fd.SLAB_DEX - 1e-12)) and np.all(x[S > 0] > lc)
    N_back, S_back = ff.band_inverse(t, y)
    assert_array_equal(np.asarray(N_back), N)
    assert np.all(np.asarray(S_back)[S == 0] == 0)
    ## S far below C loses relative precision to the shift (S + C - C), as expected
    assert_allclose(np.asarray(S_back)[S > 0], S[S > 0], rtol=1e-9)
    ## log|det| against jacfwd of (N_deq, S of nonzero bins, v of zero bins) -> y
    for i in np.flatnonzero((S == 0).any(axis=1) & (S > 0).any(axis=1))[:4]:
        zero = S[i] == 0

        def g(p):
            Si = jnp.where(zero, 0.0, p[1:])
            vi = jnp.where(zero, p[1:], 0.5)
            return ff.band_forward(t, p[0], Si, vi)[0]

        p0 = jnp.concatenate([jnp.asarray([N[i] + u[i]]), jnp.asarray(np.where(zero, v[i], S[i]))])
        J = jax.jacfwd(g)(p0)
        assert_allclose(float(logdet[i]), float(jnp.linalg.slogdet(J)[1]), rtol=1e-11)
    ## without zeros, the v-free path is the same transform
    pos = S[(S > 0).all(axis=1)]
    n35 = np.full(len(pos), 3.5)
    assert_allclose(np.asarray(ff.band_forward(t, n35, pos)[0]), np.asarray(ff.band_forward(t, n35, pos, 0*pos)[0]),
                    rtol=1e-14)
    ## the pre-Stage-3 transform (C = 0) has no slab: zeros are refused
    with pytest.raises(ValueError, match="positive"):
        ff.fit_band_transform(N, S, rng, B=4.0, floor=0.0)


def test_noise_floor_is_the_minimum_of_the_lisa_noise():
    from pelargir.utils import lisa_noise_psd
    f = np.logspace(-4, 0, 400001)
    assert_allclose(fd.NOISE_FLOOR, lisa_noise_psd(f, cpu=True).min(), rtol=1e-3)
    S = np.array([0.0, 1e-42, 1e-38])
    x = fd.log10_shifted(S)
    assert x[0] == np.log10(fd.NOISE_FLOOR)
    assert_array_equal(fd.unshift(x)[:1], [0.0])
    assert_allclose(fd.unshift(x)[1:], S[1:], rtol=1e-6)
    ## the slab below log10 C, and anything under it, is S_gw = 0; C = 0 is plain 10**x
    assert np.all(fd.unshift(np.log10(fd.NOISE_FLOOR) - np.array([0.0, 0.1, 3.0])) == 0)
    assert_allclose(fd.unshift(np.array([-40.0]), 0.0), [1e-40], rtol=1e-15)


@pytest.mark.parametrize("base", ["flax", "zuko"])
def test_training_with_zero_bins(base):
    '''Zero bins are kept and dequantized: training runs, the zero fraction is recorded, and samples include exact zeros.'''
    ts = zero_set(600)
    psd = ts.psd.copy()
    zero = ts.psd == 0
    bands = fd.make_bands(ts.fs, bins_per_band=5)
    if base == "zuko":
        pytest.importorskip("zuko")
        from pelargir.flows import zuko_base as flows
        em, hist = flows.train_emulator(ts, bands=bands, n_epochs=2, progress=False, flow_kwargs=SMALL_ZUKO)
    else:
        em, hist = ff.train_emulator(ts, bands=bands, flow_kwargs=SMALL_FLAX, n_epochs=2, progress=False)
    assert_allclose([h['zero_fraction'] for h in hist], [zero[:, b.slice].mean() for b in bands], rtol=1e-12)
    assert all(np.all(np.isfinite(h['train'] + h['val'])) for h in hist)
    assert em.floor == fd.NOISE_FLOOR
    ## the training set itself is untouched (no rows dropped)
    assert_array_equal(ts.psd[~zero], psd[~zero])
    Ns, Ss = em.sample(ts.context[:2000 if base == 'flax' else 600])
    assert np.all(Ss >= 0) and np.mean(Ss == 0) > 0.05
    ## densities of rows with zeros are finite (scrambled-Sobol path)
    rows = np.flatnonzero(zero.any(axis=1))[:5]
    N_rows = np.stack([fd.band_view(ts, b)[1] for b in bands], axis=1)[rows]
    assert np.all(np.isfinite(em.log_prob(ts.context[rows], N_rows, ts.psd[rows])))


def test_qmc_convergence_on_rows_with_zeros():
    ts = zero_set(400)
    em = fitted_emulator(ts)
    for f in em.flows:
        randomize(f.model, jax.random.key(f.band.index + 3))
    rows = np.arange(12)
    c, N, S = (np.asarray(a) for a in band_arrays(ts, em, rows))
    S = S.copy()
    S[:, [2, 6, 7]] = 0.0
    err, k_ok = fd.qmc_convergence(em, c, N, S, n_qmcs=(64, 256, 1024, 4096))
    assert err.shape == (len(em.bands), 4) and np.all(np.isfinite(err[:, :3]))
    ## the default (QMC_POINTS = 1024) is within 1e-2 nats of 4096 points
    assert fd.QMC_POINTS == 1024
    assert np.all(err[:, 0] >= err[:, 2]) and np.all(err[:, 2] < 1e-2) and k_ok is not None
    ## rows with zeros are left out of the Gauss-Legendre check, which still runs on the rest
    gl, _ = fd.quadrature_convergence(em, c, N, S, n_quads=(8, 64))
    assert np.all(np.isnan(gl))
    gl, _ = fd.quadrature_convergence(em, c, N, np.where(S == 0, 1e-41, S), n_quads=(8, 64))
    assert np.all(np.isfinite(gl))


# =============================================================================
# Count marginalization on a toy problem
# =============================================================================

@pytest.fixture(scope="module")
def toy_flow():
    '''A flow fitted to N ~ Poisson(lam(c)), log10 S ~ N(-38 + 0.3 c, 0.05), c ~ U(0,1).'''
    rng = np.random.default_rng(0)
    n = 20000
    c = rng.uniform(size=(n, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    S = 10**(-38 + 0.3*c + 0.05*rng.standard_normal((n, 1)))
    flow = ff.BandFlow(fd.FrequencyBand(0, 0, 1, np.array([1e-4])), n_context=1, flow_num_layers=4, hidden_size=32)
    ## lr 3e-3 is unstable for this flow (the validation loss rises with epochs); 1e-3 converges
    ff.train_band_flow(flow, c, N, S, n_epochs=15, batch_size=256, lr=1e-3, seed=0, progress=False)
    return flow


def test_count_quadrature_converges_and_recovers_the_joint_density(toy_flow):
    rng = np.random.default_rng(9)
    c = rng.uniform(size=(300, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    logS = -38 + 0.3*c[:, 0] + 0.05*rng.standard_normal(300)
    S = 10**logS[:, None]
    lp = {n: np.asarray(toy_flow.log_prob(c, N, S, n_quad=n)) for n in (8, 32, 64, 512)}
    err = {n: np.max(np.abs(lp[n] - lp[512])) for n in (8, 32, 64)}
    assert err[8] > err[32] > err[64]
    assert err[32] < 1e-4 and err[64] < 2e-5
    exact = ss.poisson.logpmf(N, lam) + ss.norm.logpdf(logS, -38 + 0.3*c[:, 0], 0.05) - np.log(S[:, 0]*np.log(10))
    assert abs(np.mean(lp[512] - exact)) < 0.05
    assert np.std(lp[512] - exact) < 0.2


# =============================================================================
# Autodiff
# =============================================================================

@pytest.fixture(scope="module")
def random_emulator():
    ts = synthetic_set()
    em = fitted_emulator(ts, **SMALL_FLAX)
    for j, f in enumerate(em.flows):
        randomize(f.model, jax.random.key(10 + j))
    return ts, em


def central_difference(fn, x, h):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = h[idx] if np.ndim(h) else h
        g[idx] = (float(fn(jnp.asarray(x + e))) - float(fn(jnp.asarray(x - e))))/(2*e[idx])
    return g


def test_log_prob_gradient_matches_finite_differences(random_emulator):
    ts, em = random_emulator
    c, N, S = band_arrays(ts, em, np.arange(2))
    f_c = lambda cc: jnp.sum(em.jax_log_prob(cc, N, S))
    g = np.asarray(jax.grad(f_c)(c))
    assert np.all(np.isfinite(g)) and np.any(g != 0)
    ## (a 1e-5 relative step is already roundoff-dominated here; 1e-4 agrees to ~4e-9)
    assert_allclose(g, central_difference(f_c, c, 1e-4*np.maximum(np.abs(np.asarray(c)), 1e-2)), rtol=1e-6, atol=1e-8)
    ## with respect to the spectra, in log space for conditioning
    f_s = lambda logS: jnp.sum(em.jax_log_prob(c, N, 10**logS))
    logS = jnp.log10(S)
    g = np.asarray(jax.grad(f_s)(logS))
    assert np.all(np.isfinite(g))
    assert_allclose(g, central_difference(f_s, logS, 1e-6), rtol=1e-6, atol=1e-8)


def test_log_prob_composes_under_jit_vmap_and_an_outer_jit(random_emulator):
    ts, em = random_emulator
    c, N, S = band_arrays(ts, em, np.arange(5))
    one = lambda cc, nn, s: em.jax_log_prob(cc[None], nn[None], s[None])[0]
    g_vmap = jax.jit(jax.vmap(jax.grad(one), in_axes=(0, 0, 0)))(c, N, S)
    g_full = jax.grad(lambda cc: jnp.sum(em.jax_log_prob(cc, N, S)))(c)
    assert_allclose(np.asarray(g_vmap), np.asarray(g_full), rtol=1e-10, atol=1e-12)

    @jax.jit
    def outer(theta):
        ## a caller's jitted function of the population parameters, with the flow inside
        cc = jnp.broadcast_to(theta, c.shape)
        return jnp.sum(em.jax_log_prob(cc, N, S))
    theta = c[0]
    v, g = jax.value_and_grad(outer)(theta)
    assert np.isfinite(float(v)) and np.all(np.isfinite(np.asarray(g)))
    assert_allclose(float(v), float(jnp.sum(em.jax_log_prob(jnp.broadcast_to(theta, c.shape), N, S))), rtol=1e-12)


# =============================================================================
# Training, save/load, dtype, API parity
# =============================================================================

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ts = synthetic_set()
    em, hist = ff.train_emulator(ts, bands=fd.make_bands(ts.fs, bins_per_band=5), flow_kwargs=SMALL_FLAX,
                                 n_epochs=3, batch_size=128, lr=1e-3, progress=False)
    path = tmp_path_factory.mktemp("flax_em")
    em.save(path)
    return ts, em, hist, path


def test_training_reduces_the_loss(trained):
    _, _, hist, _ = trained
    for h in hist:
        assert h['train'][-1] < h['train'][0] and np.all(np.isfinite(h['val']))


def step_loop_reference(flow, context, N, S, n_epochs, batch_size, lr, val_frac=0.1, seed=0):
    '''train_band_flow as it was before the epoch scan: one jitted _train_step per batch, same rng order.'''
    import optax
    rng = np.random.default_rng(seed)
    val, trn = fd.split_by_draw(fd.context_draws(context), val_frac, rng)
    n_val = val.size
    flow.fit_transforms(context[trn], N[trn], S[trn], rng)
    dt = flow.dtype
    c_all = ff.context_forward(flow.context_transform, context).astype(dt)
    N_all, S_all = jnp.asarray(N, jnp.float64), jnp.asarray(S, jnp.float64)
    y_val = ff.band_forward(flow.transform, N_all[val] + jnp.asarray(rng.uniform(size=n_val)), S_all[val])[0].astype(dt)
    model = flow.model
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    history = dict(train=[], val=[])
    n_batches = len(trn)//batch_size
    for _ in range(n_epochs):
        u = jnp.asarray(rng.uniform(size=len(trn)))
        y_trn = ff.band_forward(flow.transform, N_all[trn] + u, S_all[trn])[0].astype(dt)
        order = jnp.asarray(rng.permutation(len(trn))[:n_batches*batch_size].reshape(n_batches, batch_size))
        losses = []
        for b in range(n_batches):
            model, optimizer, loss = ff._train_step(model, optimizer, y_trn[order[b]], c_all[trn][order[b]])
            losses.append(loss)
        history['train'].append(float(jnp.mean(jnp.stack(losses))))
        history['val'].append(float(ff._nll(model, y_val, c_all[val])))
    flow.model = model
    return history


def test_epoch_scan_matches_the_step_loop():
    ts = synthetic_set(1200)
    band = fd.make_bands(ts.fs, bins_per_band=5)[0]
    c, N, S = fd.band_view(ts, band)
    kw = dict(n_epochs=3, batch_size=64, lr=1e-3)
    scan_flow, loop_flow = ff.BandFlow(band, **SMALL_FLAX), ff.BandFlow(band, **SMALL_FLAX)
    h_scan = ff.train_band_flow(scan_flow, c, N, S, progress=False, **kw)
    h_loop = step_loop_reference(loop_flow, c, N, S, **kw)
    for k in ('train', 'val'):
        assert_allclose(h_scan[k], h_loop[k], rtol=1e-9)
    ## the reference keeps the last epoch's parameters; the scan run keeps them in final_params
    p_scan, p_loop = scan_flow.final_params, ff._flat_params(loop_flow.model)
    assert p_scan.keys() == p_loop.keys()
    for k in p_scan:
        assert_allclose(p_scan[k], p_loop[k], rtol=1e-8, atol=1e-12)
    ## the parameters moved, so the comparison is not trivially of two untrained flows
    assert h_scan['train'][-1] < h_scan['train'][0]


def test_save_load_reproduces_log_prob_and_rejects_other_architectures(trained):
    ts, em, _, path = trained
    c, N, S = band_arrays(ts, em, np.arange(6))
    em2 = fd.load_emulator(path)
    assert type(em2) is ff.BandedFlowEmulator
    assert em2.floor == em.floor == fd.NOISE_FLOOR
    assert_array_equal(em2.log_prob(np.asarray(c), np.asarray(N), np.asarray(S)),
                       em.log_prob(np.asarray(c), np.asarray(N), np.asarray(S)))
    ## rows with zeros (the scrambled-Sobol path) too
    Sz = np.array(S)
    Sz[:, [1, 8]] = 0.0
    assert_array_equal(em2.log_prob(np.asarray(c), np.asarray(N), Sz), em.log_prob(np.asarray(c), np.asarray(N), Sz))
    other = ff.BandFlow(em.bands[0], hidden_size=8)
    with pytest.raises(ValueError, match="different architecture"):
        ff._set_params(other.model, ff._flat_params(em.flows[0].model), 'test')


def test_v2_saves_load_as_the_c0_transform(tmp_path):
    '''A pre-Stage-3 save (format v2, transforms without floor and slab) loads as C = 0 and gives the same densities.'''
    import json
    ts = synthetic_set(300)
    em = fitted_emulator(ts, floor=0.0)
    for f in em.flows:
        randomize(f.model, jax.random.key(f.band.index))
    em.save(tmp_path)
    with open(tmp_path/'emulator.json') as fh:
        meta = json.load(fh)
    meta['format'] = 'pelargir-flax-nnx-v2'
    for t in meta['transforms']:
        del t['floor'], t['slab']
    with open(tmp_path/'emulator.json', 'w') as fh:
        json.dump(meta, fh)
    old = fd.load_emulator(tmp_path)
    assert old.floor == 0.0
    c, N, S = (np.asarray(a) for a in band_arrays(ts, em, np.arange(5)))
    assert_array_equal(old.log_prob(c, N, S), em.log_prob(c, N, S))
    with pytest.raises(ValueError, match="can't represent"):
        old.log_prob(c, N, np.where(np.arange(NF) == 3, 0.0, S))
    meta['format'] = 'pelargir-flax-nnx-v1'
    with open(tmp_path/'emulator.json', 'w') as fh:
        json.dump(meta, fh)
    with pytest.raises(ValueError, match="retrain"):
        fd.load_emulator(tmp_path)


def test_api_parity_with_the_zuko_base(trained, tmp_path):
    pytest.importorskip("zuko")
    from pelargir.flows import zuko_base as flows
    ts, em, _, path = trained
    zem, _ = flows.train_emulator(ts, bands=em.bands, n_epochs=1, progress=False, flow_kwargs=SMALL_ZUKO)
    zem.save(tmp_path)
    assert type(fd.load_emulator(tmp_path)) is flows.BandedFlowEmulator
    c, N, S = band_arrays(ts, em, np.arange(4))
    for e in (em, zem):
        lp = e.log_prob(np.asarray(c), np.asarray(N), np.asarray(S))
        assert lp.shape == (4,) and isinstance(lp, np.ndarray)
        Ns, Ss = e.sample(np.asarray(c))
        assert Ns.shape == (4, len(em.bands)) and Ss.shape == (4, NF) and np.all(Ss >= 0)
        assert e.band_log_prob(0, np.asarray(c), np.asarray(N[:, 0]), np.asarray(S[:, :5])).shape == (4,)
    err, _ = fd.quadrature_convergence(em, np.asarray(c), np.asarray(N), np.asarray(S), n_quads=(8, 64))
    assert err.shape == (len(em.bands), 2)


# =============================================================================
# Best-validation checkpoints, saved weights, defaults
# =============================================================================

def scripted_val(monkeypatch, values):
    '''Make flax training report the given validation losses, epoch after epoch.'''
    it = iter(values)
    monkeypatch.setattr(ff, '_nll', lambda model, y, c: next(it))


def test_best_validation_parameters_are_restored(monkeypatch):
    ts = synthetic_set(600)
    band = fd.make_bands(ts.fs, bins_per_band=5)[0]
    c, N, S = fd.band_view(ts, band)
    kw = dict(batch_size=64, lr=1e-3, progress=False)
    ## epoch 1 has the lowest (scripted) validation loss
    scripted_val(monkeypatch, [3.0, 1.0, 2.0, 4.0])
    flow = ff.BandFlow(band, **SMALL_FLAX)
    h = ff.train_band_flow(flow, c, N, S, n_epochs=4, **kw)
    assert h['val'] == [3.0, 1.0, 2.0, 4.0] and h['best_epoch'] == 1 and h['best_val'] == 1.0
    ## the same training stopped after epoch 1 ends where the restored parameters are
    scripted_val(monkeypatch, [5.0, 5.0])
    ref = ff.BandFlow(band, **SMALL_FLAX)
    ff.train_band_flow(ref, c, N, S, n_epochs=2, **kw)
    best, final = ff._flat_params(flow.model), flow.final_params
    for k in best:
        assert_allclose(best[k], ref.final_params[k], rtol=1e-9, atol=1e-13)
    assert any(not np.allclose(final[k], best[k]) for k in best)


def test_save_load_best_and_final_weights(tmp_path, monkeypatch):
    ts = synthetic_set(600)
    bands = fd.make_bands(ts.fs, bins_per_band=5)
    scripted_val(monkeypatch, [2.0, 1.0, 3.0]*len(bands))
    em, hist = ff.train_emulator(ts, bands=bands, flow_kwargs=SMALL_FLAX, n_epochs=3, batch_size=64, progress=False)
    assert [h['best_epoch'] for h in hist] == [1]*len(bands)
    em.save(tmp_path)
    assert (tmp_path/'emulator.npz').exists() and (tmp_path/'emulator_final.npz').exists()
    c, N, S = (np.asarray(a) for a in band_arrays(ts, em, np.arange(6)))
    lp_best = em.log_prob(c, N, S)
    assert_array_equal(fd.load_emulator(tmp_path).log_prob(c, N, S), lp_best)
    lp_final = fd.load_emulator(tmp_path, weights='final').log_prob(c, N, S)
    ## the final weights, put into the trained emulator, give the same density
    for f in em.flows:
        ff._set_params(f.model, f.final_params, 'test')
    assert_array_equal(em.log_prob(c, N, S), lp_final)
    assert np.all(lp_final != lp_best)


def test_production_defaults(monkeypatch):
    import inspect
    from pelargir.scripts import train_flow_emulator as tfe
    band = fd.FrequencyBand(0, 0, 5, FBINS[1:6])
    assert ff.BandFlow(band).arch() == dict(n_context=len(fd.CONTEXT_NAMES), flow_num_layers=8, hidden_size=512,
                                            mlp_num_layers=2, num_bins=8, B=4.0, dtype='float64')
    trainers = [ff.train_band_flow]
    try:
        from pelargir.flows import zuko_base as flows
    except ImportError:
        flows = None
    if flows is not None:
        arch = flows.BandFlow(band).arch()
        assert (arch['transforms'], arch['hidden_size'], arch['mlp_num_layers']) == (8, 512, 2)
        trainers.append(flows.train_band_flow)
    for fn in trainers:
        p = inspect.signature(fn).parameters
        assert (p['n_epochs'].default, p['batch_size'].default, p['lr'].default) == (500, 4096, 1e-4)
    seen = {}
    monkeypatch.setattr(tfe, 'main', lambda args: seen.update(vars(args)))
    tfe.cli(['train.npz', 'out'])
    assert (seen['n_epochs'], seen['batch_size'], seen['lr']) == (500, 4096, 1e-4)
    ## unset architecture options keep the flow base's defaults
    assert seen['flow_layers'] is None and seen['hidden_size'] is None and seen['mlp_layers'] is None
