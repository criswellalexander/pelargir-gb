"""
Tests for the flax flow base (flax_flows.py): the exact density of an untrained flow (which is its
base distribution, so log P(N, S) is analytic), transforms and Jacobians, the count
marginalization on a toy problem, autodiff of the conditional density (against finite
differences, and under jit / vmap / an outer jit), save/load, the float32 option, and API parity
with the zuko base. Runs under any backend; needs flax, distrax and optax.
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

from pelargir import flow_data as fd
from pelargir import flax_flows as ff

FBINS = fd.model_fbins(1e-4, 3e-4, 2e-5)   ## 10 modelled bins
NF = FBINS.size - 1


def synthetic_set(n=2000, seed=0):
    '''A TrainingSet with Poisson counts and log-normal spectra that depend on the context.'''
    rng = np.random.default_rng(seed)
    c = fd.sample_context(rng, n)
    lam = 5 + 40*(c[:, 0] - 0.2)/0.9                               ## depends on m_mu
    nres_f = rng.poisson(lam[:, None]*np.ones(NF)/NF + 1)
    psd = 10**(-38 + 0.3*c[:, 5:6] + 0.05*rng.standard_normal((n, NF)))
    return fd.TrainingSet(c, nres_f.astype(np.int32), psd, np.full(n, 10**6), FBINS, {})


def fitted_emulator(ts, bins_per_band=5, **kw):
    '''An untrained emulator whose transforms are fitted to ts.'''
    em = ff.BandedFlowEmulator(fd.make_bands(ts.fs, bins_per_band=bins_per_band), ts.fbins, **kw)
    rng = np.random.default_rng(1)
    for f in em.flows:
        c, N, S = fd.band_view(ts, f.band)
        f.fit_transforms(c, N, S, rng)
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

def test_untrained_flow_log_prob_is_analytic():
    '''
    An untrained flow is its N(0, I) base, so P(N, S) = [Phi(y0(N+1)) - Phi(y0(N))] prod_i phi(y_i) |dy_i/dS_i|.
    '''
    ts = synthetic_set()
    em = fitted_emulator(ts)
    rows = np.arange(40)
    c, N, S = band_arrays(ts, em, rows)
    for j, f in enumerate(em.flows):
        t = f.transform
        Nj, Sj = np.asarray(N[:, j]), np.asarray(S[:, f.band.slice])
        assert Nj.min() >= 1
        y0 = lambda n: (np.log10(n) - t.loc[0])/t.scale[0]*t.spline_scale[0]
        y = (np.log10(Sj) - t.loc[1:])/t.scale[1:]*t.spline_scale[1:]
        exact = (np.log(ss.norm.cdf(y0(Nj + 1)) - ss.norm.cdf(y0(Nj)))
                 + np.sum(ss.norm.logpdf(y) + np.log(t.spline_scale[1:]/(Sj*np.log(10)*t.scale[1:])), axis=1))
        assert_allclose(np.asarray(f.log_prob(c, Nj, Sj, n_quad=64)), exact, rtol=1e-10)


def test_band_transform_round_trip_and_logdet():
    rng = np.random.default_rng(2)
    N = rng.integers(0, 500, 200)
    S = 10**rng.uniform(-41, -37, (200, 3))
    t = ff.fit_band_transform(N, S, rng, B=4.0)
    u = rng.uniform(size=200)
    y, logdet = ff.band_forward(t, N + u, S)
    ## the spectra fill [-B, B]; a fresh dequantization can move the count slightly past B,
    ## still inside the spline domain [-B-1, B+1]
    assert float(jnp.max(jnp.abs(y[:, 1:]))) <= 4.0 + 1e-12
    assert float(jnp.max(jnp.abs(y[:, 0]))) < 5.0
    N_back, S_back = ff.band_inverse(t, y)
    assert_array_equal(np.asarray(N_back), N)
    assert_allclose(np.asarray(S_back), S, rtol=1e-12)
    f = lambda x: ff.band_forward(t, x[0], x[1:])[0]
    for i in range(3):
        x = jnp.concatenate([jnp.asarray([N[i] + u[i]]), jnp.asarray(S[i])])
        J = jax.jacfwd(f)(x)
        assert_allclose(float(logdet[i]), float(jnp.linalg.slogdet(J)[1]), rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        ff.fit_band_transform(N, np.where(S > 1e-40, S, 0.0), rng, B=4.0)


def test_drop_zero_spectra():
    rng = np.random.default_rng(3)
    c, N, S = rng.normal(size=(6, 8)), np.arange(6), 10**rng.uniform(-40, -38, (6, 3))
    S[1, 2] = S[4, 0] = 0.0
    with pytest.warns(UserWarning, match="dropped 2 of 6"):
        c2, N2, S2, n = fd.drop_zero_spectra(c, N, S)
    assert n == 2
    kept = [0, 2, 3, 5]
    assert_array_equal(N2, kept)
    assert_array_equal(S2, S[kept])
    assert_array_equal(c2, c[kept])
    assert not np.shares_memory(S2, S) and not np.shares_memory(c2, c)
    with pytest.raises(ValueError, match="nothing to train on"):
        fd.drop_zero_spectra(c, N, np.zeros_like(S))


@pytest.mark.parametrize("base", ["flax", "zuko"])
def test_zero_spectra_are_dropped_per_band(base):
    ts = synthetic_set(600)
    ts.psd[[3, 10, 11], 1] = 0.0     ## band 0 holds bins 0-4
    ts.psd[20, 7] = 0.0              ## band 1
    psd = ts.psd.copy()
    bands = fd.make_bands(ts.fs, bins_per_band=5)
    if base == "zuko":
        pytest.importorskip("zuko")
        from pelargir import flows
        train = lambda: flows.train_emulator(ts, bands=bands, n_epochs=1, progress=False)
    else:
        train = lambda: ff.train_emulator(ts, bands=bands, flow_kwargs=dict(hidden_size=16), n_epochs=1,
                                          progress=False)
    with pytest.warns(UserWarning, match="dropped") as record:
        _, hist = train()
    assert sum("dropped" in str(w.message) for w in record) == 2
    assert [h['n_dropped'] for h in hist] == [3, 1]
    assert all(np.all(np.isfinite(h['train'] + h['val'])) for h in hist)
    ## the training set itself is untouched
    assert_array_equal(ts.psd, psd)


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
    flow = ff.BandFlow(fd.FrequencyBand(0, 0, 1, np.array([1e-4])), n_context=1, hidden_size=32)
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
    em = fitted_emulator(ts, hidden_size=16)
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
    assert_allclose(g, central_difference(f_c, c, 1e-5*np.maximum(np.abs(np.asarray(c)), 1e-2)), rtol=1e-6, atol=1e-8)
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


def test_samples_are_differentiable_in_the_context_at_fixed_latents(random_emulator):
    ts, em = random_emulator
    f = em.flows[0]
    c = jnp.asarray(ts.context[:3])
    z = jax.random.normal(jax.random.key(4), (3, f.features), dtype=jnp.float64)

    def flow_output(cc):
        return jnp.sum(f.model.forward(z, ff.context_forward(f.context_transform, cc).astype(f.dtype)))
    ## the flow output y is O(1), so central differences are not swamped by round-off; S is a fixed
    ## affine-exponential map of y, differentiated exactly by autodiff
    g = np.asarray(jax.grad(flow_output)(c))
    assert np.all(np.isfinite(g)) and np.any(g != 0)
    assert_allclose(g, central_difference(flow_output, c, 1e-5*np.maximum(np.abs(np.asarray(c)), 1e-2)),
                    rtol=1e-6, atol=1e-9)
    S_grad = jax.grad(lambda cc: jnp.sum(jnp.log10(ff.band_inverse(
        f.transform, f.model.forward(z, ff.context_forward(f.context_transform, cc).astype(f.dtype)))[1])))(c)
    assert np.all(np.isfinite(np.asarray(S_grad)))


# =============================================================================
# Training, save/load, dtype, API parity
# =============================================================================

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ts = synthetic_set()
    em, hist = ff.train_emulator(ts, bands=fd.make_bands(ts.fs, bins_per_band=5), flow_kwargs=dict(hidden_size=16),
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
    rows = rng.permutation(len(N))
    n_val = int(round(val_frac*len(N)))
    val, trn = rows[:n_val], rows[n_val:]
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
    scan_flow, loop_flow = ff.BandFlow(band, hidden_size=16), ff.BandFlow(band, hidden_size=16)
    h_scan = ff.train_band_flow(scan_flow, c, N, S, progress=False, **kw)
    h_loop = step_loop_reference(loop_flow, c, N, S, **kw)
    for k in ('train', 'val'):
        assert_allclose(h_scan[k], h_loop[k], rtol=1e-9)
    p_scan, p_loop = ff._flat_params(scan_flow.model), ff._flat_params(loop_flow.model)
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
    assert_array_equal(em2.log_prob(np.asarray(c), np.asarray(N), np.asarray(S)),
                       em.log_prob(np.asarray(c), np.asarray(N), np.asarray(S)))
    other = ff.BandFlow(em.bands[0], hidden_size=8)
    with pytest.raises(ValueError, match="different architecture"):
        ff._set_params(other.model, ff._flat_params(em.flows[0].model), 'test')


def test_float32_option():
    ts = synthetic_set(600)
    em, _ = ff.train_emulator(ts, flow_kwargs=dict(dtype='float32', hidden_size=16), n_epochs=1, batch_size=64,
                              progress=False)
    leaves = jax.tree_util.tree_leaves(nnx.state(em.flows[0].model, nnx.Param))
    assert all(l.dtype == jnp.float32 for l in leaves)
    c, N, S = band_arrays(ts, em, np.arange(4))
    lp = em.log_prob(np.asarray(c), np.asarray(N), np.asarray(S))
    assert lp.dtype == np.float64 and np.all(np.isfinite(lp))


def test_api_parity_with_the_zuko_base(trained, tmp_path):
    pytest.importorskip("zuko")
    from pelargir import flows
    ts, em, _, path = trained
    zem, _ = flows.train_emulator(ts, bands=em.bands, n_epochs=1, progress=False)
    zem.save(tmp_path)
    assert type(fd.load_emulator(tmp_path)) is flows.BandedFlowEmulator
    c, N, S = band_arrays(ts, em, np.arange(4))
    for e in (em, zem):
        lp = e.log_prob(np.asarray(c), np.asarray(N), np.asarray(S))
        assert lp.shape == (4,) and isinstance(lp, np.ndarray)
        Ns, Ss = e.sample(np.asarray(c))
        assert Ns.shape == (4, len(em.bands)) and Ss.shape == (4, NF) and np.all(Ss > 0)
        assert e.band_log_prob(0, np.asarray(c), np.asarray(N[:, 0]), np.asarray(S[:, :5])).shape == (4,)
    err, _ = fd.quadrature_convergence(em, np.asarray(c), np.asarray(N), np.asarray(S), n_quads=(8, 64))
    assert err.shape == (len(em.bands), 2)
