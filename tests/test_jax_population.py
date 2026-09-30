"""
Tests for the JAX forward model (jax_population.py): the samplers against analytic CDFs and
against inference.GalacticBinaryPrior, exact parity of the fused sample-and-threshold kernel
with serial_array_sort/block_array_sort on the same (materialized) draw, per-galaxy N and
padding, and PopModel's jax path.

Runs under any backend: `PELARGIR_BACKEND=cupy pytest tests/test_jax_population.py` uses the
cupy reference (host numpy under numpy and jax); the PopModel tests need PELARGIR_BACKEND=jax.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
import scipy.stats as ss
from numpy.testing import assert_allclose, assert_array_equal

from pelargir import backend
from pelargir.models import PopModel
from pelargir.inference import GalacticBinaryPrior
from pelargir.utils import get_amp_freq, to_numpy

pytest.importorskip("jax")
import jax
from pelargir import jax_population as jp

xp = backend.xp

FIDUCIAL = np.array([0.6, 0.15, 3.31, 0.75, 0.33, 0.5])  ## Table 1, arXiv:2604.03390
NAMES = ['m_mu', 'm_sigma', 'rh_disk', 'r_bulge', 'q_bd', 'a_alpha']
## fiducial plus prior edges: power-law slope, bulge fraction, mass mean near each bound
LAMBDAS = {
    "fiducial":     FIDUCIAL,
    "alpha -0.5":   np.array([0.6, 0.15, 3.31, 0.75, 0.33, -0.5]),
    "alpha 1.5":    np.array([0.6, 0.15, 3.31, 0.75, 0.33, 1.5]),
    "q_bd 0.01":    np.array([0.6, 0.15, 3.31, 0.75, 0.01, 0.5]),
    "q_bd 0.99":    np.array([0.6, 0.15, 9.0, 1.9, 0.99, 0.5]),
    "m_mu 0.2":     np.array([0.2, 0.15, 3.31, 0.75, 0.33, 0.5]),
    "m_mu 1.1":     np.array([1.1, 0.3, 3.31, 0.75, 0.33, 0.5]),
}

GBPRIOR = GalacticBinaryPrior(xp.random.default_rng(0))
BOUNDS = jp.prior_bounds(GBPRIOR)
M_MIN, M_MAX, A_MIN, A_MAX, X0 = BOUNDS


def dL_cdf(x, r_bulge, rh_disk, q_bd, x0=X0):
    """CDF of |X|, X = x0 + N(0, r_bulge) w.p. q_bd, else x0 + Laplace(0, rh_disk)."""
    def cdf_X(y):
        lap = np.where(y < x0, 0.5*np.exp((y - x0)/rh_disk), 1 - 0.5*np.exp(-(y - x0)/rh_disk))
        return q_bd*ss.norm.cdf(y, x0, r_bulge) + (1 - q_bd)*lap
    return cdf_X(x) - cdf_X(-x)


def marginal_cdfs(theta):
    m_mu, m_sigma, rh_disk, r_bulge, q_bd, a_alpha = theta
    m = ss.truncnorm((M_MIN - m_mu)/m_sigma, (M_MAX - m_mu)/m_sigma, loc=m_mu, scale=m_sigma).cdf
    return [m, m, lambda x: dL_cdf(x, r_bulge, rh_disk, q_bd),
            lambda x: ((x - A_MIN)/(A_MAX - A_MIN))**(a_alpha + 1)]


# =============================================================================
# Samplers
# =============================================================================

@pytest.mark.parametrize("name", list(LAMBDAS))
def test_samplers_match_analytic_cdfs(name):
    theta = LAMBDAS[name]
    draw = jp.sample_galaxy_draw(jax.random.key(1), 0, theta, 200000, BOUNDS)
    for label, sample, cdf in zip(["m_1", "m_2", "d_L", "a"], draw, marginal_cdfs(theta)):
        assert ss.kstest(sample, cdf).pvalue > 1e-3, label
    assert np.all((draw[:2] > M_MIN) & (draw[:2] < M_MAX))
    assert np.all((draw[3] >= A_MIN) & (draw[3] <= A_MAX))


@pytest.mark.parametrize("name", ["fiducial", "q_bd 0.99", "alpha -0.5"])
def test_samplers_match_reference_sampler(name):
    theta = LAMBDAS[name]
    n = 100000
    gbprior = GalacticBinaryPrior(xp.random.default_rng(7))
    gbprior.condition({k: xp.array([v]) for k, v in zip(NAMES, theta)})
    ref = to_numpy(gbprior.sample_conditional(n)).reshape(4, n)
    draw = jp.sample_galaxy_draw(jax.random.key(2), 0, theta, n, BOUNDS)
    for label, a, b in zip(["m_1", "m_2", "d_L", "a"], draw, ref):
        assert ss.ks_2samp(a, b).pvalue > 1e-3, label


def test_masses_are_independent():
    draw = jp.sample_galaxy_draw(jax.random.key(3), 0, FIDUCIAL, 50000, BOUNDS)
    assert abs(ss.spearmanr(draw[0], draw[1])[0]) < 0.02


# =============================================================================
# Forward model: exact parity with the thresholders on the materialized draw
# =============================================================================

@pytest.fixture(scope="module")
def grid():
    pm = PopModel(int(1e4), xp.random.default_rng(11), Nreal=1, block_after=4)
    th = pm.thresher
    return pm.fbins, th, (pm.fbins + 0.5*th.delf, th.noisePSD, th.LISA_rx, th.duration, th.duration_eff)


def forward(key, thetas, Ns, grid, **kw):
    fbins, th, consts = grid
    return jp.jax_forward_model(key, np.atleast_2d(thetas), np.atleast_2d(Ns), BOUNDS, *consts,
                                return_mask=True, out_module=xp, **kw)


def reference(key, g, theta, N, grid):
    """serial and block results, and serial's res_idx, for galaxy g's materialized draw."""
    fbins, th, _ = grid
    if N == 0:
        ## the reference sorts reject an empty galaxy; its result is known
        return 0, np.zeros(len(fbins)), set()
    A, f = get_amp_freq(jp.sample_galaxy_draw(key, g, theta, N, BOUNDS, out_module=xp))
    obs = xp.array([f, A])
    s_N, s_fg, idx = th.serial_array_sort(obs, fbins, get_indices=True)
    b_N, b_fg = th.block_array_sort(obs, fbins)
    assert int(s_N) == int(b_N)
    assert_allclose(to_numpy(s_fg), to_numpy(b_fg), rtol=1e-12, atol=0.0)
    return int(s_N), to_numpy(s_fg), {int(i) for i in idx}


def assert_galaxy_matches(Nres, fg, mask, ref, N):
    r_N, r_fg, r_idx = ref
    assert int(Nres) == r_N
    assert_allclose(to_numpy(fg), r_fg, rtol=1e-12, atol=0.0)
    mask = to_numpy(mask)
    assert not mask[N:].any()
    assert set(np.flatnonzero(mask[:N]).tolist()) == r_idx


@pytest.mark.parametrize("cut", [None, 1.0, 7.0], ids=["unfiltered", "cut-1", "cut-7"])
@pytest.mark.parametrize("N", [100000, 73000])
def test_forward_matches_thresholders_on_same_draw(grid, N, cut):
    key = jax.random.key(5)
    Nres, fg, mask = forward(key, FIDUCIAL, [[N]], grid, prefilter_snr=cut)
    assert_galaxy_matches(Nres, fg, mask, reference(key, 0, FIDUCIAL, N, grid), N)


def test_mixed_N_and_hyperparameters_in_one_batch(grid):
    ## Nreal = 2, Nparallel = 3; galaxy g = r*3 + p uses theta p and N[r, p]
    key = jax.random.key(6)
    thetas = np.stack([LAMBDAS["fiducial"], LAMBDAS["alpha 1.5"], LAMBDAS["q_bd 0.99"]])
    Ns = np.array([[60000, 25000, 0], [1500, 90000, 40000]])
    Nres, fg, mask = forward(key, thetas, Ns, grid)
    assert to_numpy(Nres).shape == (2, 3)
    for r in range(2):
        for p in range(3):
            ref = reference(key, r*3 + p, thetas[p], Ns[r, p], grid)
            assert_galaxy_matches(Nres[r, p], fg[:, r, p], mask[:, r, p], ref, Ns[r, p])


def test_results_do_not_depend_on_the_padded_size(grid):
    ## galaxy 0 (N = 1000) is padded to 1024 alone, and to pad_bucket(60000) next to a larger galaxy
    key = jax.random.key(8)
    thetas = np.stack([FIDUCIAL, FIDUCIAL])
    alone = forward(key, FIDUCIAL, [[1000]], grid)
    shared = forward(key, thetas, [[1000, 60000]], grid)
    assert jp.pad_bucket(1000) != jp.pad_bucket(60000)
    assert int(alone[0]) == int(shared[0][0, 0])
    assert_allclose(to_numpy(alone[1]), to_numpy(shared[1][:, 0, 0]), rtol=1e-14, atol=0.0)
    assert_array_equal(to_numpy(alone[2])[:1000], to_numpy(shared[2][:1000, 0, 0]))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_results_do_not_depend_on_batch_size(grid, batch_size):
    key = jax.random.key(9)
    thetas = np.stack([LAMBDAS["fiducial"], LAMBDAS["m_mu 1.1"], LAMBDAS["alpha -0.5"]])
    Ns = np.array([[20000, 30000, 25000], [22000, 18000, 30000]])
    ref = forward(key, thetas, Ns, grid)
    out = forward(key, thetas, Ns, grid, batch_size=batch_size)
    assert_array_equal(to_numpy(out[0]), to_numpy(ref[0]))
    assert_allclose(to_numpy(out[1]), to_numpy(ref[1]), rtol=1e-14, atol=0.0)
    assert_array_equal(to_numpy(out[2]), to_numpy(ref[2]))


def test_same_key_same_result_and_different_key_differs(grid):
    a = forward(jax.random.key(10), FIDUCIAL, [[30000]], grid)
    b = forward(jax.random.key(10), FIDUCIAL, [[30000]], grid)
    c = forward(jax.random.key(11), FIDUCIAL, [[30000]], grid)
    assert_array_equal(to_numpy(a[2]), to_numpy(b[2]))
    assert_allclose(to_numpy(a[1]), to_numpy(b[1]), rtol=1e-14, atol=0.0)
    assert not np.array_equal(to_numpy(a[2]), to_numpy(c[2]))


def test_empty_galaxy(grid):
    Nres, fg, mask = forward(jax.random.key(12), FIDUCIAL, [[0]], grid)
    assert int(Nres) == 0
    assert not to_numpy(fg).any()
    assert not to_numpy(mask).any()


def test_pad_bucket():
    buckets = [jp.pad_bucket(n) for n in [0, 1, 1024, 1025, 10**6, 10**8]]
    assert buckets[:3] == [1024, 1024, 1024]
    for n in [1025, 5000, 10**6, 10**8, 123456789]:
        b = jp.pad_bucket(n)
        assert n <= b <= 1.25*n + 1


# =============================================================================
# PopModel on the jax backend
# =============================================================================

jax_backend_only = pytest.mark.skipif(backend.BACKEND != "jax", reason="needs PELARGIR_BACKEND=jax")


@jax_backend_only
def test_popmodel_shapes_with_realizations_and_parallel_points():
    pm = PopModel(int(2e4), xp.random.default_rng(1), Nreal=2, block_after=4, jax_seed=3)
    fs, fg, Nres = pm.run_model(pop_theta=np.stack([FIDUCIAL, LAMBDAS["alpha 1.5"], LAMBDAS["q_bd 0.01"]]))
    assert fg.shape == (len(pm.fbins) - 1, 2, 3)
    assert Nres.shape == (2, 3)
    assert np.isfinite(to_numpy(fg)).all()
    assert_array_equal(pm.last_Ntot, np.full((2, 3), 20000))


@jax_backend_only
def test_popmodel_return_extras_matches_serial_on_returned_draw():
    pm = PopModel(int(1e5), xp.random.default_rng(3), Nreal=1, block_after=4, jax_seed=4)
    fs, fg, Nres, res_idx, galaxy_draw = pm.run_model(pop_theta=FIDUCIAL[None], return_extras=True)
    assert galaxy_draw.shape == (4, 100000, 1, 1)
    A, f = get_amp_freq(galaxy_draw)
    r_N, r_fg, r_idx = pm.thresher.serial_array_sort(xp.array([f, A]), pm.fbins, snr_thresh=pm.thresh_val,
                                                     get_indices=True)
    assert int(Nres) == int(r_N)
    assert_allclose(to_numpy(fg), to_numpy(pm.reweight_foreground(r_fg)[1:, ...]), rtol=1e-12, atol=0.0)
    assert set(res_idx) == {int(i) for i in r_idx}


@jax_backend_only
def test_popmodel_statistically_matches_the_reference_sampler():
    """Mean N_res and banded mean foreground over 20 realizations agree with the xp sampler + block sort."""
    Nreal, N = 20, int(1e5)
    pm = PopModel(N, xp.random.default_rng(5), Nreal=Nreal, block_after=4, jax_seed=6)
    fs, fg, Nres = pm.run_model(pop_theta=FIDUCIAL[None])
    fg, Nres = to_numpy(fg).reshape(-1, Nreal), to_numpy(Nres).reshape(Nreal)

    gbprior = GalacticBinaryPrior(xp.random.default_rng(8), Nreal=Nreal)
    gbprior.condition({k: xp.array([v]) for k, v in zip(NAMES, FIDUCIAL)})
    A, f = get_amp_freq(gbprior.sample_conditional(N))
    r_N, r_fg = pm.thresher.block_array_sort(xp.array([f, A]), pm.fbins, snr_thresh=pm.thresh_val)
    r_fg = to_numpy(pm.reweight_foreground(r_fg)[1:, ...]).reshape(-1, Nreal)
    r_N = to_numpy(r_N).reshape(Nreal)

    def agree(a, b):
        se = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)
        return abs(a.mean() - b.mean()) <= 5*se + 1e-300
    assert agree(Nres.astype(float), r_N.astype(float))
    for band in np.array_split(np.arange(fg.shape[0]), 5):
        assert agree(fg[band].sum(axis=0), r_fg[band].sum(axis=0))


@jax_backend_only
def test_popmodel_poisson_Ntot():
    rate = 3000
    pm = PopModel(None, xp.random.default_rng(9), Nreal=8, block_after=4, Ntot_rate=rate, jax_seed=10)
    Ns = []
    for _ in range(40):
        pm.run_model(pop_theta=FIDUCIAL[None])
        Ns.append(pm.last_Ntot.ravel())
    Ns = np.concatenate(Ns)
    assert abs(Ns.mean() - rate) < 5*np.sqrt(rate/Ns.size)
    assert 0.7 < Ns.var(ddof=1)/rate < 1.3
    assert len(np.unique(Ns)) > 50


@pytest.mark.skipif(backend.BACKEND == "jax", reason="checks the non-jax backends")
def test_poisson_Ntot_requires_the_jax_backend():
    with pytest.raises(NotImplementedError, match="requires the jax backend"):
        PopModel(None, xp.random.default_rng(1), Ntot_rate=1e4)
