"""
Tests for the JAX likelihood terms (jax_likelihood.py) against inference.FG_Likelihood,
Nres_Likelihood, Res_Astro_Likelihood and GalacticBinaryPrior on identical inputs, plus
independent references (scipy.stats) for the negative binomial and truncated normal.

Runs under any backend: `PELARGIR_BACKEND=cupy pytest tests/test_jax_likelihood.py` uses the
cupy reference; the PopModel tests need PELARGIR_BACKEND=jax.
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
import scipy.stats as ss
from scipy.special import gammaln
from numpy.testing import assert_allclose, assert_array_equal

from pelargir import backend
from pelargir.models import PopModel
from pelargir.inference import GalacticBinaryPrior, FG_Likelihood, Nres_Likelihood, Res_Astro_Likelihood
from pelargir import distributions as st
from pelargir.utils import apply_theta_lims, lisa_noise_psd

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from pelargir import jax_population as jp
from pelargir import jax_likelihood as jl

xp = backend.xp
to_numpy = jp._host  ## numpy, cupy or jax array as numpy

FIDUCIAL = np.array([0.6, 0.15, 3.31, 0.75, 0.33, 0.5])  ## Table 1, arXiv:2604.03390
NAMES = ['m_mu', 'm_sigma', 'rh_disk', 'r_bulge', 'q_bd', 'a_alpha']
## fiducial and hyperprior-edge points
THETAS = np.array([FIDUCIAL,
                   [0.2, 0.3, 1.0, 0.05, 0.01, -0.5],
                   [1.1, 0.05, 10.0, 2.0, 0.99, 1.5]])
BOUNDS = jp.prior_bounds(GalacticBinaryPrior(xp.random.default_rng(0)))

FBINS = np.arange(1e-4, 1e-3, 5e-5)
NF = len(FBINS)
DELF = FBINS[1] - FBINS[0]
DURATION = 1.2623e8
G, MSUN_KG, AU_M = 6.6743e-11, 1.98840987e30, 1.495978707e11

## FG hyperparameters used for the paper's toy analysis (App. B)
FG_HP = dict(hp_alpha=5, hp_beta=0.05)


def rel(a, b):
    return to_numpy(a), to_numpy(b)


def negbin_atol(N_obs, Nhat):
    '''The NegBin log-gamma terms (~1e5) cancel to O(1-1e3): compare at their float64 precision.'''
    return 1e-14*np.max(gammaln(3 + to_numpy(Nhat).sum(axis=0) + float(to_numpy(N_obs))))


def fg_setup(Nr, Np, seed=1):
    '''Observed foreground, noise and model draws (Nf-1, Nr, Np) on fbins[1:].'''
    rng = np.random.default_rng(seed)
    noise = lisa_noise_psd(FBINS[1:], cpu=True)
    fg_true = noise*10**rng.uniform(-1, 1, NF - 1)
    draws = fg_true[:, None, None]*10**(0.1*rng.standard_normal((NF - 1, Nr, Np)))
    return fg_true, noise, draws


def fg_reference(fg_true, noise, Nr, sigma=0.1, **hp):
    return FG_Likelihood(xp.asarray(fg_true), xp.asarray(sigma), xp.asarray(noise), Nreal=Nr, **hp)


def nres_reference(N_obs):
    return Nres_Likelihood(N_obs)


def consts_for(fg_like, nres_like, rx=None):
    rx = np.ones(NF) if rx is None else rx
    return jl.make_consts(fg_like, nres_like, FBINS + 0.5*DELF, lisa_noise_psd(FBINS, cpu=True), rx,
                          DURATION, 7.0, BOUNDS)


def thetas_at(freqs, m=0.6, d_L=8.0):
    '''(Nres, 4) binaries of mass m, distance d_L at the given GW frequencies.'''
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    a = (G*2*m*MSUN_KG/(np.pi*freqs)**2)**(1/3)/AU_M
    return np.column_stack([np.full(freqs.shape, m), np.full(freqs.shape, m),
                            np.broadcast_to(d_L, freqs.shape).astype(float), a])


def conditioned_prior(thetas, Nr):
    gbprior = GalacticBinaryPrior(xp.random.default_rng(0), Nreal=Nr)
    gbprior.condition({k: xp.asarray(thetas[:, i]) for i, k in enumerate(NAMES)})
    return gbprior


# =============================================================================
# N_res
# =============================================================================

@pytest.mark.parametrize("Nr,Np", [(2, 1), (5, 3)])
def test_negbin_matches_marginal_poisson_gamma_and_scipy(Nr, Np):
    rng = np.random.default_rng(2)
    N_obs = 11879
    Nhat = rng.integers(11000, 13000, (Nr, Np))
    ref = nres_reference(N_obs)
    out = jl.negbin_logpmf(float(N_obs), jnp.asarray(Nhat), ref.base_dist.alpha, ref.base_dist.beta)
    atol = negbin_atol(N_obs, Nhat)
    assert_allclose(*rel(out, ref.ln_prob(xp.asarray(Nhat))), rtol=0, atol=atol)
    ## Eq. A8 directly
    betap = 1e-3 + Nr
    assert_allclose(to_numpy(out), ss.nbinom.logpmf(N_obs, 3 + Nhat.sum(axis=0), betap/(1 + betap)),
                    rtol=0, atol=10*atol)


# =============================================================================
# Foreground
# =============================================================================

def test_lognormal_grid_matches_FG_Likelihood():
    fg_true, noise, _ = fg_setup(2, 1)
    ref = fg_reference(fg_true, noise, 2, **FG_HP)
    c = consts_for(ref, nres_reference(10))
    assert_allclose(to_numpy(c.ln_pgrid[:, 0, :]), to_numpy(ref.ln_pgrid), rtol=1e-13)


@pytest.mark.parametrize("Nr,Np", [(2, 1), (5, 3)])
def test_fg_ln_like_matches_FG_Likelihood(Nr, Np):
    fg_true, noise, draws = fg_setup(Nr, Np)
    ref = fg_reference(fg_true, noise, Nr, **FG_HP)
    c = consts_for(ref, nres_reference(10))
    out = jl.fg_ln_like(jnp.asarray(draws), c.noise, c.log10_cgrid, c.ln_pgrid, c.mu0, c.alpha, c.beta, c.nu)
    assert out.shape == (Np,)
    assert_allclose(*rel(out, ref.ln_prob(xp.asarray(draws))), rtol=1e-12)


def test_fg_per_bin_hyperparameters_match_per_bin_scalar_references():
    '''Arrays of per-bin hyperparameters give, in each bin, the scalar-hyperparameter result.'''
    Nr, Np = 3, 2
    fg_true, noise, draws = fg_setup(Nr, Np)
    rng = np.random.default_rng(4)
    alpha, beta = rng.uniform(2, 6, NF - 1), rng.uniform(0.02, 0.1, NF - 1)
    sigma = rng.uniform(0.05, 0.2, NF - 1)
    base = consts_for(fg_reference(fg_true, noise, Nr, **FG_HP), nres_reference(10))
    per_bin = base._replace(alpha=jnp.asarray(alpha)[:, None], beta=jnp.asarray(beta)[:, None],
                            ln_pgrid=jl.lognormal_grid(10**base.log10_cgrid, jnp.asarray(fg_true + noise),
                                                       jnp.asarray(sigma))[:, None, :])
    ## the band sum equals the sum of single-bin scalar-hyperparameter references
    total = to_numpy(jl.fg_ln_like(jnp.asarray(draws), per_bin.noise, per_bin.log10_cgrid, per_bin.ln_pgrid,
                                   per_bin.mu0, per_bin.alpha, per_bin.beta, per_bin.nu))
    expected = np.zeros(Np)
    for j in range(NF - 1):
        ref = fg_reference(fg_true[j:j+1], noise[j:j+1], Nr, sigma=sigma[j], hp_alpha=alpha[j], hp_beta=beta[j])
        expected += to_numpy(ref.ln_prob(xp.asarray(draws[j:j+1])))
    assert_allclose(total, expected, rtol=1e-12)


# =============================================================================
# Resolved-binary prior
# =============================================================================

def test_conditional_params_match_GalacticBinaryPrior_condition():
    gbprior = conditioned_prior(THETAS, 2)
    d = gbprior.conditional_dict
    p = jp.conditional_params(jnp.asarray(THETAS), BOUNDS)
    for key in ('m_1', 'm_2'):
        assert_allclose(to_numpy(d[key].loc[0]), to_numpy(p.m_loc), rtol=0)
        assert_allclose(to_numpy(d[key].scale[0]), to_numpy(p.m_scale), rtol=0)
        assert (to_numpy(d[key].a_min).item(), to_numpy(d[key].a_max).item()) == (p.m_min, p.m_max)
    mix = d['d_L']
    assert to_numpy(mix.x0).item() == p.x0
    assert_allclose(to_numpy(mix.bulge_scale[0]), to_numpy(p.r_bulge), rtol=0)
    assert_allclose(to_numpy(mix.disk_scale[0]), to_numpy(p.rh_disk), rtol=0)
    assert_allclose(to_numpy(mix.beta[0]), to_numpy(p.q_bd), rtol=0)
    pl = d['a']
    assert_allclose(to_numpy(pl.alpha[0]), to_numpy(p.a_alpha), rtol=0)
    assert (to_numpy(pl.loc).item(), to_numpy(pl.scale).item()) == (p.a_loc, p.a_scale)


def test_log_gauss_mass_matches_distributions_in_every_case():
    a = np.array([-3.0, -40.0, -1.0, 0.5, 2.0, 30.0, -0.2])
    b = np.array([-1.0, -39.0, 1.0, 3.0, 2.5, 31.0, 0.1])
    out = to_numpy(jl.log_gauss_mass(jnp.asarray(a), jnp.asarray(b)))
    assert_allclose(out, to_numpy(st._log_gauss_mass(xp.asarray(a), xp.asarray(b))), rtol=1e-12)
    ## direct differences are accurate away from the far tails
    ok = np.abs(a) < 5
    assert_allclose(out[ok], np.log(ss.norm.cdf(b[ok]) - ss.norm.cdf(a[ok])), rtol=1e-10)


def test_prior_logpdfs_match_distributions():
    rng = np.random.default_rng(5)
    Nr, Np, Nres = 2, THETAS.shape[0], 400
    gbprior = conditioned_prior(THETAS, Nr)
    ## include out-of-bounds masses, and d_L on both sides of the Galactic centre
    state = np.column_stack([rng.uniform(0.1, 1.5, Nres), rng.uniform(0.1, 1.5, Nres),
                             rng.uniform(1e-3, 30, Nres), 10**rng.uniform(-4, -2, Nres)])
    ref = to_numpy(gbprior.conditional_logpdf(xp.asarray(state.T)))[:, :, 0, :]   ## (4, Nres, Np)
    p = jp.conditional_params(jnp.asarray(THETAS)[None], BOUNDS)
    s = jnp.asarray(state)
    terms = [jl.truncnorm_logpdf(s[:, 0, None], p.m_loc, p.m_scale, p.m_min, p.m_max),
             jl.truncnorm_logpdf(s[:, 1, None], p.m_loc, p.m_scale, p.m_min, p.m_max),
             jl.mixture_logpdf(s[:, 2, None], p.x0, p.r_bulge, p.rh_disk, p.q_bd),
             jl.powerlaw_logpdf(s[:, 3, None], p.a_alpha, p.a_loc, p.a_scale)]
    for i, t in enumerate(terms):
        t = to_numpy(t)
        assert_array_equal(np.isfinite(t), np.isfinite(ref[i]))
        assert_allclose(t[np.isfinite(t)], ref[i][np.isfinite(t)], rtol=1e-12)
    assert not np.isfinite(ref[0]).all()
    assert_allclose(to_numpy(jl.gb_log_prior(s, jnp.asarray(THETAS), BOUNDS)), ref.sum(axis=0), rtol=1e-12)


def test_truncnorm_matches_scipy():
    x = np.linspace(0.18, 1.43, 50)
    for m_mu, m_sigma in [(0.6, 0.15), (0.2, 0.3), (1.1, 0.05)]:
        out = jl.truncnorm_logpdf(jnp.asarray(x), m_mu, m_sigma, 0.17, 1.44)
        ref = ss.truncnorm.logpdf(x, (0.17 - m_mu)/m_sigma, (1.44 - m_mu)/m_sigma, loc=m_mu, scale=m_sigma)
        assert_allclose(to_numpy(out), ref, rtol=1e-10)


# =============================================================================
# Resolved-binary term
# =============================================================================

def res_setup(Nr, Np, seed=6):
    '''Resolved binaries across the band (incl. bin 0, above the band, near edges) and draws.'''
    rng = np.random.default_rng(seed)
    freqs = np.concatenate([FBINS[1:-1] + rng.uniform(-0.45, 0.45, NF - 2)*DELF,
                            [FBINS[0] - 0.3*DELF, FBINS[0], FBINS[-1] + 2*DELF, FBINS[3] + 0.4999*DELF]])
    rx = 0.3 + rng.uniform(0, 1, NF)
    fg_true, noise, draws = fg_setup(Nr, Np, seed)
    ## distances putting each binary's SNR against noise + fg_true within ~0.1 dex of 7
    k = np.clip(np.digitize(freqs, FBINS + 0.5*DELF), 1, NF - 1)
    A1 = np.asarray(jp.amp_freq(jnp.asarray(thetas_at(freqs, m=0.6, d_L=1.0).T))[0])
    A_target = 7*10**rng.uniform(-0.1, 0.1, freqs.size)*np.sqrt((noise + fg_true)[k - 1]/(DURATION*rx[k]))
    state = thetas_at(freqs, m=0.6, d_L=A1/A_target)
    return state, rx, draws


@pytest.mark.parametrize("Nr,Np", [(2, 1), (5, 3)])
def test_res_ln_like_matches_Res_Astro_Likelihood(Nr, Np):
    state, rx, draws = res_setup(Nr, Np)
    ra = Res_Astro_Likelihood(xp.random.default_rng(1), xp.asarray(state), xp.asarray(FBINS), xp.asarray(rx),
                              duration=DURATION, scatter=False, dynamic_scatter=False)
    thetas = THETAS[:Np]
    Sn = lisa_noise_psd(FBINS, cpu=True)
    ref = ra.static_ln_conditional_prob(conditioned_prior(thetas, Nr), xp.asarray(draws), xp.asarray(Sn), 7.0)
    p_ref = to_numpy(ra.res_prob(xp.asarray(draws), xp.asarray(Sn), 7.0))
    p = to_numpy(jl.res_prob(jnp.asarray(state), jnp.asarray(draws), jnp.asarray(Sn), jnp.asarray(rx),
                             jnp.asarray(FBINS + 0.5*DELF), DURATION, 7.0))
    assert_allclose(p, p_ref, rtol=1e-15, atol=0)
    assert 0 < p.mean() < 1 and ((p > 0) & (p < 1)).any()
    out = jl.res_ln_like(jnp.asarray(state), jnp.asarray(thetas), jnp.asarray(draws), jnp.asarray(Sn),
                         jnp.asarray(rx), jnp.asarray(FBINS + 0.5*DELF), DURATION, 7.0, BOUNDS)
    assert_allclose(*rel(out, ref), rtol=1e-12)


def test_clamp_state_matches_apply_theta_lims():
    rng = np.random.default_rng(7)
    lims = np.array(jl.DEFAULT_THETA_LIMS)
    state = rng.uniform(lims[:, 0] - 0.5*(lims[:, 1] - lims[:, 0]), lims[:, 1] + 0.5*(lims[:, 1] - lims[:, 0]),
                        (500, 4))
    state[:4] = lims[:, 0]
    state[4:8] = lims[:, 1]
    ref = to_numpy(apply_theta_lims(xp.asarray(state.copy())))
    assert_array_equal(to_numpy(jl.clamp_state(jnp.asarray(state), jnp.asarray(lims))), ref)


def test_scatter_state_statistics():
    n = 20000
    maxL = np.tile([0.6, 0.7, 8.0, 2e-3], (n, 1))
    err, log_mask, lims = jl.scatter_settings({})
    s = to_numpy(jl.scatter_state(jax.random.key(0), jnp.asarray(maxL), err, log_mask, lims))
    dev = s - maxL
    dev[:, 3] = np.log10(s[:, 3]) - np.log10(maxL[:, 3])
    err = to_numpy(err)
    assert np.all(np.abs(dev.mean(axis=0)) < 5*err/np.sqrt(n))
    assert_allclose(dev.std(axis=0), err, rtol=0.03)
    s2 = to_numpy(jl.scatter_state(jax.random.key(1), jnp.asarray(maxL), *jl.scatter_settings({})))
    assert not np.allclose(s, s2)


# =============================================================================
# Combined, jitted
# =============================================================================

def test_ln_like_is_the_sum_of_the_references_and_compiles_once():
    Nr, Np = 3, 2
    state, rx, draws = res_setup(Nr, Np)
    fg_true, noise, _ = fg_setup(Nr, Np)
    fg_ref, n_ref = fg_reference(fg_true, noise, Nr, **FG_HP), nres_reference(40)
    c = consts_for(fg_ref, n_ref, rx=rx)
    ra = Res_Astro_Likelihood(xp.random.default_rng(1), xp.asarray(state), xp.asarray(FBINS), xp.asarray(rx),
                              duration=DURATION, scatter=False, dynamic_scatter=False)
    Sn = lisa_noise_psd(FBINS, cpu=True)
    rng = np.random.default_rng(8)
    for it in range(3):
        Nhat = rng.integers(30, 50, (Nr, Np))
        d = draws*10**(0.05*rng.standard_normal(draws.shape))
        terms = jl.ln_like(jnp.asarray(d), jnp.asarray(Nhat), jnp.asarray(state), jnp.asarray(THETAS[:Np]), c)
        ref = [fg_ref.ln_prob(xp.asarray(d)), n_ref.ln_prob(xp.asarray(Nhat)),
               ra.static_ln_conditional_prob(conditioned_prior(THETAS[:Np], Nr), xp.asarray(d), xp.asarray(Sn), 7.0)]
        assert_allclose(*rel(terms[0], ref[0]), rtol=1e-12)
        assert_allclose(*rel(terms[1], ref[1]), rtol=0, atol=negbin_atol(40, Nhat))
        assert_allclose(*rel(terms[2], ref[2]), rtol=1e-12)
        if it == 0:
            n_compiled = jl.ln_like._cache_size()
    assert jl.ln_like._cache_size() == n_compiled


def test_smooth_terms_have_finite_gradients():
    Nr, Np = 3, 2
    state, rx, draws = res_setup(Nr, Np)
    fg_true, noise, _ = fg_setup(Nr, Np)
    c = consts_for(fg_reference(fg_true, noise, Nr, **FG_HP), nres_reference(40), rx=rx)
    g_psd = jax.grad(lambda p: jnp.sum(jl.fg_ln_like(p, c.noise, c.log10_cgrid, c.ln_pgrid,
                                                     c.mu0, c.alpha, c.beta, c.nu)))(jnp.asarray(draws))
    g_n = jax.grad(lambda n: jnp.sum(jl.negbin_logpmf(c.N_obs, n, c.nb_alpha, c.nb_beta)))(jnp.full((Nr, Np), 40.0))
    s = jnp.asarray(state)
    s = s.at[:, :2].set(0.6)
    g_th = jax.grad(lambda t: jnp.sum(jl.gb_log_prior(s, t, BOUNDS)))(jnp.asarray(THETAS[:Np]))
    for g in (g_psd, g_n, g_th):
        assert np.isfinite(to_numpy(g)).all()


# =============================================================================
# PopModel on the jax backend
# =============================================================================

jax_backend_only = pytest.mark.skipif(backend.BACKEND != "jax", reason="needs PELARGIR_BACKEND=jax")


def popmodel_with_data(Nreal=2, scatter=False, dynamic_scatter=False):
    data_pm = PopModel(int(2e5), xp.random.default_rng(3), Nreal=1, block_after=4, jax_seed=4)
    fs, fg, Nres, res_idx, draw = data_pm.run_model(pop_theta=FIDUCIAL[None], return_extras=True)
    gb_thetas = draw[:, res_idx, 0, 0].T
    assert gb_thetas.shape[0] >= 4
    data = {'fg': fg, 'fg_sigma': xp.asarray(0.1), 'Nres': Nres, 'gb_thetas': gb_thetas}
    pm = PopModel(int(2e5), xp.random.default_rng(5), Nreal=Nreal, block_after=4, jax_seed=6,
                  res_scatter=scatter, res_dynamic_scatter=dynamic_scatter)
    pm.construct_likelihood(data, **FG_HP)
    return pm


@jax_backend_only
def test_popmodel_ln_prob_matches_the_cupy_terms_on_the_returned_draw():
    pm = popmodel_with_data()
    thetas = np.stack([FIDUCIAL, THETAS[1]])
    ln_p, (fs, fg, Nres) = pm.ln_prob(thetas, return_spec=True)
    fg, Nres = xp.asarray(fg), xp.asarray(Nres)
    ref = [pm.fg_like.ln_prob(fg), pm.Nres_like.ln_prob(Nres),
           pm.res_astro_like.static_ln_conditional_prob(conditioned_prior(thetas, 2), fg, pm.approx_lisa_psd,
                                                        pm.thresh_val)]
    atol = negbin_atol(pm.Nres_like.N_res_obs, Nres)
    t_fg, t_n, t_res = pm.last_ln_terms
    assert_allclose(*rel(t_fg, ref[0]), rtol=1e-12)
    assert_allclose(*rel(t_n, ref[1]), rtol=0, atol=atol)
    assert_allclose(*rel(t_res, ref[2]), rtol=1e-12)
    assert_allclose(to_numpy(ln_p), to_numpy(sum(ref)), rtol=1e-12, atol=atol)

    ln_p, (fs, fg, Nres) = pm.fg_N_ln_prob(thetas, return_spec=True)
    ref = pm.fg_like.ln_prob(xp.asarray(fg)) + pm.Nres_like.ln_prob(xp.asarray(Nres))
    assert_allclose(to_numpy(ln_p), to_numpy(ref), rtol=1e-12, atol=negbin_atol(pm.Nres_like.N_res_obs, Nres))


@jax_backend_only
def test_popmodel_dynamic_scatter_redraws_the_state():
    pm = popmodel_with_data(scatter=True, dynamic_scatter=True)
    s1, s2 = to_numpy(pm._jax_res_state()), to_numpy(pm._jax_res_state())
    assert s1.shape == to_numpy(pm.res_astro_like.theta_maxL).shape
    assert not np.allclose(s1, s2)
    assert np.isfinite(to_numpy(pm.ln_prob(FIDUCIAL[None]))).all()


@jax_backend_only
def test_popmodel_jax_likelihood_needs_two_realizations():
    pm = popmodel_with_data(Nreal=1)
    with pytest.raises(ValueError, match="Nreal >= 2"):
        pm.ln_prob(FIDUCIAL[None])
