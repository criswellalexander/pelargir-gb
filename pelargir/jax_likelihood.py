"""
Pure JAX likelihood terms (arXiv:2604.03390), ported from inference.FG_Likelihood,
Nres_Likelihood and Res_Astro_Likelihood; results match them on identical inputs
(tests/test_jax_likelihood.py).

- Foreground: the Poisson-marginalized t on log10 of the total PSD (Eq. B8), convolved with
  the log-normal PSD likelihood over a grid.
- N_res: the negative binomial marginal (Eq. A8).
- Resolved binaries: log sum_i pi(theta_i|Lambda) p(res|theta_i) (Eq. 20).

Shapes: psd (Nf', Nr, Np) is the foreground PSD on fbins[1:] (PopModel.run_model); Nres (Nr, Np);
state (Nres, 4) holds the resolved binaries' (m_1, m_2, d_L, a); thetas (Np, 6) are in
GalacticBinaryPrior.pop_params order.
"""
from functools import partial
from typing import NamedTuple

import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy, logsumexp, ndtr, log_ndtr

import numpy as np

import jax_population as jp
import jax_thresholding as jt

_LOG_SQRT_2PI = float(np.log(np.sqrt(2*np.pi)))
## utils.apply_theta_lims defaults: (m_1, m_2, d_L, a) in (Msun, Msun, kpc, AU)
DEFAULT_THETA_LIMS = ((0.17, 1.44), (0.17, 1.44), (1e-3, 100.0), (1e-4, 1e-2))
## utils.scatter_thetas defaults; a is scattered in log10
DEFAULT_SCATTER_ERR = (0.05, 0.05, 0.1, 0.001)


def to_jax(arr):
    '''numpy, cupy (zero-copy via DLPack), or scalar input as a float64/int JAX array.'''
    if jt._is_cupy(arr):
        return jt._to_jax(arr)
    return jnp.asarray(arr)


class LikelihoodConsts(NamedTuple):
    '''Data and settings the likelihood terms need; see make_consts.'''
    noise: object        ## (Nf',) noise PSD added to the foreground draws
    log10_cgrid: object  ## (Ngrid,) log10 of the PSD integration grid
    ln_pgrid: object     ## (Nf', 1, Ngrid) log-normal data likelihood on the grid
    mu0: object          ## (Nf', 1) normal-inverse-gamma hyperparameters of the t marginal
    alpha: object
    beta: object
    nu: object
    N_obs: object        ## observed N_res
    nb_alpha: object     ## Gamma hyperparameters of the N_res rate
    nb_beta: object
    Sn: object           ## (Nf,) noise PSD on fbins
    lisa_rx: object      ## (Nf,) LISA response on fbins
    edges: object        ## (Nf,) upper bin edges, fbins + delf/2
    duration: object     ## observation time [s] for the resolved-binary SNR
    rho_thresh: object
    bounds: tuple        ## (m_min, m_max, a_min, a_max, x0), jax_population.prior_bounds


def _host(v):
    return jp._host(v) if hasattr(v, 'shape') else np.asarray(v)


def _per_bin(v, Nf):
    '''Scalar or per-bin hyperparameter as an (Nf, 1) array.'''
    v = np.asarray(_host(v), dtype=np.float64).reshape(-1)
    if v.size not in (1, Nf):
        raise ValueError("hyperparameter has {} values; expected 1 or {}".format(v.size, Nf))
    return jnp.asarray(np.broadcast_to(v, (Nf,)).reshape(Nf, 1))


def make_consts(fg_like, nres_like, edges, Sn, lisa_rx, duration, rho_thresh, bounds):
    '''
    LikelihoodConsts from constructed inference.FG_Likelihood and Nres_Likelihood objects, so the
    hyperparameters are identical to theirs.

    Arguments
    -----------
    fg_like (FG_Likelihood), nres_like (Nres_Likelihood) : Constructed likelihoods.
    edges, Sn, lisa_rx (array) : Upper bin edges, noise PSD and LISA response on the full fbins grid.
    duration, rho_thresh (float) : As passed to Res_Astro_Likelihood.
    bounds (tuple) : jax_population.prior_bounds.
    '''
    t = fg_like.conditional_t
    noise = to_jax(np.asarray(_host(fg_like.noise_psd), dtype=np.float64))
    Nf = noise.shape[0]
    cgrid = np.asarray(_host(fg_like.cgrid), dtype=np.float64).reshape(-1)
    ln_pgrid = lognormal_grid(jnp.asarray(cgrid), to_jax(np.asarray(_host(fg_like.mu_vec), dtype=np.float64)),
                              _per_bin(fg_like.cov, Nf)[:, 0])
    dist = nres_like.base_dist
    return LikelihoodConsts(noise=noise, log10_cgrid=jnp.log10(jnp.asarray(cgrid)), ln_pgrid=ln_pgrid[:, None, :],
                            mu0=_per_bin(t.mu0, Nf), alpha=_per_bin(t.alpha, Nf), beta=_per_bin(t.beta, Nf),
                            nu=float(t.nu), N_obs=float(_host(nres_like.N_res_obs)),
                            nb_alpha=float(_host(dist.alpha)), nb_beta=float(_host(dist.beta)),
                            Sn=to_jax(np.asarray(_host(Sn), dtype=np.float64)),
                            lisa_rx=to_jax(np.asarray(_host(lisa_rx), dtype=np.float64)),
                            edges=to_jax(np.asarray(_host(edges), dtype=np.float64)),
                            duration=float(duration), rho_thresh=float(rho_thresh),
                            bounds=tuple(float(b) for b in bounds))


def negbin_logpmf(N_obs, Nhat, alpha, beta):
    '''
    log p(N_obs | {Nhat}_r), Eq. A8: NegBin(r = alpha + sum_r Nhat, p = beta'/(1+beta')), beta' = beta + Nr.
    Nhat has shape (Nr, Np); returns (Np,). As distributions.marginal_poisson_gamma._logpmf_of_N_hat.
    '''
    alphap = alpha + jnp.sum(jnp.asarray(Nhat, dtype=jnp.float64), axis=0)
    betap = beta + Nhat.shape[0]
    p = betap/(1 + betap)
    coeff = gammaln(alphap + N_obs) - gammaln(N_obs + 1) - gammaln(alphap)
    return coeff + alphap*jnp.log(p) + xlog1py(N_obs, -p)


def lognormal_grid(cgrid, mu_vec, sigma):
    '''
    Log-normal data likelihood of each grid PSD, shape (Nf', Ngrid), without the 1/x factor (it
    cancels against the log-spaced grid). As Likelihood.grid_lognormal_logpdf.

    cgrid (Ngrid,), mu_vec (Nf',) the observed total PSD, sigma (Nf',) or scalar.
    '''
    sigma = jnp.broadcast_to(sigma, mu_vec.shape)[:, None]
    return -(jnp.log(cgrid)[None, :] - jnp.log(mu_vec)[:, None])**2/(2*sigma**2)


def fg_ln_like(psd, noise, log10_cgrid, ln_pgrid, mu0, alpha, beta, nu):
    '''
    Foreground term, shape (Np,): sum over bins of log int p(d|S) p(S|{S_hat}_r) dS on the grid,
    with p(S|{S_hat}_r) the t marginal (Eq. B8) of log10(psd + noise) as distributions.vector_marginal_logt.

    psd (Nf', Nr, Np); noise (Nf',); log10_cgrid (Ngrid,); ln_pgrid (Nf', 1, Ngrid);
    mu0, alpha, beta (Nf', 1); nu scalar.
    '''
    Nr = psd.shape[1]
    y = jnp.log10(psd + noise[:, None, None])
    y_mean = jnp.mean(y, axis=1)                      ## (Nf', Np)
    y_dev2 = jnp.sum((y - y_mean[:, None])**2, axis=1)

    nup = nu + Nr
    alphap = alpha + Nr/2
    df = 2*alphap
    mup = (nu*mu0 + Nr*y_mean)/(nu + Nr)
    betap = beta + 0.5*y_dev2 + 0.5*((nu*Nr)/(nu + Nr))*(y_mean - mu0)**2
    ## as vector_marginal_logt: sigp is used in the place of the scale below
    sigp = (betap*(nup + 1))/(alphap*nup)

    x = log10_cgrid
    df3 = df[..., None]
    ln_coeff = (gammaln(0.5*df3 + 0.5) - gammaln(0.5*df3)) - 0.5*(jnp.log(df3) + jnp.log(jnp.pi)) \
               - jnp.log(sigp)[..., None] - x
    ln_t = ln_coeff - 0.5*(df3 + 1)*jnp.log1p((((x - mup[..., None])/sigp[..., None])**2)/df3)
    return jnp.sum(logsumexp(ln_t + ln_pgrid, axis=-1), axis=0)


def log_gauss_mass(a, b):
    '''log(Phi(b) - Phi(a)) for a <= b, as distributions._log_gauss_mass.'''
    a, b = jnp.broadcast_arrays(a, b)

    def left(a, b):
        la, lb = log_ndtr(a), log_ndtr(b)
        return lb + jnp.log1p(-jnp.exp(la - lb))

    central = jnp.log1p(-ndtr(a) - ndtr(-b))
    return jnp.where(b <= 0, left(a, b), jnp.where(a > 0, left(-b, -a), central))


def truncnorm_logpdf(x, loc, scale, lo, hi):
    '''As distributions.truncnorm._logpdf: -inf outside [lo, hi].'''
    z = (x - loc)/scale
    lp = -0.5*z**2 - _LOG_SQRT_2PI - jnp.log(scale) - log_gauss_mass((lo - loc)/scale, (hi - loc)/scale)
    return jnp.where((x >= lo) & (x <= hi), lp, -jnp.inf)


def mixture_logpdf(x, x0, r_bulge, rh_disk, q_bd):
    '''As distributions.gaussian_exponential_mixture._logpdf (bulge toward the observer, disk both ways).'''
    x_tw = x - x0
    x_aw = -x - x0
    bulge_tw = jnp.log(q_bd) + (-0.5*(x_tw/r_bulge)**2 - _LOG_SQRT_2PI - jnp.log(r_bulge))
    disk_tw = jnp.log(1 - q_bd) - jnp.log(rh_disk) - jnp.abs(x_tw/rh_disk) - jnp.log(2)
    disk_aw = jnp.log(1 - q_bd) - jnp.log(rh_disk) - jnp.abs(x_aw/rh_disk) - jnp.log(2)
    return logsumexp(jnp.stack(jnp.broadcast_arrays(bulge_tw, disk_tw, disk_aw)), axis=0)


def powerlaw_logpdf(x, alpha, loc, scale):
    '''As distributions.powerlaw._logpdf: p(x) ∝ ((x - loc)/scale)^alpha.'''
    return jnp.log(alpha + 1) + xlogy(alpha, (x - loc)/scale) - jnp.log(scale)


def gb_log_prior(state, thetas, bounds):
    '''log pi(theta_i | Lambda) summed over (m_1, m_2, d_L, a), shape (Nres, Np).'''
    p = jp.conditional_params(thetas[None], bounds)   ## fields (1, Np)
    m_1, m_2, d_L, a = (state[:, i, None] for i in range(4))
    return (truncnorm_logpdf(m_1, p.m_loc, p.m_scale, p.m_min, p.m_max)
            + truncnorm_logpdf(m_2, p.m_loc, p.m_scale, p.m_min, p.m_max)
            + mixture_logpdf(d_L, p.x0, p.r_bulge, p.rh_disk, p.q_bd)
            + powerlaw_logpdf(a, p.a_alpha, p.a_loc, p.a_scale))


def res_prob(state, psd, Sn, lisa_rx, edges, duration, rho_thresh):
    '''
    Fraction of realizations in which each resolved binary is resolved, shape (Nres, Np).
    As Res_Astro_Likelihood.get_phenom and res_prob: bin 0 and above-band binaries read the
    nearest modelled bin.
    '''
    A, f = jp.amp_freq(state.T)
    Nf = edges.shape[0]
    k = jnp.clip(jnp.digitize(f, edges), 1, Nf - 1)
    S = Sn[1:, None, None] + psd
    rho = jnp.sqrt((duration*lisa_rx[k]*A**2)[:, None, None]/S[k - 1])
    return jnp.mean((rho >= rho_thresh).astype(jnp.float64), axis=1)


def res_ln_like(state, thetas, psd, Sn, lisa_rx, edges, duration, rho_thresh, bounds):
    '''Resolved-binary term (Eq. 20), shape (Np,): log sum_i pi(theta_i|Lambda) p(res|theta_i).'''
    return logsumexp(gb_log_prior(state, thetas, bounds),
                     b=res_prob(state, psd, Sn, lisa_rx, edges, duration, rho_thresh), axis=0)


@partial(jax.jit, static_argnames=('include_res',))
def ln_like(psd, Nres, state, thetas, consts, include_res=True):
    '''
    The three likelihood terms, each of shape (Np,): (foreground, N_res, resolved binaries).
    The resolved-binary term is zero if include_res is False (state is then unused).
    '''
    c = consts
    ln_fg = fg_ln_like(psd, c.noise, c.log10_cgrid, c.ln_pgrid, c.mu0, c.alpha, c.beta, c.nu)
    ln_nres = negbin_logpmf(c.N_obs, Nres, c.nb_alpha, c.nb_beta)
    if not include_res:
        return ln_fg, ln_nres, jnp.zeros_like(ln_fg)
    ln_res = res_ln_like(state, thetas, psd, c.Sn, c.lisa_rx, c.edges, c.duration, c.rho_thresh, c.bounds)
    return ln_fg, ln_nres, ln_res


def clamp_state(state, lims):
    '''As utils.apply_theta_lims: values <= lo go to 1.0000001*lo, values >= hi to 0.9999999*hi.'''
    lo, hi = lims[:, 0], lims[:, 1]
    below = state <= lo
    above = state >= hi
    return jnp.where(below, 1.0000001*lo, jnp.where(above, 0.9999999*hi, state))


@jax.jit
def scatter_state(key, theta_maxL, err, log_mask, lims):
    '''
    A new resolved-binary state: theta_maxL (Nres, 4) plus Gaussian scatter err (4,), applied in
    log10 where log_mask (4,) is True, then clamped to lims (4, 2). As utils.scatter_thetas(bound=True).
    '''
    t = jnp.where(log_mask, jnp.log10(theta_maxL), theta_maxL)
    s = t + err*jax.random.normal(key, t.shape, dtype=jnp.float64)
    s = jnp.where(log_mask, 10**s, s)
    return clamp_state(s, lims)


def scatter_settings(kwargs):
    '''(err, log_mask, lims) for scatter_state from Res_Astro_Likelihood's scatter_thetas kwargs.'''
    err = np.asarray(_host(kwargs.get('err', DEFAULT_SCATTER_ERR)), dtype=np.float64)
    log_mask = np.zeros(4, dtype=bool)
    log_mask[list(kwargs.get('log_args_idx', [-1]))] = True
    lims = kwargs.get('theta_lims', 'default')
    lims = np.asarray(DEFAULT_THETA_LIMS if (type(lims) is str and lims == 'default') else _host(lims), dtype=np.float64)
    return jnp.asarray(err), jnp.asarray(log_mask), jnp.asarray(lims)
