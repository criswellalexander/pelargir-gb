"""
Checks of the Poisson-marginalized conditional spectral prior (vector_marginal_t,
vector_marginal_logt) and of the foreground likelihood integral that uses it.

- The normal-inverse-gamma posterior predictive is a Student-t with df = 2 alpha',
  location mu' and SQUARED scale beta'(nu'+1)/(alpha' nu'); the reference formula is
  itself verified against a direct quadrature over the variance.
- vector_marginal_logt is the density in S (not log10 S) and must integrate to 1 over S.
- FG_Likelihood must reproduce int p(d|S) p(S|Shat) dS computed independently.
"""
import os
import sys

os.environ["PELARGIR_GPU"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import quad
from numpy.testing import assert_allclose

from distributions import vector_marginal_t, vector_marginal_logt
from inference import FG_Likelihood

NF, NREAL, NPAR = 3, 4, 2
MU0, ALPHA, BETA = -40.0, 1.0, 0.15


def nig_predictive(y, mu0, nu, alpha, beta):
    """Posterior-predictive t parameters (df, loc, scale) of a normal-inverse-gamma model,
    from draws y of shape (..., N, ...) along axis=1."""
    N = y.shape[1]
    ybar = y.mean(axis=1)
    ss_dev = ((y - np.expand_dims(ybar, 1))**2).sum(axis=1)
    kappa_n = nu + N
    mu_n = (nu*mu0 + N*ybar)/kappa_n
    alpha_n = alpha + N/2
    beta_n = beta + 0.5*ss_dev + 0.5*nu*N*(ybar - mu0)**2/kappa_n
    scale = np.sqrt(beta_n*(kappa_n + 1)/(alpha_n*kappa_n))
    return 2*alpha_n, mu_n, scale, alpha_n, beta_n, kappa_n


@pytest.fixture
def spectra():
    rng = np.random.default_rng(42)
    ## lognormal-ish draws around a few 1e-38 -- 1e-37 levels, with ~0.2 dex scatter
    base = 10**np.array([-38.0, -37.5, -37.0])
    return base[:, None, None]*10**(0.2*rng.standard_normal((NF, NREAL, NPAR)))


def test_nig_predictive_reference_matches_quadrature():
    """The reference t (scale = sqrt(beta'(nu'+1)/(alpha' nu'))) is the NIG predictive:
    p(y) = int IG(s2; alpha', beta') N(y; mu', s2 (1 + 1/nu')) ds2."""
    rng = np.random.default_rng(1)
    y = (-37.3 + 0.3*rng.standard_normal(NREAL))[None, :]
    df, loc, scale, a_n, b_n, k_n = nig_predictive(y, MU0, 1e-10, ALPHA, BETA)
    df, loc, scale, a_n, b_n, k_n = (np.squeeze(v) for v in (df, loc, scale, a_n, b_n, k_n))
    for yy in loc + scale*np.array([-4.0, -1.3, 0.0, 0.7, 3.0]):
        integrand = lambda s2: ss.invgamma.pdf(s2, a_n, scale=b_n)*ss.norm.pdf(yy, loc, np.sqrt(s2*(1 + 1/k_n)))
        p_quad = quad(integrand, 0, np.inf, limit=200)[0]
        assert_allclose(ss.t.pdf(yy, df, loc, scale), p_quad, rtol=1e-6)
        ## using the squared scale as the scale is not the predictive
        assert not np.isclose(ss.t.pdf(yy, df, loc, scale**2), p_quad, rtol=1e-2)


def test_vector_marginal_logt_matches_scipy(spectra):
    dist = vector_marginal_logt(np.random.default_rng(0), MU0, NREAL, alpha=ALPHA, beta=BETA)
    dist.update(spectra)
    S = np.logspace(-40, -35, 301)[None, :]
    logpdf = dist.logpdf(S)
    assert logpdf.shape == (NF, NPAR, S.shape[-1])

    df, loc, scale, *_ = nig_predictive(np.log10(spectra), MU0, dist.nu, ALPHA, BETA)
    ref = (ss.t.logpdf(np.log10(S)[None, :, :], df, loc=loc[..., None], scale=scale[..., None])
           - np.log(S) - np.log(np.log(10)))
    assert_allclose(logpdf, ref, rtol=1e-10, atol=1e-10)


def test_vector_marginal_logt_integrates_to_one_in_S(spectra):
    dist = vector_marginal_logt(np.random.default_rng(0), MU0, NREAL, alpha=ALPHA, beta=BETA)
    dist.update(spectra)
    ## wide enough to hold the t tails (df = 2 alpha' = 6 here)
    S = np.logspace(-80, 5, 200001)
    pdf = np.exp(dist.logpdf(S[None, :]))
    assert_allclose(np.trapezoid(pdf, S, axis=-1), 1.0, rtol=1e-5)


def test_vector_marginal_t_matches_scipy():
    rng = np.random.default_rng(3)
    theta = 5.0 + rng.standard_normal((NF, NREAL, NPAR))
    dist = vector_marginal_t(rng, 0.0, NREAL, alpha=ALPHA, beta=BETA)
    dist.update(theta)
    df, loc, scale, *_ = nig_predictive(theta, 0.0, dist.nu, ALPHA, BETA)
    x = np.linspace(0, 10, 7)
    for xx in x:
        assert_allclose(dist.logpdf(np.array(xx)), ss.t.logpdf(xx, df, loc=loc, scale=scale), rtol=1e-10)


def test_fg_likelihood_matches_direct_integral(spectra):
    """ln_prob must equal sum_f ln int L(d_f|S) p(S|Shat_f) dS, with L the (unnormalized)
    lognormal data likelihood exp(-(ln S - ln d)^2 / 2 sigma^2), evaluated here by quadrature in y = log10 S."""
    noise = np.array([2e-38, 3e-38, 5e-38])
    fg_data = np.array([1.2e-38, 3e-38, 1.1e-37])
    sigma = 0.1
    like = FG_Likelihood(fg_data, np.array(sigma), noise, Nreal=NREAL)
    ln_p = like.ln_prob(spectra)
    assert ln_p.shape == (NPAR,)

    df, loc, scale, *_ = nig_predictive(np.log10(spectra + noise[:, None, None]), -40, 1e-10, 1, 0.15)
    d = fg_data + noise
    ref = np.zeros(NPAR)
    for f in range(NF):
        for j in range(NPAR):
            integrand = lambda y: (np.exp(-(np.log(10)*y - np.log(d[f]))**2/(2*sigma**2))
                                   *ss.t.pdf(y, df, loc[f, j], scale[f, j]))
            yc = np.log10(d[f])
            ref[j] += np.log(quad(integrand, yc - 1, yc + 1, points=[yc], epsabs=0, epsrel=1e-10)[0])
    assert_allclose(ln_p, ref, rtol=0, atol=1e-6)
