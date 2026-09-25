"""
Checks of the Poisson-marginalized conditional spectral prior (vector_marginal_t).

- The normal-inverse-gamma posterior predictive is a Student-t with df = 2 alpha',
  location mu' and SQUARED scale beta'(nu'+1)/(alpha' nu'); the reference formula is
  itself verified against a direct quadrature over the variance.
"""
import os
import sys

os.environ["PELARGIR_BACKEND"] = "numpy"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir"))

import numpy as np
import scipy.stats as ss
from scipy.integrate import quad
from numpy.testing import assert_allclose

from distributions import vector_marginal_t

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


def test_vector_marginal_t_matches_scipy():
    rng = np.random.default_rng(3)
    theta = 5.0 + rng.standard_normal((NF, NREAL, NPAR))
    dist = vector_marginal_t(rng, 0.0, NREAL, alpha=ALPHA, beta=BETA)
    dist.update(theta)
    df, loc, scale, *_ = nig_predictive(theta, 0.0, dist.nu, ALPHA, BETA)
    x = np.linspace(0, 10, 7)
    for xx in x:
        assert_allclose(dist.logpdf(np.array(xx)), ss.t.logpdf(xx, df, loc=loc, scale=scale), rtol=1e-10)
