"""
Inference with the flow emulator on the toy model (arXiv:2604.03390, Sec. 4): a toy dataset on the
emulator's grid and the log posterior shared by every sampler.

The flow is the population model: it gives p(N_res per band, S_gw per bin | Lambda, rho_thresh,
lambda_tot), replacing the semi-analytic estimators and their NegBin and t marginals (Eqs. A8, B8).
S_gw is therefore a latent sampled alongside the eight context parameters, as y = log10 S_gw. The
resolved-GB state (N_res, theta_i) is fixed. The log posterior over (theta, y) is

    ln pi(theta)                                    hyperprior (flow_data.HYPERPRIOR)
  + ln p(N_res, S_gw | theta) + sum ln(S_gw ln 10)  flow density, with the Jacobian of y = log10 S_gw
  + ln L(d | S_gw)                                  log-normal data likelihood, natural-log width sigma
  + ln sum_i pi(theta_i | Lambda) p(res | theta_i)  resolved binaries (Eq. 20), the step in SNR
                                                    smoothed by a logistic of width w

The data follow run_pelargir: d is the unperturbed S_gw + S_n and sigma (0.1) is a natural-log
width. Every function is pure JAX, batched over rows (Eryn walkers) and differentiable (HMC).
"""
import json
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import numpy as np

from . import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp, ndtr

from . import flow_data as fd
from . import flax_flows as ff
from . import jax_likelihood as jl
from . import jax_population as jp

## Table 1 of arXiv:2604.03390 (the paper's toy-model truth)
FIDUCIAL = (0.6, 0.15, 3.31, 0.75, 0.33, 0.5)
_LN10 = float(np.log(10.0))


# =============================================================================
# Toy dataset
# =============================================================================

@dataclass
class ToyData:
    '''

    A toy-model dataset on the flow grid: the observed total PSD, the fixed resolved-GB state, and
    the truth it was simulated from.

    Arguments
    ------------------------------
    fbins : numpy.ndarray
    	(Nf,) model bin centres; bin 0 is the below-band catch-all, so fbins[1:] is modelled.
    data_psd : numpy.ndarray
    	(Nf-1,) observed total PSD d on fbins[1:] [1/Hz], the unperturbed S_gw + S_n (as run_pelargir).
    sigma : float
    	Natural-log width of the log-normal data likelihood.
    noise_psd : numpy.ndarray
    	(Nf-1,) instrumental noise PSD S_n on fbins[1:] [1/Hz].
    nres_f : numpy.ndarray
    	(Nf-1,) resolved count in each modelled bin.
    theta_res : numpy.ndarray
    	(N_res, 4) resolved binaries' (m_1 [Msun], m_2 [Msun], d_L [kpc], a [AU]), modelled bins only.
    truth_theta : numpy.ndarray
    	(8,) true context, flow_data.CONTEXT_NAMES order.
    truth_S : numpy.ndarray
    	(Nf-1,) true foreground PSD S_gw on fbins[1:] [1/Hz].
    meta : dict
    	Seed, drawn N_tot and conventions. (Default: {})

    Returns
    ------------------------------
    None

    '''
    fbins: np.ndarray
    data_psd: np.ndarray
    sigma: float
    noise_psd: np.ndarray
    nres_f: np.ndarray
    theta_res: np.ndarray
    truth_theta: np.ndarray
    truth_S: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def fs(self):
        return self.fbins[1:]

    def N_bands(self, bands):
        '''

        Observed resolved counts per band.

        Arguments
        ------------------------------
        bands : list of flow_data.FrequencyBand
        	Bands on fbins[1:] (e.g. an emulator's).

        Returns
        ------------------------------
        N_bands : numpy.ndarray
        	(n_bands,) resolved count in each band.

        '''
        return np.array([self.nres_f[b.slice].sum() for b in bands], dtype=np.float64)

    def save(self, path):
        np.savez(path, fbins=self.fbins, data_psd=self.data_psd, sigma=self.sigma, noise_psd=self.noise_psd,
                 nres_f=self.nres_f, theta_res=self.theta_res, truth_theta=self.truth_theta, truth_S=self.truth_S,
                 meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d['fbins'], d['data_psd'], float(d['sigma']), d['noise_psd'], d['nres_f'], d['theta_res'],
                   d['truth_theta'], d['truth_S'], json.loads(str(d['meta'])))


def simulate_toy_data(theta6=FIDUCIAL, rho_thresh=7.0, lambda_tot=1e7, fbins=None, seed=150914, sigma=0.1):
    '''

    Simulate one galaxy with the JAX forward model (the training set's constants) and build the toy
    dataset: N_tot ~ Poisson(lambda_tot), the thresholded foreground and resolved counts, and the
    resolved binaries' true parameters as the fixed state.

    Arguments
    ------------------------------
    theta6 : sequence of float
    	True Lambda, GalacticBinaryPrior.pop_params order. (Default: FIDUCIAL, Table 1)
    rho_thresh : float
    	True SNR threshold. (Default: 7.0)
    lambda_tot : float
    	Rate of N_tot ~ Poisson(lambda_tot). (Default: 1e7)
    fbins : numpy.ndarray or None
    	Model bin centres; None uses flow_data.model_fbins(). (Default: None)
    seed : int
    	Seed of N_tot and the galaxy draw. (Default: 150914)
    sigma : float
    	Natural-log width of the data likelihood (run_pelargir's --logsigma). (Default: 0.1)

    Returns
    ------------------------------
    data : ToyData
    	The dataset.

    '''
    fbins = fd.model_fbins() if fbins is None else np.asarray(fbins, dtype=np.float64)
    consts, bounds = fd._forward_consts(fbins)
    edges, noise, rx, duration, duration_eff = consts
    k_N, k_gal = jax.random.split(jax.random.key(seed))
    N_tot = int(jax.random.poisson(k_N, lambda_tot))
    theta = np.asarray(theta6, dtype=np.float64).reshape(1, 6)
    _, fg, mask, nres_f = jp.jax_forward_model(k_gal, theta, np.full((1, 1), N_tot), bounds, edges, noise, rx, duration,
                                               duration_eff, snr_thresh=rho_thresh, return_mask=True,
                                               return_nres_f=True)
    ## galaxy 0 of the call, as the forward model drew it (the PopModel convention)
    draw = np.asarray(jp.sample_galaxy_draw(k_gal, 0, theta[0], N_tot, bounds))
    resolved = np.asarray(mask)[:N_tot]
    A, f = (np.asarray(x) for x in jp.amp_freq(jnp.asarray(draw[:, resolved])))
    ## bin-0 (below-band) resolved sources are not modelled by the flow, so they leave the state
    in_band = np.digitize(f, edges) >= 1
    nres_f = np.asarray(nres_f)[1:].astype(np.int64)
    theta_res = draw[:, resolved][:, in_band].T
    if theta_res.shape[0] != nres_f.sum():
        raise RuntimeError("resolved state has {} binaries but the modelled bins hold {}".format(
            theta_res.shape[0], nres_f.sum()))
    S_true = np.asarray(fg)[1:]*duration_eff
    truth = np.concatenate([theta[0], [rho_thresh, np.log10(lambda_tot)]])
    meta = dict(seed=seed, N_tot=N_tot, lambda_tot=lambda_tot, data="unperturbed S_gw + S_n (run_pelargir)",
                sigma="natural-log width (run_pelargir)", duration=duration)
    return ToyData(fbins, S_true + noise[1:], float(sigma), noise[1:].copy(), nres_f, theta_res, truth, S_true, meta)


# =============================================================================
# Hyperprior
# =============================================================================

def _prior_spec():
    '''

    The flow_data.HYPERPRIOR entries as (kind, parameters) in CONTEXT_NAMES order, for the JAX logpdf.

    Arguments
    ------------------------------
    None

    Returns
    ------------------------------
    spec : list of tuple
    	('uniform', lo, hi), ('invgamma', a) or ('truncnorm', lo, hi, loc, scale) per parameter.

    '''
    spec = []
    for name in fd.CONTEXT_NAMES:
        d = fd.HYPERPRIOR[name]
        loc, scale = d.kwds.get('loc', 0.0), d.kwds.get('scale', 1.0)
        kind = d.dist.name
        if kind == 'uniform':
            spec.append(('uniform', loc, loc + scale))
        elif kind == 'invgamma' and loc == 0.0 and scale == 1.0:
            spec.append(('invgamma', d.args[0]))
        elif kind == 'truncnorm':
            a, b = d.args
            spec.append(('truncnorm', loc + a*scale, loc + b*scale, loc, scale))
        else:
            raise NotImplementedError("no JAX logpdf for hyperprior {} ({})".format(name, kind))
    return spec


_PRIOR = _prior_spec()


def log_hyperprior(theta):
    '''

    log of flow_data.HYPERPRIOR (independent parameters), -inf outside its support.

    Arguments
    ------------------------------
    theta : jax.Array
    	(M, 8) contexts, flow_data.CONTEXT_NAMES order.

    Returns
    ------------------------------
    ln_prior : jax.Array
    	(M,) log prior density.

    '''
    total = 0.0
    for j, p in enumerate(_PRIOR):
        x = theta[:, j]
        if p[0] == 'uniform':
            lp = jnp.where((x >= p[1]) & (x <= p[2]), -np.log(p[2] - p[1]), -jnp.inf)
        elif p[0] == 'invgamma':
            a = p[1]
            xs = jnp.where(x > 0, x, 1.0)
            lp = jnp.where(x > 0, -(a + 1)*jnp.log(xs) - 1/xs - gammaln(a), -jnp.inf)
        else:
            lo, hi, loc, scale = p[1:]
            z = (x - loc)/scale
            ## log of the normal's mass on [lo, hi]
            ln_mass = jnp.log(ndtr((hi - loc)/scale) - ndtr((lo - loc)/scale))
            lp = jnp.where((x >= lo) & (x <= hi), -0.5*z**2 - 0.5*np.log(2*np.pi) - np.log(scale) - ln_mass, -jnp.inf)
        total = total + lp
    return total


# =============================================================================
# Log posterior
# =============================================================================

class PosteriorConsts(NamedTuple):
    '''Data-side constants of FlowPosterior (device arrays), passed to its jitted functions.'''
    N_bands: object    ## (n_bands,) observed resolved counts per band
    ln_d: object       ## (Nf',) log of the observed total PSD
    noise: object      ## (Nf',) noise PSD S_n
    sigma: float       ## natural-log width of the data likelihood
    res_T_rx_A2: object  ## (N_res,) T rx_k A_i^2 of each resolved binary
    res_k: object      ## (N_res,) index into fbins[1:] of each resolved binary's bin
    theta_res: object  ## (N_res, 4) resolved binaries' (m_1, m_2, d_L, a)
    bounds: tuple      ## jax_population.prior_bounds


def _ln_flow(models, transforms, ctx_transforms, theta, y, N_bands, slices, n_quad):
    '''

    Flow density of (N_res per band, S_gw = 10**y) given theta, plus the Jacobian of y = log10 S_gw.

    Arguments
    ------------------------------
    models, transforms, ctx_transforms : list
    	Each band's flax model and fitted transforms (passed as arguments, never closed over).
    theta : jax.Array
    	(M, 8) contexts.
    y : jax.Array
    	(M, Nf') log10 S_gw.
    N_bands : jax.Array
    	(n_bands,) observed resolved counts per band.
    slices : tuple of tuple
    	Each band's (start, stop) on fbins[1:].
    n_quad : int
    	Gauss-Legendre nodes of the count dequantization.

    Returns
    ------------------------------
    ln_flow : jax.Array
    	(M,) log density in (N_res, y).

    '''
    S = 10**y
    N = jnp.broadcast_to(N_bands, (theta.shape[0], N_bands.shape[0]))
    total = jnp.sum(jnp.log(S*_LN10), axis=1)
    for j, (lo, hi) in enumerate(slices):
        total = total + ff._log_prob_band(models[j], transforms[j], ctx_transforms[j], theta, N[:, j], S[:, lo:hi],
                                          n_quad)
    return total


def _ln_data(y, c):
    '''(M,) log-normal data likelihood of S_gw = 10**y: -sum (ln d - ln(S_gw + S_n))^2 / (2 sigma^2).'''
    return -jnp.sum((c.ln_d - jnp.log(10**y + c.noise))**2, axis=1)/(2*c.sigma**2)


def _ln_res(theta, y, c, width):
    '''

    Resolved-GB term (Eq. 20), log sum_i pi(theta_i | Lambda) p(res | theta_i, S_gw), with
    p(res) = sigmoid((rho_i - rho_thresh)/width), or the exact step if width is None.

    Arguments
    ------------------------------
    theta : jax.Array
    	(M, 8) contexts (Lambda and rho_thresh are used).
    y : jax.Array
    	(M, Nf') log10 S_gw.
    c : PosteriorConsts
    	Constants of the fixed state.
    width : float or None
    	Logistic width in SNR; None for the step function.

    Returns
    ------------------------------
    ln_res : jax.Array
    	(M,) the term.

    '''
    ## SNR of each fixed binary against the noise plus this foreground, as jax_likelihood.res_prob
    rho = jnp.sqrt(c.res_T_rx_A2[None, :]/(c.noise[c.res_k][None, :] + 10**y[:, c.res_k]))   ## (M, N_res)
    lp = jl.gb_log_prior(c.theta_res, theta[:, :6], c.bounds).T                              ## (M, N_res)
    z = rho - theta[:, 6:7]
    if width is None:
        return logsumexp(lp, b=(z >= 0).astype(lp.dtype), axis=1)
    return logsumexp(lp + jax.nn.log_sigmoid(z/width), axis=1)


@partial(jax.jit, static_argnames=('slices', 'n_quad', 'width'))
def _terms(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width):
    '''Every term of the log posterior, each (M,), as a dict.'''
    return dict(prior=log_hyperprior(theta),
                flow=_ln_flow(models, transforms, ctx_transforms, theta, y, c.N_bands, slices, n_quad),
                data=_ln_data(y, c), res=_ln_res(theta, y, c, width))


def _ln_post(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width):
    t = _terms(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width)
    total = t['flow'] + t['data'] + t['res']
    ## outside the prior's support the other terms may be NaN
    return jnp.where(jnp.isfinite(t['prior']), t['prior'] + total, -jnp.inf)


_ln_post_jit = jax.jit(_ln_post, static_argnames=('slices', 'n_quad', 'width'))


@partial(jax.jit, static_argnames=('slices', 'n_quad'))
def _ln_flow_prior(models, transforms, ctx_transforms, c, theta, y, slices, n_quad):
    '''(M,) hyperprior + flow density (with the Jacobian of y): the untempered part of a population prior.'''
    lp = log_hyperprior(theta)
    total = lp + _ln_flow(models, transforms, ctx_transforms, theta, y, c.N_bands, slices, n_quad)
    return jnp.where(jnp.isfinite(lp), total, -jnp.inf)


@partial(jax.jit, static_argnames=('width',))
def _ln_data_res(c, theta, y, width):
    '''(M,) data and resolved-GB terms (no flow evaluation).'''
    return _ln_data(y, c) + _ln_res(theta, y, c, width)


@partial(jax.jit, static_argnames=('nth', 'slices', 'n_quad', 'width'))
def _hessian(models, transforms, ctx_transforms, c, x, nth, slices, n_quad, width):
    '''Hessian of ln_post at one point x = (theta, y) concatenated.'''
    one = lambda v: _ln_post(models, transforms, ctx_transforms, c, v[None, :nth], v[None, nth:], slices, n_quad, width)[0]
    return jax.hessian(one)(x)


@partial(jax.jit, static_argnames=('slices', 'n_quad', 'width'))
def _conditional_hessians(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width):
    '''y-Hessians at one point of ln_flow_prior and of ln_data_res, which tempering weights separately.'''
    prior = lambda v: _ln_flow_prior(models, transforms, ctx_transforms, c, theta[None], v[None], slices, n_quad)[0]
    like = lambda v: _ln_data_res(c, theta[None], v[None], width)[0]
    return jax.hessian(prior)(y), jax.hessian(like)(y)


@partial(jax.jit, static_argnames=('slices', 'n_quad', 'width'))
def _ln_like(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width):
    t = _terms(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width)
    return t['flow'] + t['data'] + t['res']


@partial(jax.jit, static_argnames=('slices', 'n_quad', 'width'))
def _value_and_grad(models, transforms, ctx_transforms, c, theta, y, slices, n_quad, width):
    one = lambda th, yy: _ln_post(models, transforms, ctx_transforms, c, th[None], yy[None], slices, n_quad, width)[0]
    return jax.value_and_grad(one, argnums=(0, 1))(theta, y)


class FlowPosterior:
    '''

    The toy model's log posterior over theta (the flow's eight context parameters) and y = log10 S_gw
    (one per modelled bin), given a flax flow emulator and a ToyData. All methods are jitted and
    batched over rows; the emulator's parameters are passed to the jitted functions as arguments.

    Arguments
    ------------------------------
    emulator : flax_flows.BandedFlowEmulator
    	The population model.
    data : ToyData
    	Dataset on the emulator's grid.
    width : float
    	Logistic width (SNR) replacing the step in the resolved-GB term. (Default: 0.1)
    n_quad : int
    	Gauss-Legendre nodes of the flow's count dequantization. (Default: 32)
    duration : float
    	Observation time [s] of the resolved binaries' SNR (the flow's training value). (Default: flow_data.DURATION)

    Returns
    ------------------------------
    None

    '''

    def __init__(self, emulator, data, width=0.1, n_quad=32, duration=fd.DURATION):
        if not isinstance(emulator, ff.BandedFlowEmulator):
            raise TypeError("FlowPosterior needs a flax emulator (flax_flows.BandedFlowEmulator)")
        if not np.allclose(emulator.fbins, data.fbins, rtol=1e-12, atol=0):
            raise ValueError("the emulator and the data are on different frequency grids")
        self.emulator, self.data = emulator, data
        self.width, self.n_quad, self.duration = float(width), int(n_quad), float(duration)
        self.models = [f.model for f in emulator.flows]
        self.transforms = [f.transform for f in emulator.flows]
        self.ctx_transforms = [f.context_transform for f in emulator.flows]
        self.slices = tuple((b.start, b.stop) for b in emulator.bands)

        consts, bounds = fd._forward_consts(data.fbins)
        edges, _, rx = consts[:3]
        Nf = edges.shape[0]
        A, f = (np.asarray(x) for x in jp.amp_freq(jnp.asarray(data.theta_res.T)))
        ## each binary's bin on the full grid, clipped as jax_likelihood.res_prob (bin 0 reads bin 1)
        k = np.clip(np.digitize(f, edges), 1, Nf - 1)
        self.consts = PosteriorConsts(N_bands=jnp.asarray(data.N_bands(emulator.bands)),
                                      ln_d=jnp.asarray(np.log(data.data_psd)), noise=jnp.asarray(data.noise_psd),
                                      sigma=float(data.sigma), res_T_rx_A2=jnp.asarray(duration*rx[k]*A**2),
                                      res_k=jnp.asarray(k - 1), theta_res=jnp.asarray(data.theta_res),
                                      bounds=tuple(float(b) for b in bounds))

    @property
    def truth(self):
        '''(theta, y) of the simulation truth, each with a leading row axis.'''
        return jnp.asarray(self.data.truth_theta)[None], jnp.log10(jnp.asarray(self.data.truth_S))[None]

    def _args(self):
        return self.models, self.transforms, self.ctx_transforms, self.consts

    def terms(self, theta, y, width='default'):
        '''

        Each term of the log posterior.

        Arguments
        ------------------------------
        theta : array
        	(M, 8) contexts.
        y : array
        	(M, Nf') log10 S_gw.
        width : float, None or 'default'
        	Logistic width of the resolved term; None for the exact step. (Default: 'default', self.width)

        Returns
        ------------------------------
        terms : dict
        	'prior', 'flow' (with the Jacobian of y), 'data' and 'res', each (M,).

        '''
        w = self.width if width == 'default' else width
        return _terms(*self._args(), jnp.atleast_2d(theta), jnp.atleast_2d(y), self.slices, self.n_quad, w)

    def ln_post(self, theta, y):
        '''

        The log posterior density in (theta, y).

        Arguments
        ------------------------------
        theta : array
        	(M, 8) contexts.
        y : array
        	(M, Nf') log10 S_gw.

        Returns
        ------------------------------
        ln_post : jax.Array
        	(M,) log posterior, -inf outside the prior's support.

        '''
        return _ln_post_jit(*self._args(), jnp.atleast_2d(theta), jnp.atleast_2d(y), self.slices, self.n_quad,
                            self.width)

    def ln_like(self, theta, y):
        '''

        The log posterior without the hyperprior: flow (with the Jacobian of y), data and resolved terms.

        Arguments
        ------------------------------
        theta : array
        	(M, 8) contexts, inside the prior's support.
        y : array
        	(M, Nf') log10 S_gw.

        Returns
        ------------------------------
        ln_like : jax.Array
        	(M,) the sum of the three terms.

        '''
        return _ln_like(*self._args(), jnp.atleast_2d(theta), jnp.atleast_2d(y), self.slices, self.n_quad,
                        self.width)

    def ln_flow_prior(self, theta, y):
        '''

        Hyperprior plus flow density (with the Jacobian of y): the population prior over (theta, y)
        that the flow_prior tempering of flow_eryn leaves untempered.

        Arguments
        ------------------------------
        theta : array
        	(M, 8) contexts.
        y : array
        	(M, Nf') log10 S_gw.

        Returns
        ------------------------------
        ln_prior : jax.Array
        	(M,) log density, -inf outside the hyperprior's support.

        '''
        return _ln_flow_prior(*self._args(), jnp.atleast_2d(theta), jnp.atleast_2d(y), self.slices, self.n_quad)

    def ln_data_res(self, theta, y):
        '''

        The data and resolved-GB terms alone (the tempered likelihood of flow_eryn's flow_prior option).

        Arguments
        ------------------------------
        theta : array
        	(M, 8) contexts.
        y : array
        	(M, Nf') log10 S_gw.

        Returns
        ------------------------------
        ln_like : jax.Array
        	(M,) the two terms' sum.

        '''
        return _ln_data_res(self.consts, jnp.atleast_2d(theta), jnp.atleast_2d(y), self.width)

    def hessian(self, theta, y):
        '''

        Hessian of ln_post at one point, over (theta, y) concatenated.

        Arguments
        ------------------------------
        theta : array
        	(8,) context.
        y : array
        	(Nf',) log10 S_gw.

        Returns
        ------------------------------
        H : numpy.ndarray
        	(8 + Nf', 8 + Nf') Hessian.

        '''
        x = jnp.concatenate([jnp.asarray(theta, jnp.float64), jnp.asarray(y, jnp.float64)])
        return np.asarray(_hessian(*self._args(), x, 8, self.slices, self.n_quad, self.width))

    def conditional_hessians(self, theta, y):
        '''

        The two y-Hessians of the conditional of y at fixed theta, kept apart because the tempering
        options weight them differently: H_prior of ln_flow_prior (flow + Jacobian; the hyperprior is
        constant in y) and H_like of ln_data_res. Their sum is the y-block of hessian().

        Arguments
        ------------------------------
        theta : array
        	(8,) context.
        y : array
        	(Nf',) log10 S_gw.

        Returns
        ------------------------------
        H_prior : numpy.ndarray
        	(Nf', Nf') Hessian of the hyperprior + flow term in y.
        H_like : numpy.ndarray
        	(Nf', Nf') Hessian of the data + resolved terms in y.

        '''
        Hp, Hl = _conditional_hessians(*self._args(), jnp.asarray(theta, jnp.float64), jnp.asarray(y, jnp.float64),
                                       self.slices, self.n_quad, self.width)
        return np.asarray(Hp), np.asarray(Hl)

    def value_and_grad(self, theta, y):
        '''

        The log posterior at one point and its gradient (for the HMC samplers).

        Arguments
        ------------------------------
        theta : array
        	(8,) context.
        y : array
        	(Nf',) log10 S_gw.

        Returns
        ------------------------------
        ln_post : jax.Array
        	Scalar log posterior.
        grads : tuple of jax.Array
        	Gradients with respect to theta (8,) and y (Nf',).

        '''
        return _value_and_grad(*self._args(), jnp.asarray(theta, jnp.float64), jnp.asarray(y, jnp.float64),
                               self.slices, self.n_quad, self.width)


def quadrature_errors(post, theta, y, n_quads=(4, 8, 16, 32, 64), ref=512, n_grad=4):
    '''

    How far the log posterior and its gradient move with the number of Gauss-Legendre nodes of the
    flow's count dequantization, against a reference node count, over a set of points.

    Arguments
    ------------------------------
    post : FlowPosterior
    	The posterior (its emulator, data and width are reused).
    theta : numpy.ndarray
    	(M, 8) contexts inside the prior's support.
    y : numpy.ndarray
    	(M, Nf') log10 S_gw.
    n_quads : tuple of int
    	Node counts to test. (Default: (4, 8, 16, 32, 64))
    ref : int
    	Reference node count. (Default: 512)
    n_grad : int
    	Rows (the first ones) at which gradients are compared. (Default: 4)

    Returns
    ------------------------------
    errors : dict
    	Per node count: 'value', the largest |ln_like(n) - ln_like(ref)| over rows [nats], and 'grad',
    	the largest gradient difference relative to the reference gradient's largest component.

    '''
    make = lambda n: FlowPosterior(post.emulator, post.data, width=post.width, n_quad=n, duration=post.duration)
    theta, y = np.atleast_2d(theta), np.atleast_2d(y)
    p_ref = make(ref)
    v_ref = np.asarray(p_ref.ln_like(theta, y))
    g_ref = [np.concatenate([np.asarray(g) for g in p_ref.value_and_grad(theta[i], y[i])[1]]) for i in range(n_grad)]
    errors = {}
    for n in n_quads:
        p = make(n)
        g = [np.concatenate([np.asarray(gg) for gg in p.value_and_grad(theta[i], y[i])[1]]) for i in range(n_grad)]
        errors[int(n)] = dict(value=float(np.max(np.abs(np.asarray(p.ln_like(theta, y)) - v_ref))),
                              grad=float(max(np.max(np.abs(a - b))/np.max(np.abs(b)) for a, b in zip(g, g_ref))))
    return errors


def laplace_covariance(post, theta, y, max_std=None):
    '''

    Gaussian (Laplace) covariance of the posterior about a point, -H^-1, made positive definite and
    no wider than max_std in any coordinate: -H is regularized in coordinates scaled by max_std, its
    eigenvalues there floored at 1, so flat directions (e.g. r_bulge) get at most their own cap. Also
    returns the conditional covariances of each block given the other, for Gibbs moves.

    Arguments
    ------------------------------
    post : FlowPosterior
    	The posterior.
    theta : array
    	(8,) context (e.g. the MAP).
    y : array
    	(Nf',) log10 S_gw.
    max_std : numpy.ndarray or None
    	(8 + Nf',) widest allowed standard deviation of each coordinate; None uses each hyperprior's
    	1-99% half-range for theta and 1 dex for y. (Default: None)

    Returns
    ------------------------------
    cov : numpy.ndarray
    	(8 + Nf', 8 + Nf') joint covariance.
    cov_theta : numpy.ndarray
    	(8, 8) covariance of theta given y, the inverse of -H's theta block (regularized).
    cov_y : numpy.ndarray
    	(Nf', Nf') covariance of y given theta, the inverse of -H's y block (regularized).

    '''
    if max_std is None:
        half = np.array([np.diff(fd.HYPERPRIOR[n].ppf([0.01, 0.99]))[0]/2 for n in fd.CONTEXT_NAMES])
        max_std = np.concatenate([half, np.ones(post.data.fs.size)])
    max_std = np.asarray(max_std, dtype=np.float64)
    P = -post.hessian(theta, y)
    return (_regularized_inverse(P, max_std), _regularized_inverse(P[:8, :8], max_std[:8]),
            _regularized_inverse(P[8:, 8:], max_std[8:]))


def _regularized_inverse(P, cap):
    '''

    Covariance from a precision matrix P (e.g. -H) that need not be positive definite, no wider than
    cap in any coordinate: P is scaled by the caps, its eigenvalues there floored at 1, and inverted.

    Arguments
    ------------------------------
    P : numpy.ndarray
    	(n, n) precision (symmetrized here).
    cap : numpy.ndarray
    	(n,) widest allowed standard deviation of each coordinate.

    Returns
    ------------------------------
    cov : numpy.ndarray
    	(n, n) positive-definite covariance.

    '''
    ## in units of each coordinate's cap a covariance eigenvalue <= 1 keeps every marginal std <= its cap
    Q = cap[:, None]*((P + P.T)/2)*cap[None, :]
    w, V = np.linalg.eigh(Q)
    return cap[:, None]*((V/np.maximum(w, 1.0)) @ V.T)*cap[None, :]
