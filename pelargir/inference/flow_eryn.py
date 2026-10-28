"""
Joint parallel-tempered MCMC (Eryn) on the flow toy-model posterior (flow_inference.FlowPosterior),
with two branches: 'pop' (the eight context parameters, CONTEXT_NAMES order) and 'spectrum'
(y = log10(S_gw + C) in every modelled bin, S_gw = max(10**y - C, 0)), given a fixed resolved-GB state.

Two tempering options (Eryn tempers only the likelihood, beta ln L + ln pi):
  flow_prior : prior = the hierarchical flow-emulator prior, hyperprior + flow p(N_res, S_gw | theta)
               (with y's Jacobian), as one joint prior over both branches; likelihood = data +
               resolved-GB terms. The T = inf chain
               samples the population model's prior predictive.
  all        : likelihood = flow + data + resolved terms; prior = hyperprior on 'pop' and a uniform
               box of +-y_box dex about the data spectrum on 'spectrum'.
Both have FlowPosterior.ln_post as the T = 1 target (up to the box's constant).

Moves:
    - a joint Gaussian
    - a Gibbs Gaussian on 'pop'
    - a vector Gibbs Gaussian on 'spectrum' at fixed population parameters (the analogue of the original toy model's Poisson move; N_res is fixed by the resolved state), and hyperprior draws on 'pop'. The Gaussians use the Laplace covariance at the starting point (flow_inference.laplace_covariance), scaled by 2.38^2 / (dimensions moved). 
    - numpyro NUTS/HMC on 'spectrum' at fixed 'pop' (flow_gibbs); this makes the sampler HMC-within-Gibbs. This last move can be triggered externally via run_eryn(hmc=...).
"""
import json
import time
import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as ss

from .. import backend

jax = backend.import_jax()
import jax.numpy as jnp

## REQUIRES INPUT/VERIFICATION: Eryn still calls np.trapz (thermodynamic-integration evidence), which numpy 2.4
## removed; restore it as np.trapezoid until Eryn switches
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer
from eryn.state import State
from eryn.moves import GaussianMove, DistributionGenerate
from eryn.utils.utility import get_integrated_act
from ..sampler_utils.backends import ShiftedSpectrumBackend

from ..flows import data as fd
from . import flow_inference as fi
from ..log_utils import timed, format_seconds

logger = logging.getLogger('pelargir')

BRANCHES = ('pop', 'spectrum')
TEMPERINGS = ('flow_prior', 'all')
DEFAULT_WEIGHTS = dict(joint=0.3, pop=0.25, spectrum=0.3, prior=0.15)


def _split(coords, ndim_pop=8):
    '''(theta (N, 8), y (N, Nf')) from Eryn's vectorized branch list [pop (N, 8), spectrum (N, Nf')].'''
    return np.asarray(coords[0]), np.asarray(coords[1])


class JointFlowPrior:
    '''

    The hierarchical flow-emulator prior as Eryn's joint prior (priors={"all_models_together": ...})
    over both branches: hyperprior + flow density of (N_res, y), y = log10(S_gw + C)
    (FlowPosterior.ln_flow_prior).

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.

    Returns
    ------------------------------
    None

    '''

    ## Eryn (>= 1.2.6) reads every prior's per-branch parameter order; this joint prior spans both
    ## branches, so it has none (in-memory backends only store it; HDFBackend would need one per branch)
    key_order = None

    def __init__(self, post):
        self.post = post

    def logpdf(self, coords, inds, supps=None, branch_supps=None):
        '''

        The joint log prior of every walker.

        Arguments
        ------------------------------
        coords : dict
        	{branch: (ntemps, nwalkers, 1, ndim)} coordinates.
        inds : dict
        	{branch: (ntemps, nwalkers, 1)} leaf flags (always True here).
        supps, branch_supps :
        	Unused (Eryn passes them).

        Returns
        ------------------------------
        ln_prior : numpy.ndarray
        	(ntemps, nwalkers) log prior.

        '''
        ntemps, nwalkers = coords['pop'].shape[:2]
        theta = coords['pop'].reshape(ntemps*nwalkers, -1)
        y = coords['spectrum'].reshape(ntemps*nwalkers, -1)
        ## a writable copy: Eryn edits the prior array in place (np.asarray of a JAX array is read-only)
        return np.array(self.post.ln_flow_prior(theta, y)).reshape(ntemps, nwalkers)


class ErynFlowModel:
    '''

    Eryn likelihood and priors of a FlowPosterior for one tempering option.

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.
    tempering : str
    	'flow_prior' or 'all' (see the module docstring). (Default: 'flow_prior')
    y_box : float
    	Half-width [dex] of the uniform box on y about the data spectrum ('all' only). (Default: 3.0)

    Returns
    ------------------------------
    None

    '''

    def __init__(self, post, tempering='flow_prior', y_box=3.0):
        if tempering not in TEMPERINGS:
            raise ValueError("tempering must be one of {}, got {!r}".format(TEMPERINGS, tempering))
        self.post, self.tempering, self.y_box = post, tempering, float(y_box)
        self.y_data = post.data_spectrum()

    def log_like(self, coords):
        '''

        Vectorized log likelihood for Eryn.

        Arguments
        ------------------------------
        coords : list of numpy.ndarray
        	[pop (N, 8), spectrum (N, Nf')].

        Returns
        ------------------------------
        ln_like : numpy.ndarray
        	(N,) data + resolved terms ('flow_prior'), or with the flow too ('all').

        '''
        theta, y = _split(coords)
        n = theta.shape[0]
        ## Eryn passes only the walkers inside the prior, so the batch size varies; pad it to a power of 2 so
        ## the jitted flow compiles for a handful of shapes instead of once per size
        m = 1 << max(n - 1, 0).bit_length()
        if m > n:
            theta = np.concatenate([theta, np.repeat(theta[:1], m - n, axis=0)])
            y = np.concatenate([y, np.repeat(y[:1], m - n, axis=0)])
        ## writable copies (Eryn may edit them in place)
        if self.tempering == 'flow_prior':
            return np.array(self.post.ln_data_res(theta, y))[:n]
        return np.array(self.post.ln_like(theta, y))[:n]

    def hyperprior_container(self):
        '''

        The hyperprior on 'pop' as an Eryn ProbDistContainer (scipy distributions, CONTEXT_NAMES order).

        Arguments
        ------------------------------
        None

        Returns
        ------------------------------
        container : eryn.prior.ProbDistContainer
        	logpdf and rvs over (N, 8).

        '''
        return ProbDistContainer({j: fd.HYPERPRIOR[name] for j, name in enumerate(fd.CONTEXT_NAMES)})

    def box_container(self):
        '''The uniform box on 'spectrum' ('all' tempering) as a ProbDistContainer.'''
        return ProbDistContainer({j: ss.uniform(loc=v - self.y_box, scale=2*self.y_box)
                                  for j, v in enumerate(self.y_data)})

    def box_log_density(self):
        '''The box prior's constant log density (it adds to ln_post at T = 1 inside the box).'''
        return -self.y_data.size*np.log(2*self.y_box)

    @property
    def priors(self):
        '''The priors argument of EnsembleSampler.'''
        if self.tempering == 'flow_prior':
            return {"all_models_together": JointFlowPrior(self.post)}
        return {'pop': self.hyperprior_container(), 'spectrum': self.box_container()}

    def log_post(self, theta, y):
        '''

        ln prior + ln likelihood at beta = 1, computed as the sampler computes them (for tests).

        Arguments
        ------------------------------
        theta : numpy.ndarray
        	(N, 8) contexts.
        y : numpy.ndarray
        	(N, Nf') log10(S_gw + C).

        Returns
        ------------------------------
        ln_post : numpy.ndarray
        	(N,) the sum.

        '''
        if self.tempering == 'flow_prior':
            coords = {'pop': theta[None, :, None], 'spectrum': y[None, :, None]}
            lp = JointFlowPrior(self.post).logpdf(coords, None)[0]
        else:
            lp = self.hyperprior_container().logpdf(theta) + self.box_container().logpdf(y)
        return lp + self.log_like([theta, y])


def make_moves(model, cov, cov_pop, cov_spec, weights=None, hmc_move=None):
    '''

    The weighted move list: joint Gaussian, Gibbs Gaussian on 'pop', vector Gibbs Gaussian on
    'spectrum' (the Poisson-move analogue), hyperprior draws on 'pop', and optionally the HMC move on
    'spectrum' (flow_gibbs.HMCSpectrumMove).

    Arguments
    ------------------------------
    model : ErynFlowModel
    	For the hyperprior container.
    cov : numpy.ndarray
    	(8 + Nf', 8 + Nf') joint covariance (flow_inference.laplace_covariance).
    cov_pop : numpy.ndarray
    	(8, 8) conditional covariance of 'pop'.
    cov_spec : numpy.ndarray
    	(Nf', Nf') conditional covariance of 'spectrum'.
    weights : dict or None
    	Relative weights of 'joint', 'pop', 'spectrum', 'prior' and optionally 'hmc' (zero leaves a
    	move out). (Default: None, DEFAULT_WEIGHTS)
    hmc_move : flow_gibbs.HMCSpectrumMove or None
    	The move behind an 'hmc' weight. (Default: None)

    Returns
    ------------------------------
    moves : list of tuple
    	[(move, weight), ...] for EnsembleSampler, weights normalized.
    names : list of str
    	The kept moves' names, in the same order.

    '''
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    d = cov.shape[0]
    ## the joint move proposes both branches with the pop-pop and spectrum-spectrum blocks of the joint
    ## covariance (Eryn's GaussianMove draws each branch independently)
    joint = GaussianMove({'pop': cov[:8, :8]*2.38**2/d, 'spectrum': cov[8:, 8:]*2.38**2/d}, mode='vector')
    all_moves = dict(joint=joint,
                     pop=GaussianMove({'pop': cov_pop*2.38**2/8, 'spectrum': cov_spec}, mode='vector',
                                      gibbs_sampling_setup='pop'),
                     spectrum=GaussianMove({'pop': cov_pop, 'spectrum': cov_spec*2.38**2/cov_spec.shape[0]},
                                           mode='vector', gibbs_sampling_setup='spectrum'),
                     prior=DistributionGenerate({'pop': model.hyperprior_container()}, gibbs_sampling_setup='pop'))
    if w.get('hmc', 0) > 0:
        if hmc_move is None:
            raise ValueError("an 'hmc' weight needs hmc_move")
        all_moves['hmc'] = hmc_move
    chosen = [(all_moves[k], float(v)) for k, v in w.items() if v > 0]
    total = sum(v for _, v in chosen)
    return [(m, v/total) for m, v in chosen], [k for k, v in w.items() if v > 0]


def initial_walkers(post, cov, start, ntemps, nwalkers, strategy='map', seed=0, max_tries=100):
    '''

    Starting coordinates for every walker: draws from N(start, cov) inside the hyperprior's support
    ('map', 'truth'), or hyperprior draws with the data spectrum ('prior').

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.
    cov : numpy.ndarray
    	(8 + Nf', 8 + Nf') covariance of the scatter.
    start : tuple of numpy.ndarray
    	(theta (8,), y (Nf',)) centre for 'map' and 'truth'.
    ntemps, nwalkers : int
    	Ladder size.
    strategy : str
    	'map', 'prior' or 'truth' (only 'prior' differs: hyperprior draws for theta, data spectrum
    	plus 0.01 dex scatter for y). (Default: 'map')
    seed : int
    	Seed. (Default: 0)
    max_tries : int
    	Redraw rounds for walkers outside the support. (Default: 100)

    Returns
    ------------------------------
    coords : dict
    	{'pop': (ntemps, nwalkers, 1, 8), 'spectrum': (ntemps, nwalkers, 1, Nf')}.

    '''
    rng = np.random.default_rng(seed)
    n = ntemps*nwalkers
    nf = post.data.fs.size
    if strategy == 'prior':
        theta = fd.sample_context(rng, n)
        y = post.data_spectrum() + 0.01*rng.standard_normal((n, nf))
    else:
        x0 = np.concatenate([np.asarray(start[0]), np.asarray(start[1])])
        x = rng.multivariate_normal(x0, cov, size=n)
        for _ in range(max_tries):
            bad = ~np.isfinite(np.asarray(fi.log_hyperprior(jnp.asarray(x[:, :8]))))
            if not bad.any():
                break
            x[bad] = rng.multivariate_normal(x0, cov, size=int(bad.sum()))
        else:
            raise RuntimeError("could not draw every walker inside the prior's support")
        theta, y = x[:, :8], x[:, 8:]
    return {'pop': theta.reshape(ntemps, nwalkers, 1, 8), 'spectrum': y.reshape(ntemps, nwalkers, 1, nf)}


@dataclass
class ErynResult:
    '''

    Chains and diagnostics of run_eryn.

    Arguments
    ------------------------------
    theta : numpy.ndarray
    	(steps, walkers, 8) T = 1 context samples.
    y : numpy.ndarray
    	(steps, walkers, Nf') T = 1 samples of y = log10(S_gw + C).
    log_like : numpy.ndarray
    	(steps, ntemps, walkers) log likelihood.
    log_prior : numpy.ndarray
    	(steps, ntemps, walkers) log prior.
    betas : numpy.ndarray
    	(steps, ntemps) inverse temperatures.
    meta : dict
    	Settings, move and swap acceptance, autocorrelation times, timings. (Default: {})
    S : numpy.ndarray or None
    	(steps, walkers, Nf') T = 1 chain of S_gw = max(10**y - C, 0), the spectrum branch's supplemental
    	(ShiftedSpectrumBackend). (Default: None)

    Returns
    ------------------------------
    None

    '''
    theta: np.ndarray
    y: np.ndarray
    log_like: np.ndarray
    log_prior: np.ndarray
    betas: np.ndarray
    meta: dict = field(default_factory=dict)
    S: np.ndarray = None

    def save(self, path):
        np.savez(path, theta=self.theta, y=self.y, log_like=self.log_like, log_prior=self.log_prior, betas=self.betas,
                 meta=json.dumps(self.meta), **({} if self.S is None else {'S': self.S}))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ## files from before Stage 3 have no S: their y is log10 S_gw (C = 0)
        S = d['S'] if 'S' in d.files else 10**d['y']
        return cls(d['theta'], d['y'], d['log_like'], d['log_prior'], d['betas'], json.loads(str(d['meta'])), S)


def start_point(post, strategy, seed=0, progress=False, **map_kwargs):
    '''

    Centre of the walkers and of the Laplace covariance: the MAP (flow_hmc's optimizer), the truth, or
    for 'prior' the hyperprior median with the data spectrum.

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.
    strategy : str
    	'map', 'prior' or 'truth'.
    seed : int
    	Seed of the MAP optimizer. (Default: 0)
    progress : bool
    	Show the MAP's progress bar and log its result. (Default: False)
    **map_kwargs
    	map_steps and map_lr for flow_hmc.initial_values.

    Returns
    ------------------------------
    theta : numpy.ndarray
    	(8,) context.
    y : numpy.ndarray
    	(Nf',) log10(S_gw + C).

    '''
    if strategy == 'truth':
        return tuple(np.asarray(a)[0] for a in post.truth)
    if strategy == 'prior':
        return np.array([fd.HYPERPRIOR[n].median() for n in fd.CONTEXT_NAMES]), post.data_spectrum()
    from . import flow_hmc as fh
    _, info = fh.initial_values(post, fh.make_model(post), 'map', 1, seed, jitter=0.0, progress=progress, **map_kwargs)
    return np.asarray(info['theta'][0]), np.asarray(info['y'][0])


def run_eryn(post, nsteps, ntemps=8, nwalkers=16, tempering='flow_prior', init='map', Tmax=np.inf, adaptive=True,
             move_weights=None, seed=0, thin=1, burn=0, y_box=3.0, progress=True, hmc=None, **map_kwargs):
    '''

    Parallel-tempered MCMC on a FlowPosterior with Eryn: joint (2c), or HMC-within-Gibbs (2d) with
    hmc set, where numpyro moves 'spectrum' at fixed 'pop'. Burn-in runs first (unstored, the HMC
    move adapting), then the HMC move is frozen and the stored steps run.

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.
    nsteps : int
    	Kept steps (after burn).
    ntemps : int
    	Temperatures. (Default: 8)
    nwalkers : int
    	Walkers per temperature. (Default: 16)
    tempering : str
    	'flow_prior' or 'all'. (Default: 'flow_prior')
    init : str
    	'map', 'prior' or 'truth' (see initial_walkers and start_point). (Default: 'map')
    Tmax : float
    	Hottest temperature. (Default: inf)
    adaptive : bool
    	Adapt the temperature ladder. (Default: True)
    move_weights : dict or None
    	Weights of the moves (see make_moves). (Default: None, DEFAULT_WEIGHTS, or
    	flow_gibbs.GIBBS_WEIGHTS with hmc)
    seed : int
    	Seed of the start, the walkers and Eryn (numpy's global state). (Default: 0)
    thin : int
    	Thinning of the stored chain. (Default: 1)
    burn : int
    	Discarded steps before storing. (Default: 0)
    y_box : float
    	Box half-width [dex] for tempering 'all'. (Default: 3.0)
    progress : bool
    	Eryn's progress bars. Phase messages are logged at INFO and the MAP has its own bar, both shown when
    	pelargir logs at INFO (a script's default). (Default: True)
    hmc : dict or None
    	None for the joint sampler; otherwise flow_gibbs.HMCSpectrumMove settings (kernel, adapt,
    	max_tree_depth, trajectory_length, target_accept, n_transitions; {} for the defaults).
    	(Default: None)
    **map_kwargs
    	map_steps, map_lr for the MAP start.

    Returns
    ------------------------------
    result : ErynResult
    	T = 1 chains, log likelihoods and priors, the beta history and diagnostics.

    '''
    model = ErynFlowModel(post, tempering, y_box)
    logger.info("Eryn: %d temperatures x %d walkers, tempering %r, %d burn-in + %d steps (thin %d)%s", ntemps,
               nwalkers, tempering, burn, nsteps, thin, ", HMC-within-Gibbs" if hmc is not None else "")
    t0 = time.time()
    with timed(logger, f"start point ({init})"):
        start = start_point(post, init, seed, progress=True, **map_kwargs)
    with timed(logger, "Laplace covariance at the start point"):
        cov, cov_pop, cov_spec = fi.laplace_covariance(post, *start)
    coords = initial_walkers(post, cov, start, ntemps, nwalkers, init, seed)
    hmc_move = None
    if hmc is not None:
        from . import flow_gibbs as fg
        weights = dict(fg.GIBBS_WEIGHTS if move_weights is None else move_weights)
        ## adaptation spans the HMC calls expected during burn-in
        share = weights.get('hmc', 0)/sum(weights.values())
        with timed(logger, "HMC spectrum move (conditional Hessians, kernels)"):
            hmc_move = fg.HMCSpectrumMove(post, *post.conditional_hessians(*start), tempering=tempering, y_box=y_box,
                                          num_warmup=int(round(share*burn)), seed=seed, **hmc)
    else:
        weights = dict(DEFAULT_WEIGHTS if move_weights is None else move_weights)
    t_init = time.time() - t0
    moves, move_names = make_moves(model, cov, cov_pop, cov_spec, weights, hmc_move)
    ## Eryn draws from numpy's global state (its moves and the scipy prior draws)
    np.random.seed(seed)
    sampler = EnsembleSampler(nwalkers, {'pop': 8, 'spectrum': post.data.fs.size}, model.log_like, model.priors,
                              branch_names=list(BRANCHES), nleaves_max={b: 1 for b in BRANCHES},
                              tempering_kwargs=dict(ntemps=ntemps, Tmax=Tmax, adaptive=adaptive),
                              moves=moves, vectorize=True, backend=ShiftedSpectrumBackend(post.floor))
    logger.info("moves %s; initialized in %s", ", ".join(f"{k} {v:g}" for k, v in weights.items() if v),
               format_seconds(t_init))
    t0 = time.time()
    if burn:
        logger.info("burn-in: %d steps", burn)
    state = sampler.run_mcmc(State(coords), 0, burn=burn, progress=progress) if burn else State(coords)
    t_burn = time.time() - t0
    if burn:
        logger.info("burn-in: done in %s", format_seconds(t_burn))
    if hmc_move is not None:
        hmc_move.freeze()
        d = hmc_move.burn_diagnostics
        if d and d.get('calls'):
            logger.info("HMC frozen after %d calls: accept %s, leapfrog steps %s, step size %s", d['calls'],
                       np.round(d['accept_prob'], 2).tolist(), np.round(d['num_steps'], 1).tolist(),
                       np.round(d['step_size_median'], 3).tolist())
    logger.info("sampling: %d steps", nsteps)
    t0 = time.time()
    sampler.run_mcmc(state, nsteps, thin_by=thin, progress=progress)
    t_run = time.time() - t0
    logger.info("sampling: done in %s; T=1 acceptance %.3f, swap acceptance %s", format_seconds(t_run),
               float(np.mean(sampler.acceptance_fraction[0])),
               np.round(sampler.swap_acceptance_fraction, 2).tolist() if hasattr(sampler, 'swap_acceptance_fraction') else None)
    chain = sampler.get_chain()
    theta, y = chain['pop'][:, 0, :, 0], chain['spectrum'][:, 0, :, 0]
    ## the S_gw branch supplemental, computed from the stored y at save time
    S_all = sampler.backend.get_chain_supplemental()['spectrum']['S']
    S = S_all[:, 0, :, 0]
    move_info = sampler.backend.get_move_info() or {}
    acc = {name: float(np.mean(info['acceptance_fraction'][0])) for name, info in zip(move_names, move_info.values())}
    ## integrated autocorrelation time of the T = 1 chain (Eryn's backend method refuses ntemps > 1)
    ## (bins where S_gw is always 0 have no variance: their S time is NaN)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = get_integrated_act({**{k: v[:, :1] for k, v in chain.items()}, 'S': S_all[:, :1]},
                                 window=max(2, min(50, nsteps//2)))
    meta = dict(settings=dict(nsteps=nsteps, ntemps=ntemps, nwalkers=nwalkers, tempering=tempering, init=init,
                              Tmax=float(Tmax), adaptive=adaptive, move_weights=weights, seed=seed, thin=thin,
                              burn=burn, y_box=y_box, n_quad=post.n_quad, width=post.width, floor=post.floor,
                              hmc=None if hmc_move is None else hmc_move.settings),
                start=dict(theta=start[0].tolist(), y=start[1].tolist()),
                move_acceptance_T1=acc, move_acceptance=[{k: np.asarray(v['acceptance_fraction']).mean(axis=1).tolist()}
                                                         for k, v in zip(move_names, move_info.values())],
                swap_acceptance=(sampler.swap_acceptance_fraction.tolist()
                                 if hasattr(sampler, 'swap_acceptance_fraction') else None),
                autocorr_time={k: np.asarray(v).reshape(-1).tolist() for k, v in tau.items()},
                seconds=dict(init=t_init, burn=t_burn, run=t_run), context_names=fd.CONTEXT_NAMES)
    if hmc_move is not None:
        meta['hmc'] = dict(burn=hmc_move.burn_diagnostics, sampling=hmc_move.diagnostics())
    return ErynResult(theta, y, sampler.get_log_like(), sampler.get_log_prior(), sampler.get_betas(), meta, S)
