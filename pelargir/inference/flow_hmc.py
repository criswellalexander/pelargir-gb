"""
Joint HMC (numpyro NUTS) on the flow toy-model posterior (flow_posterior.FlowPosterior): the eight
context parameters theta and y = log10 S_gw in every modelled bin, given a fixed resolved-GB state.

The numpyro model has one sample site per context parameter, with the hyperprior of
flows.data.HYPERPRIOR, an improper flat site for y (the flow term carries y's density and Jacobian),
and the rest of the posterior as a factor, so its log joint is FlowPosterior.ln_post. Chains start at
the MAP (default), at hyperprior draws, or at the truth, each with a small per-chain jitter in
unconstrained space.

The flow's parameters (~200 MB for the production flow) are passed to the model as an argument
(model_args, with jit_model_args), never closed over: a compiled function that closes over them embeds
them as constants, and NUTS's tree-building loop would embed them once per call site and run out of
GPU memory.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np
import optax

from .. import backend

jax = backend.import_jax()
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import unconstrain_fn, initialize_model
from numpyro.diagnostics import summary as numpyro_summary

from ..flows import data as fd
from . import flow_posterior as fi

EXTRA_FIELDS = ('diverging', 'num_steps', 'accept_prob', 'adapt_state.step_size')
INIT_STRATEGIES = ('map', 'prior', 'truth')


def prior_distributions():
    '''

    numpyro distributions matching flows.data.HYPERPRIOR, in CONTEXT_NAMES order.

    Arguments
    ------------------------------
    None

    Returns
    ------------------------------
    priors : dict
    	{name: numpyro distribution}.

    '''
    priors = {}
    for name, p in zip(fd.CONTEXT_NAMES, fi._PRIOR):
        if p[0] == 'uniform':
            priors[name] = dist.Uniform(p[1], p[2])
        elif p[0] == 'invgamma':
            ## numpyro's InverseGamma rate plays scipy's scale (1 here)
            priors[name] = dist.InverseGamma(p[1], 1.0)
        else:
            lo, hi, loc, scale = p[1:]
            priors[name] = dist.TruncatedNormal(loc, scale, low=lo, high=None if np.isinf(hi) else hi)
    return priors


def model_args(post):
    '''

    The model's argument: the flow's per-band models and transforms and the data constants, as one
    pytree (pass it as the model's argument, never close over it).

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.

    Returns
    ------------------------------
    args : tuple
    	(models, transforms, ctx_transforms, consts).

    '''
    return post._args()


def make_model(post):
    '''

    The numpyro model of a FlowPosterior, called as model(model_args(post)): its log joint (in
    constrained coordinates) is post.ln_post. Only the static settings (bands, n_quad, width) are
    taken from post.

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.

    Returns
    ------------------------------
    model : callable
    	numpyro model with sites CONTEXT_NAMES and 'y', and the factor 'like'.

    '''
    priors = prior_distributions()
    nf = post.data.fs.size
    slices, n_quad, width = post.slices, post.n_quad, post.width

    def model(args):
        theta = jnp.stack([numpyro.sample(name, priors[name]) for name in fd.CONTEXT_NAMES])
        y = numpyro.sample('y', dist.ImproperUniform(dist.constraints.real, (), (nf,)))
        numpyro.factor('like', fi._ln_like(*args, theta[None], y[None], slices, n_quad, width)[0])
    return model


def _as_sites(theta, y):
    '''{site: value} from a context theta (8,) and y (Nf',).'''
    return {**{name: jnp.asarray(theta[j]) for j, name in enumerate(fd.CONTEXT_NAMES)}, 'y': jnp.asarray(y)}


def _from_sites(sites):
    '''(theta (..., 8), y (..., Nf')) from a dict of site values with any leading axes.'''
    return np.stack([np.asarray(sites[name]) for name in fd.CONTEXT_NAMES], axis=-1), np.asarray(sites['y'])


def data_spectrum(post):
    '''

    The data's foreground estimate, log10(d - S_n), floored at 1e-3 d where d - S_n is not positive.

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.

    Returns
    ------------------------------
    y : numpy.ndarray
    	(Nf',) log10 S_gw.

    '''
    d, Sn = post.data.data_psd, post.data.noise_psd
    return np.log10(np.where(d - Sn > 0, d - Sn, 1e-3*d))


def initial_values(post, model, strategy='map', num_chains=4, seed=0, jitter=0.01, map_steps=2000, map_lr=1e-2):
    '''

    Per-chain initial values for NUTS in unconstrained space. The MAP is found by Adam on numpyro's
    potential energy in unconstrained space (the space NUTS samples), with the flow passed as an argument.

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.
    model : callable
    	make_model(post).
    strategy : str
    	'map' (optimized from the prior median and the data spectrum), 'prior' (hyperprior
    	draws per chain, data spectrum) or 'truth'. (Default: 'map')
    num_chains : int
    	Chains. (Default: 4)
    seed : int
    	Seed of the prior draws, the MAP optimizer and the jitter. (Default: 0)
    jitter : float
    	Standard deviation of the Gaussian jitter added per chain in unconstrained space. (Default: 0.01)
    map_steps : int
    	Optimizer steps for 'map'. (Default: 2000)
    map_lr : float
    	Adam learning rate for 'map'. (Default: 1e-2)

    Returns
    ------------------------------
    init : dict
    	{site: (num_chains, ...) unconstrained values}.
    info : dict
    	The constrained start point(s) 'theta' and 'y' (per chain, before jitter), and for 'map' the
    	start point and the MAP loss history.

    '''
    if strategy not in INIT_STRATEGIES:
        raise ValueError("init strategy must be one of {}, got {!r}".format(INIT_STRATEGIES, strategy))
    rng = np.random.default_rng(seed)
    args = model_args(post)
    unconstrain = lambda th, y: unconstrain_fn(model, (args,), {}, _as_sites(th, y))
    y_data = data_spectrum(post)
    info = {}
    if strategy == 'truth':
        th0, y0 = (np.asarray(a)[0] for a in post.truth)
        theta, y = np.repeat(th0[None], num_chains, 0), np.repeat(y0[None], num_chains, 0)
    elif strategy == 'prior':
        theta, y = fd.sample_context(rng, num_chains), np.repeat(y_data[None], num_chains, 0)
    else:
        th_start = np.array([fd.HYPERPRIOR[name].median() for name in fd.CONTEXT_NAMES])
        u_map, losses = _map_unconstrained(model, args, unconstrain(th_start, y_data), map_steps, map_lr, seed)
        _, _, postprocess, _ = initialize_model(jax.random.key(seed), model, model_args=(args,), dynamic_args=True)
        th_map, y_map = _from_sites(postprocess(args)(u_map))
        theta, y = np.repeat(th_map[None], num_chains, 0), np.repeat(y_map[None], num_chains, 0)
        info.update(start_theta=th_start.tolist(), start_y=y_data.tolist(), map_loss=losses.tolist())
    info.update(theta=theta.tolist(), y=y.tolist())
    init = {}
    for c in range(num_chains):
        u = unconstrain(theta[c], y[c])
        for name, v in u.items():
            init.setdefault(name, []).append(np.asarray(v))
    init = {name: np.stack(v) for name, v in init.items()}
    for name in init:
        init[name] = init[name] + jitter*rng.standard_normal(init[name].shape)
    return {name: jnp.asarray(v) for name, v in init.items()}, info


def _map_unconstrained(model, args, u0, steps, lr, seed):
    '''

    Minimize numpyro's potential energy (negative log joint in unconstrained space) with Adam.

    Arguments
    ------------------------------
    model : callable
    	make_model's model.
    args : tuple
    	model_args(post), passed to the jitted step as an argument.
    u0 : dict
    	Unconstrained start point.
    steps : int
    	Adam steps.
    lr : float
    	Learning rate.
    seed : int
    	Seed for numpyro's model initialization.

    Returns
    ------------------------------
    u : dict
    	Unconstrained optimum.
    losses : numpy.ndarray
    	(steps,) potential energy per step.

    '''
    _, potential_gen, _, _ = initialize_model(jax.random.key(seed), model, model_args=(args,), dynamic_args=True)
    opt = optax.adam(lr)

    @jax.jit
    def run(u, args):
        def step(carry, _):
            u, state = carry
            loss, g = jax.value_and_grad(lambda v: potential_gen(args)(v))(u)
            updates, state = opt.update(g, state, u)
            return (optax.apply_updates(u, updates), state), loss
        (u, _), losses = jax.lax.scan(step, (u, opt.init(u)), None, length=steps)
        return u, losses

    u, losses = run({k: jnp.asarray(v) for k, v in u0.items()}, args)
    return u, np.asarray(losses)


@dataclass
class HMCResult:
    '''

    Samples and diagnostics of a run_nuts call.

    Arguments
    ------------------------------
    theta : numpy.ndarray
    	(chains, draws, 8) context samples, CONTEXT_NAMES order.
    y : numpy.ndarray
    	(chains, draws, Nf') log10 S_gw samples.
    extra : dict
    	{field: (chains, draws) array} of EXTRA_FIELDS.
    meta : dict
    	Settings, initialization info, summary (per-site mean, std, n_eff, r_hat) and timings. (Default: {})

    Returns
    ------------------------------
    None

    '''
    theta: np.ndarray
    y: np.ndarray
    extra: dict
    meta: dict = field(default_factory=dict)

    def save(self, path):
        np.savez(path, theta=self.theta, y=self.y, meta=json.dumps(self.meta),
                 **{'extra/' + k: v for k, v in self.extra.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        extra = {k[len('extra/'):]: d[k] for k in d.files if k.startswith('extra/')}
        return cls(d['theta'], d['y'], extra, json.loads(str(d['meta'])))


def _json_summary(samples):
    '''numpyro's summary of grouped samples as JSON-ready lists.'''
    return {site: {stat: np.asarray(v).tolist() for stat, v in stats.items()}
            for site, stats in numpyro_summary(samples).items()}


def run_nuts(post, num_warmup=500, num_samples=500, num_chains=4, init='map', seed=0, dense_mass=True,
             target_accept_prob=0.8, max_tree_depth=10, chain_method='vectorized', progress=True, **init_kwargs):
    '''

    Joint NUTS on a FlowPosterior.

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.
    num_warmup : int
    	Warmup (adaptation) iterations per chain. (Default: 500)
    num_samples : int
    	Kept samples per chain. (Default: 500)
    num_chains : int
    	Chains. (Default: 4)
    init : str
    	Initialization strategy, see initial_values. (Default: 'map')
    seed : int
    	Seed of the initialization and the sampler. (Default: 0)
    dense_mass : bool
    	Adapt a dense mass matrix (else diagonal). (Default: True)
    target_accept_prob : float
    	NUTS target acceptance probability. (Default: 0.8)
    max_tree_depth : int
    	NUTS maximum tree depth. (Default: 10)
    chain_method : str
    	numpyro chain method; 'vectorized' runs the chains as one batch on one device. (Default: 'vectorized')
    progress : bool
    	Show numpyro's progress bar. (Default: True)
    **init_kwargs
    	Passed to initial_values (jitter, map_steps, map_lr).

    Returns
    ------------------------------
    result : HMCResult
    	Samples, extra fields and diagnostics.

    '''
    model = make_model(post)
    t0 = time.time()
    init_params, info = initial_values(post, model, init, num_chains, seed, **init_kwargs)
    t_init = time.time() - t0
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method=chain_method, progress_bar=progress, jit_model_args=True)
    t0 = time.time()
    mcmc.run(jax.random.key(seed + 1), model_args(post), init_params=init_params, extra_fields=EXTRA_FIELDS)
    samples = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(samples)
    t_run = time.time() - t0
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    theta, y = _from_sites(samples)
    n_grad = int(extra['num_steps'].sum())
    meta = dict(settings=dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, init=init,
                              seed=seed, dense_mass=dense_mass, target_accept_prob=target_accept_prob,
                              max_tree_depth=max_tree_depth, chain_method=chain_method, n_quad=post.n_quad,
                              width=post.width, **init_kwargs),
                init=info, summary=_json_summary({k: np.asarray(v) for k, v in samples.items()}),
                seconds=dict(init=t_init, run=t_run), n_divergent=int(extra['diverging'].sum()),
                gradients_sampling=n_grad, context_names=fd.CONTEXT_NAMES)
    return HMCResult(theta, y, extra, meta)


def time_warmup(post, num_warmup, num_chains=4, init='truth', seed=0, dense_mass=True, target_accept_prob=0.8,
                max_tree_depth=10, chain_method='vectorized', **init_kwargs):
    '''

    Wall time and gradient count of a warmup-only NUTS run (for projecting the cost of a full run).
    The time includes compiling the sampler, so callers difference two run lengths.

    Arguments
    ------------------------------
    post : flow_posterior.FlowPosterior
    	The posterior.
    num_warmup : int
    	Warmup iterations per chain.
    num_chains : int
    	Chains. (Default: 4)
    init : str
    	Initialization strategy, see initial_values. (Default: 'truth')
    seed : int
    	Seed. (Default: 0)
    dense_mass, target_accept_prob, max_tree_depth, chain_method :
    	As in run_nuts.
    **init_kwargs
    	Passed to initial_values.

    Returns
    ------------------------------
    seconds : float
    	Wall time of the warmup (excluding initialization).
    num_steps : numpy.ndarray
    	(num_chains, num_warmup) leapfrog steps (gradients) per iteration.

    '''
    model = make_model(post)
    init_params, _ = initial_values(post, model, init, num_chains, seed, **init_kwargs)
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=1, num_chains=num_chains, chain_method=chain_method,
                progress_bar=False, jit_model_args=True)
    t0 = time.time()
    mcmc.warmup(jax.random.key(seed + 1), model_args(post), init_params=init_params, extra_fields=('num_steps',),
                collect_warmup=True)
    steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)['num_steps'])
    return time.time() - t0, steps
