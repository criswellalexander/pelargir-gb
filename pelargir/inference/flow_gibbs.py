"""
HMC-within-Gibbs on the flow toy-model posterior (stage 2d): an Eryn move that runs numpyro NUTS or
HMC on the 'spectrum' branch (y = log10(S_gw + C)) at fixed population parameters, alongside flow_eryn's
moves on 'pop' (flow_eryn.run_eryn with hmc=...).

At every temperature the move targets the chain's tempered conditional of y given theta,

  flow_prior : ln_flow_prior(theta, y) + beta ln_data_res(theta, y)
  all        : beta [ln_flow_prior(theta, y) + ln_data_res(theta, y)] inside the y box (-inf outside)

(up to theta-only constants, Eryn's beta ln L + ln pi), so each transition leaves it invariant and the
parallel tempering stays valid. One numpyro chain runs per (temperature, walker) slot, all vmapped.

Mass matrix ('laplace', the default): the inverse of the tempered -H_yy, built for every beta from the
two y-Hessians at the start point (FlowPosterior.conditional_hessians), H_prior of ln_flow_prior (the
hierarchical flow-emulator prior) and H_like of ln_data_res, since the two tempering options weight them differently:
  flow_prior : -(H_prior + beta H_like)        all : -beta (H_prior + H_like)
regularized as flow_inference.laplace_covariance (no direction wider than 1 dex) and rebuilt whenever
the adaptive ladder moves; only the step size is adapted (numpyro's dual averaging) during burn-in.
'numpyro' instead runs numpyro's windowed warmup (step size and a dense mass matrix, starting from the
Laplace matrix). Either way freeze() ends adaptation after burn-in, after which the move is a fixed
kernel for each temperature.
"""
import logging

import numpy as np

from .. import backend

jax = backend.import_jax()
import jax.numpy as jnp
from numpyro.infer.hmc import hmc, momentum_generator
from numpyro.infer.hmc_util import (IntegratorState, _initialize_mass_matrix, euclidean_kinetic_energy,
                                    find_reasonable_step_size)

## flow_eryn first: it restores the numpy functions Eryn's fork still calls
from . import flow_eryn as fe
from eryn.moves import Move
from eryn.state import State

from . import flow_inference as fi

logger = logging.getLogger('pelargir')

KERNELS = ('nuts', 'hmc')
ADAPTS = ('laplace', 'numpyro')
## stage 2d's default move weights (Alexander, 2026-10-27); the S_gw Gaussian is left out
GIBBS_WEIGHTS = dict(joint=0.1, pop=0.4, spectrum=0.0, prior=0.1, hmc=0.4)
## first guess of the step size in mass-whitened coordinates (a near-Gaussian target in 45 dimensions
## takes ~45^-1/4 = 0.4), refined per chain by numpyro's step-size search at the first call
INITIAL_STEP = 0.5


def _ln_conditional(models, transforms, ctx_transforms, c, theta, y, beta, box, tempering, slices, n_quad, width):
    '''

    One chain's tempered log conditional of y at fixed theta (theta-only terms included).

    Arguments
    ------------------------------
    models, transforms, ctx_transforms, c :
    	FlowPosterior._args() (passed as arguments, never closed over).
    theta : jax.Array
    	(8,) context.
    y : jax.Array
    	(Nf',) log10(S_gw + C).
    beta : jax.Array
    	Inverse temperature.
    box : tuple of jax.Array
    	(Nf',) lower and upper edges of the y box ('all' only).
    tempering : str
    	'flow_prior' or 'all'.
    slices, n_quad, width :
    	FlowPosterior's static settings.

    Returns
    ------------------------------
    ln_cond : jax.Array
    	Scalar log density.

    '''
    th, yy = theta[None], y[None]
    if tempering == 'flow_prior':
        return (fi._ln_flow_prior(models, transforms, ctx_transforms, c, th, yy, slices, n_quad)
                + beta*fi._ln_data_res(c, th, yy, width))[0]
    inside = jnp.all((y >= box[0]) & (y <= box[1]))
    ln_like = fi._ln_like(models, transforms, ctx_transforms, c, th, yy, slices, n_quad, width)[0]
    return jnp.where(inside, beta*ln_like, -jnp.inf)


def laplace_inverse_mass(H_prior, H_like, betas, tempering, cap=1.0):
    '''

    Each temperature's inverse mass matrix: the regularized inverse of the tempered -H_yy.

    Arguments
    ------------------------------
    H_prior, H_like : numpy.ndarray
    	(Nf', Nf') y-Hessians of ln_flow_prior and ln_data_res (FlowPosterior.conditional_hessians).
    betas : numpy.ndarray
    	(ntemps,) inverse temperatures.
    tempering : str
    	'flow_prior' (-(H_prior + beta H_like)) or 'all' (-beta (H_prior + H_like)).
    cap : float
    	Widest allowed standard deviation [dex] in any bin; it sets the matrix where the curvature
    	vanishes ('all' at beta = 0). (Default: 1.0)

    Returns
    ------------------------------
    inv_mass : numpy.ndarray
    	(ntemps, Nf', Nf') positive-definite inverse mass matrices.

    '''
    caps = np.full(H_prior.shape[0], float(cap))
    out = []
    for b in np.atleast_1d(betas):
        P = -(H_prior + b*H_like) if tempering == 'flow_prior' else -b*(H_prior + H_like)
        out.append(fi._regularized_inverse(P, caps))
    return np.array(out)


class HMCSpectrumMove(Move):
    '''

    Eryn move: numpyro NUTS or HMC on the 'spectrum' branch at fixed 'pop', one chain per
    (temperature, walker) slot, targeting each slot's tempered conditional (see the module docstring).

    Arguments
    ------------------------------
    post : flow_inference.FlowPosterior
    	The posterior.
    H_prior, H_like : numpy.ndarray
    	(Nf', Nf') y-Hessians at the start point (FlowPosterior.conditional_hessians).
    tempering : str
    	'flow_prior' or 'all', as the sampler's ErynFlowModel. (Default: 'flow_prior')
    y_box : float
    	Half-width [dex] of the y box about the data spectrum ('all' only). (Default: 3.0)
    kernel : str
    	'nuts' or 'hmc' (fixed trajectory length). (Default: 'nuts')
    adapt : str
    	'laplace' (Laplace mass per temperature, step size adapted) or 'numpyro' (step size and dense
    	mass adapted). (Default: 'laplace')
    num_warmup : int
    	HMC calls per chain with adaptation on (0: none; the step size stays at its first-call search). (Default: 0)
    max_tree_depth : int
    	NUTS tree-depth cap. (Default: 10)
    trajectory_length : float
    	HMC trajectory length in mass-whitened units; pi/2 is a quarter orbit of a Gaussian target
    	with a matched mass matrix. (Default: pi/2)
    target_accept : float
    	Dual-averaging target acceptance. (Default: 0.8)
    n_transitions : int
    	numpyro transitions per call of the move. (Default: 1)
    seed : int
    	Seed of the chains' JAX keys. (Default: 0)
    **kwargs
    	Passed to eryn.moves.Move.

    Returns
    ------------------------------
    None

    '''

    def __init__(self, post, H_prior, H_like, tempering='flow_prior', y_box=3.0, kernel='nuts', adapt='laplace',
                 num_warmup=0, max_tree_depth=10, trajectory_length=np.pi/2, target_accept=0.8, n_transitions=1,
                 seed=0, **kwargs):
        if tempering not in fe.TEMPERINGS or kernel not in KERNELS or adapt not in ADAPTS:
            raise ValueError("tempering, kernel and adapt must be in {}, {}, {}; got {!r}, {!r}, {!r}".format(
                fe.TEMPERINGS, KERNELS, ADAPTS, tempering, kernel, adapt))
        Move.__init__(self, **kwargs)
        self.post, self.tempering = post, tempering
        self.H_prior, self.H_like = np.asarray(H_prior, np.float64), np.asarray(H_like, np.float64)
        y_data = post.data_spectrum()
        self.box = (y_data - y_box, y_data + y_box)
        self.num_warmup, self.n_transitions, self.seed = int(num_warmup), int(n_transitions), int(seed)
        self.settings = dict(kernel=kernel, adapt=adapt, num_warmup=self.num_warmup, max_tree_depth=int(max_tree_depth),
                             trajectory_length=float(trajectory_length), target_accept=float(target_accept),
                             n_transitions=self.n_transitions, y_box=float(y_box))
        self.frozen = self.num_warmup == 0
        self.states, self._ntemps = None, None
        self.burn_diagnostics = None
        self._diag = None
        self._build()

    def _build(self):
        '''Build the vmapped, jitted numpyro init and step functions (flow parameters as arguments).'''
        s = self.settings
        static = dict(tempering=self.tempering, slices=self.post.slices, n_quad=self.post.n_quad, width=self.post.width)
        box = tuple(jnp.asarray(b) for b in self.box)

        def potential_gen(theta, beta, models, transforms, ctx_transforms, c):
            return lambda y: -_ln_conditional(models, transforms, ctx_transforms, c, theta, y, beta, box, **static)

        init_kernel, sample_kernel = hmc(potential_fn_gen=potential_gen, algo='NUTS' if s['kernel'] == 'nuts' else 'HMC')
        tl = s['trajectory_length'] if s['kernel'] == 'hmc' else None
        ## numpyro's built-in step-size search (find_heuristic_step_size) is off, since its warmup would keep
        ## the first trace's potential; the same search runs once per chain here instead
        init_kw = dict(adapt_step_size=True, adapt_mass_matrix=s['adapt'] == 'numpyro', dense_mass=True,
                       target_accept_prob=s['target_accept'], trajectory_length=tl, max_tree_depth=s['max_tree_depth'],
                       find_heuristic_step_size=False)
        num_warmup, n_transitions, overwrite_mass = self.num_warmup, self.n_transitions, s['adapt'] == 'laplace'

        def init_all(flow_args, y, theta, beta, inv_mass, keys):
            def one(yy, tt, bb, mm, kk):
                k_step, k_init = jax.random.split(kk)
                ## the starting step: halved or doubled from INITIAL_STEP until one leapfrog step's acceptance
                ## crosses 0.8, with momenta drawn from this chain's mass matrix
                _, sq, _ = _initialize_mass_matrix(yy, mm, True)
                momenta = lambda z, _, key: momentum_generator(z, sq, key)
                step = find_reasonable_step_size(potential_gen(tt, bb, *flow_args), euclidean_kinetic_energy, momenta,
                                                 INITIAL_STEP, mm, IntegratorState(yy, None, None, None), k_step)
                return init_kernel(yy, num_warmup, step_size=step, inverse_mass_matrix=mm,
                                   model_args=(tt, bb) + tuple(flow_args), rng_key=k_init, **init_kw)
            return jax.vmap(one)(y, theta, beta, inv_mass, keys)

        def step(flow_args, states, y, theta, beta, inv_mass):
            ## theta, beta and the walker in each slot change between calls (other moves, swaps), so the
            ## cached potential and gradient are recomputed at the current point
            pe_fn = lambda yy, tt, bb: potential_gen(tt, bb, *flow_args)(yy)
            pe, grad = jax.vmap(jax.value_and_grad(pe_fn))(y, theta, beta)
            states = states._replace(z=y, potential_energy=pe, z_grad=grad)
            if overwrite_mass:
                ## the Laplace matrix of the slot's current temperature, with numpyro's square roots for
                ## the momentum draws and kinetic energy
                _, sq, sq_inv = jax.vmap(lambda m: _initialize_mass_matrix(y[0], m, True))(inv_mass)
                states = states._replace(adapt_state=states.adapt_state._replace(
                    inverse_mass_matrix=inv_mass, mass_matrix_sqrt=sq, mass_matrix_sqrt_inv=sq_inv))
            kern = jax.vmap(lambda st, tt, bb: sample_kernel(st, model_args=(tt, bb) + tuple(flow_args)))

            def body(st, _):
                st = kern(st, theta, beta)
                return st, (st.accept_prob, st.num_steps, st.diverging)

            states, (acc, nsteps, div) = jax.lax.scan(body, states, None, length=n_transitions)
            return states, acc.mean(axis=0), nsteps.sum(axis=0), div.sum(axis=0)

        self._init_all, self._step = jax.jit(init_all), jax.jit(step)

    def inverse_mass(self, betas):
        '''(ntemps, Nf', Nf') Laplace inverse mass matrices at these inverse temperatures.'''
        return laplace_inverse_mass(self.H_prior, self.H_like, betas, self.tempering)

    def _new_diag(self, ntemps):
        return dict(calls=0, accept_prob=np.zeros(ntemps), num_steps=np.zeros(ntemps), divergences=np.zeros(ntemps))

    def propose(self, model, state):
        '''

        Run the numpyro chains on 'spectrum' at fixed 'pop', then the temperature swaps.

        Arguments
        ------------------------------
        model : eryn.model.Model
        	Carrier of the sampler's prior and likelihood functions.
        state : eryn.state.State
        	Current state.

        Returns
        ------------------------------
        state : eryn.state.State
        	State after the move and the swaps.
        moved : numpy.ndarray
        	(ntemps, nwalkers) whether each walker's y changed.

        '''
        coords = state.branches_coords
        ntemps, nwalkers = state.log_like.shape
        theta, y = coords['pop'][:, :, 0], coords['spectrum'][:, :, 0]
        nf = y.shape[-1]
        betas = (np.asarray(self.temperature_control.betas, np.float64) if self.temperature_control is not None
                 else np.ones(ntemps))
        th_f, y_f, b_f = (jnp.asarray(theta.reshape(-1, 8)), jnp.asarray(y.reshape(-1, nf)),
                          jnp.asarray(np.repeat(betas, nwalkers)))
        inv_mass = jnp.asarray(np.repeat(self.inverse_mass(betas), nwalkers, axis=0))
        flow_args = self.post._args()
        if self.states is None:
            keys = jax.random.split(jax.random.key(self.seed), ntemps*nwalkers)
            self.states = self._init_all(flow_args, y_f, th_f, b_f, inv_mass, keys)
            if self.frozen:
                self.states = self.states._replace(i=jnp.maximum(self.states.i, self.num_warmup))
        if self._diag is None:
            self._diag = self._new_diag(ntemps)
        self._ntemps = ntemps
        self.states, acc, nsteps, div = self._step(flow_args, self.states, y_f, th_f, b_f, inv_mass)

        ## write the new spectra, and let Eryn recompute its prior and likelihood there
        y_new = np.array(self.states.z).reshape(ntemps, nwalkers, 1, nf)
        q = {'pop': coords['pop'].copy(), 'spectrum': y_new}
        logp = model.compute_log_prior_fn(q, inds=state.branches_inds)
        logl, blobs = model.compute_log_like_fn(q, inds=state.branches_inds, logp=logp, supps=state.supplemental,
                                                branch_supps=state.branches_supplemental)
        new_state = State(q, log_like=logl, log_prior=logp, blobs=blobs, inds=state.branches_inds,
                          supplemental=state.supplemental, branch_supplemental=state.branches_supplemental)
        moved = np.any(y_new[:, :, 0] != y, axis=-1)
        ## numpyro already accepted or rejected each trajectory, so every slot takes its new y
        state = self.update(state, new_state, np.ones((ntemps, nwalkers), dtype=bool))
        self.accepted += moved
        self.num_proposals += 1

        d = self._diag
        d['calls'] += 1
        d['accept_prob'] += np.asarray(acc).reshape(ntemps, nwalkers).mean(axis=1)
        d['num_steps'] += np.asarray(nsteps).reshape(ntemps, nwalkers).mean(axis=1)/self.n_transitions
        d['divergences'] += np.asarray(div).reshape(ntemps, nwalkers).sum(axis=1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMC call %d: accept %s, leapfrog steps %s, %d divergences", d['calls'],
                         np.round(np.asarray(acc).reshape(ntemps, nwalkers).mean(axis=1), 2).tolist(),
                         np.round(np.asarray(nsteps).reshape(ntemps, nwalkers).mean(axis=1), 1).tolist(),
                         int(np.asarray(div).sum()))

        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)
        return state, moved

    def freeze(self):
        '''

        End adaptation: each chain keeps its averaged step size (and, for 'numpyro', its mass matrix)
        and adapts no further. The diagnostics so far are kept as burn_diagnostics and restarted.

        Arguments
        ------------------------------
        None

        Returns
        ------------------------------
        None

        '''
        if self.states is not None and not self.frozen:
            a, i = self.states.adapt_state, self.states.i
            _, log_step_avg, _, t, _ = a.ss_state
            ## numpyro switches to the averaged step only at the end of its own warmup
            step = jnp.where((i < self.num_warmup) & (t > 0), jnp.exp(log_step_avg), a.step_size)
            self.states = self.states._replace(i=jnp.maximum(i, self.num_warmup), adapt_state=a._replace(step_size=step))
        self.frozen = True
        self.burn_diagnostics = self.diagnostics()
        self._diag = None

    def diagnostics(self):
        '''

        Per-temperature diagnostics since the start (or since freeze).

        Arguments
        ------------------------------
        None

        Returns
        ------------------------------
        diagnostics : dict
        	calls; mean accept_prob, mean num_steps (leapfrog steps per transition) and total
        	divergences per temperature; per-temperature median, min and max step size; settings.

        '''
        out = dict(settings=self.settings, frozen=self.frozen)
        d = self._diag
        if d is not None and d['calls']:
            out.update(calls=d['calls'], accept_prob=(d['accept_prob']/d['calls']).tolist(),
                       num_steps=(d['num_steps']/d['calls']).tolist(), divergences=d['divergences'].astype(int).tolist())
        if self.states is not None:
            step = np.asarray(self.states.adapt_state.step_size).reshape(self._ntemps, -1)
            out.update(step_size_median=np.median(step, axis=1).tolist(), step_size_min=step.min(axis=1).tolist(),
                       step_size_max=step.max(axis=1).tolist())
        return out
