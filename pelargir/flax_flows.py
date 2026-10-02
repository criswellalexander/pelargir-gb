"""
Flax NNX flow base for the per-band emulator (flows.py is the zuko base; both share the data
layer in flow_data.py).

Adapted from ATLAS/experimental/flax_flows.py (N. Laal, adapted to Flax by A. W. Criswell): a stack of masked-coupling
rational-quadratic-spline (RQS) layers over a standard-normal base; conditioner MLPs map
[y_masked || c] to the spline parameters, with zero-initialised outputs so an untrained flow is
exactly its base; optax Adam; and pure jax.jit functions that take the model as an argument, so
log densities compose under an outer jit / grad / vmap (the entry point for NUTS is
BandedFlowEmulator.jax_log_prob).

Transforms are as follows: N_res dequantized with U(0,1); log10 of N_res and S_gw (and
lambda_tot, already in the context); every dimension standardized by (x - median)/half-range.
Each data dimension is then scaled by B/max|z| over the training set, so the training data fill
[-B, B]; the splines act on [-B-1, B+1] and are the identity outside. log_prob integrates the
dequantization out by Gauss-Legendre over u and includes every transform's Jacobian, so it is the
density of (N_res, S_gw) in physical units.
"""
import json
import math
import os
from functools import partial
from typing import NamedTuple

import numpy as np

from . import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax import lax
import distrax
import optax
from flax import nnx

from .flow_data import (CONTEXT_NAMES, FrequencyBand, make_bands, band_view, drop_zero_spectra, _standardizer,
                        gauss_legendre_01)

_LN10 = math.log(10.0)
_FORMAT = "pelargir-flax-nnx-v1"
DTYPES = {'float32': jnp.float32, 'float64': jnp.float64}


# =============================================================================
# Transforms (pure functions of fitted constants)
# =============================================================================

class BandTransform(NamedTuple):
    '''Fitted data transform of one band: y = (log10 x - loc)/scale*spline_scale, x = (N_deq, S...).'''
    loc: object           ## (1+nf,) medians of log10 x
    scale: object         ## (1+nf,) half-ranges of log10 x
    spline_scale: object  ## (1+nf,) B/max|z|, so the training data fill [-B, B]


class ContextTransform(NamedTuple):
    '''Fitted context standardization (c - loc)/scale.'''
    loc: object
    scale: object


def fit_band_transform(N, S, rng, B):
    '''BandTransform from training counts N (M,) and spectra S (M, nf); N dequantized with rng.'''
    S = np.asarray(S, dtype=np.float64)
    if np.any(S <= 0):
        raise ValueError("S_gw must be positive in every bin; {} of {} values are not (zero-foreground bins "
                         "are not supported yet)".format(int(np.sum(S <= 0)), S.size))
    x = np.column_stack([np.log10(np.asarray(N) + rng.uniform(size=len(N))), np.log10(S)])
    med, half = _standardizer(x)
    zmax = np.max(np.abs((x - med)/half), axis=0)
    return BandTransform(jnp.asarray(med), jnp.asarray(half), jnp.asarray(B/np.where(zmax > 0, zmax, 1.0)))


def fit_context_transform(context):
    med, half = _standardizer(np.asarray(context, dtype=np.float64))
    return ContextTransform(jnp.asarray(med), jnp.asarray(half))


def band_forward(t, N_deq, S):
    '''y (..., 1+nf) and log|det dy/dx| (...,) for dequantized counts N_deq (...,) and S (..., nf).'''
    x = jnp.concatenate([jnp.asarray(N_deq, jnp.float64)[..., None], jnp.asarray(S, jnp.float64)], axis=-1)
    y = (jnp.log10(x) - t.loc)/t.scale*t.spline_scale
    logdet = -jnp.sum(jnp.log(x*_LN10*t.scale/t.spline_scale), axis=-1)
    return y, logdet


def band_inverse(t, y):
    '''(N_res, S_gw) from y; N_res is the floor of the dequantized count.'''
    x = 10**(jnp.asarray(y, jnp.float64)/t.spline_scale*t.scale + t.loc)
    return jnp.floor(x[..., 0]), x[..., 1:]


def context_forward(ct, context):
    return (jnp.asarray(context, jnp.float64) - ct.loc)/ct.scale


# =============================================================================
# NNX modules (from ATLAS flax_flows.py)
# =============================================================================

class _Conditioner(nnx.Module):
    '''
    MLP mapping [y_masked || c] to RQS parameters of shape (..., D, 3K+1). Every hidden layer is
    ReLU-activated, including the last; the output layer is zero-initialised, so each spline starts
    as the identity.
    '''

    def __init__(self, in_features, D, hidden_sizes, num_params, dtype, *, rngs):
        sizes = [in_features, *hidden_sizes]
        self.hidden = nnx.List([nnx.Linear(a, b, dtype=dtype, param_dtype=dtype, rngs=rngs)
                                for a, b in zip(sizes[:-1], sizes[1:])])
        self.out = nnx.Linear(sizes[-1], D*num_params, kernel_init=nnx.initializers.zeros_init(),
                              bias_init=nnx.initializers.zeros_init(), dtype=dtype, param_dtype=dtype, rngs=rngs)
        self.D = D
        self.num_params = num_params

    def __call__(self, y_masked, c):
        h = jnp.concatenate([y_masked, c], axis=-1)
        for layer in self.hidden:
            h = jax.nn.relu(layer(h))
        return self.out(h).reshape(h.shape[:-1] + (self.D, self.num_params))


class _CouplingRQSFlow(nnx.Module):
    '''
    Masked-coupling RQS layers over N(0, I). Layer i transforms the features where the alternating
    arange(D) % 2 mask is False, with spline parameters from the others and the context. The chain
    is wrapped in Inverse, so log_prob runs data -> latent and forward runs latent -> data.
    '''

    def __init__(self, D, C, num_layers, hidden_sizes, num_bins, B, dtype, *, rngs):
        self.D = int(D)
        self.C = int(C)
        self.num_bins = int(num_bins)
        ## a Python float: RationalQuadraticSpline checks range_min < range_max at construction
        self.B = float(B)
        self.dtype = dtype
        num_params = 3*self.num_bins + 1
        self.conditioners = nnx.List([_Conditioner(self.D + self.C, self.D, hidden_sizes, num_params, dtype, rngs=rngs)
                                      for _ in range(num_layers)])

    def _bijector(self, c):
        B = self.B

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(params, range_min=-B - 1, range_max=B + 1)

        mask = (jnp.arange(self.D) % 2).astype(bool)
        layers = []
        for conditioner in self.conditioners:
            layers.append(distrax.MaskedCoupling(mask=mask, bijector=bijector_fn,
                                                 conditioner=partial(conditioner, c=c)))
            mask = jnp.logical_not(mask)
        return distrax.Inverse(distrax.Chain(layers))

    def log_prob(self, y, c):
        base = distrax.Independent(distrax.Normal(jnp.zeros(self.D, self.dtype), jnp.ones(self.D, self.dtype)),
                                   reinterpreted_batch_ndims=1)
        return distrax.Transformed(base, self._bijector(c)).log_prob(y)

    def forward(self, z, c):
        return self._bijector(c).forward(z)


# =============================================================================
# Pure jitted functions (the model is an argument, never a closed-over constant)
# =============================================================================

@partial(jax.jit, static_argnames=('n_quad',))
def _log_prob_band(model, t, ct, context, N, S, n_quad):
    '''
    log P(N_res = N, S_gw = S | context), shape (M,): log int_0^1 q(N + u, S | c) du by
    Gauss-Legendre over u, with the transform Jacobian. Transforms run in float64, the flow in its dtype.
    '''
    u, logw = gauss_legendre_01(n_quad)
    c = context_forward(ct, context).astype(model.dtype)
    N = jnp.asarray(N, jnp.float64)
    K, M = n_quad, N.shape[0]
    y, logdet = band_forward(t, N[None, :] + jnp.asarray(u)[:, None], jnp.broadcast_to(S, (K,) + S.shape))
    lq = model.log_prob(y.astype(model.dtype).reshape(K*M, -1), jnp.broadcast_to(c, (K,) + c.shape).reshape(K*M, -1))
    lq = lq.astype(jnp.float64).reshape(K, M) + logdet + jnp.asarray(logw)[:, None]
    return jax.scipy.special.logsumexp(lq, axis=0)


@jax.jit
def _sample_band(model, t, ct, key, context):
    '''One (N_res, S_gw) draw per context row.'''
    c = context_forward(ct, context).astype(model.dtype)
    z = jax.random.normal(key, (c.shape[0], model.D), dtype=model.dtype)
    return band_inverse(t, model.forward(z, c))


@jax.jit
def _nll(model, y, c):
    return -jnp.mean(model.log_prob(y, c))


@jax.jit
def _train_step(model, optimizer, y, c):
    '''One Adam step on the mean NLL in the flow's space; returns the updated (model, optimizer, loss).'''
    loss, grads = nnx.value_and_grad(lambda m: -jnp.mean(m.log_prob(y, c)))(model)
    optimizer.update(model, grads)
    return model, optimizer, loss


@jax.jit
def _train_epoch(model, optimizer, y, c, order):
    '''
    One epoch of Adam steps as a lax.scan over the batches order (n_batches, batch_size) of rows
    of y and c; returns the updated (model, optimizer) and the per-step losses.
    '''
    def step(carry, idx):
        m, opt = carry
        loss, grads = nnx.value_and_grad(lambda mm: -jnp.mean(mm.log_prob(y[idx], c[idx])))(m)
        opt.update(m, grads)
        return (m, opt), loss
    (model, optimizer), losses = lax.scan(step, (model, optimizer), order)
    return model, optimizer, losses


# =============================================================================
# Band flows and the emulator
# =============================================================================

class BandFlow:
    '''
    One band's flow: a _CouplingRQSFlow over y = BandTransform(N_res + u, S_gw), conditioned on the
    standardized context.
    '''

    def __init__(self, band, n_context=len(CONTEXT_NAMES), flow_num_layers=4, hidden_size=128, mlp_num_layers=2,
                 num_bins=8, B=4.0, dtype='float64', seed=0):
        self.band = band
        self.features = 1 + band.nf
        self.n_context = n_context
        self.kwargs = dict(flow_num_layers=flow_num_layers, hidden_size=hidden_size, mlp_num_layers=mlp_num_layers,
                           num_bins=num_bins, B=float(B), dtype=dtype)
        self.model = _CouplingRQSFlow(self.features, n_context, flow_num_layers, [hidden_size]*mlp_num_layers,
                                      num_bins, B, DTYPES[dtype], rngs=nnx.Rngs(params=jax.random.key(seed)))
        self.transform = None
        self.context_transform = None

    @property
    def dtype(self):
        return self.model.dtype

    def n_params(self):
        return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(nnx.state(self.model, nnx.Param))))

    def fit_transforms(self, context, N, S, rng):
        self.transform = fit_band_transform(N, S, rng, self.kwargs['B'])
        self.context_transform = fit_context_transform(context)
        return self

    def log_prob(self, context, N, S, n_quad=32):
        '''log P(N, S | context), shape (M,), as a JAX array; differentiable in context and S.'''
        return _log_prob_band(self.model, self.transform, self.context_transform, jnp.atleast_2d(context),
                              jnp.asarray(N), jnp.asarray(S, jnp.float64), n_quad)

    def sample(self, context, key):
        return _sample_band(self.model, self.transform, self.context_transform, key, jnp.atleast_2d(context))

    def config(self):
        return dict(band=self.band.to_dict(), features=self.features, **self.kwargs)


def train_band_flow(flow, context, N, S, n_epochs=8, batch_size=64, lr=1e-3, val_frac=0.1, seed=0, progress=True):
    '''
    Fit flow's transforms and train it by maximum likelihood in the flow's space, re-dequantizing N
    every epoch (as flows.train_band_flow). Each epoch is one jitted lax.scan over its batches. The
    last partial batch of an epoch is dropped, so every batch has one shape (with fewer rows than
    batch_size, each epoch is one batch of all of them). Returns the per-epoch train and validation
    losses.
    '''
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(N))
    n_val = int(round(val_frac*len(N)))
    val, trn = rows[:n_val], rows[n_val:]
    flow.fit_transforms(context[trn], N[trn], S[trn], rng)
    dt = flow.dtype

    c_all = context_forward(flow.context_transform, context).astype(dt)
    N_all, S_all = jnp.asarray(N, jnp.float64), jnp.asarray(S, jnp.float64)
    ## fixed dequantization for the validation loss, so it is comparable across epochs
    y_val = band_forward(flow.transform, N_all[val] + jnp.asarray(rng.uniform(size=n_val)), S_all[val])[0].astype(dt)
    c_val, c_trn = c_all[val], c_all[trn]

    model = flow.model
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    history = dict(train=[], val=[])
    ## fewer rows than one batch: train on all of them each step
    batch_size = min(batch_size, len(trn))
    n_batches = len(trn)//batch_size
    for epoch in range(n_epochs):
        u = jnp.asarray(rng.uniform(size=len(trn)))
        y_trn = band_forward(flow.transform, N_all[trn] + u, S_all[trn])[0].astype(dt)
        order = jnp.asarray(rng.permutation(len(trn))[:n_batches*batch_size].reshape(n_batches, batch_size))
        model, optimizer, losses = _train_epoch(model, optimizer, y_trn, c_trn, order)
        history['train'].append(float(jnp.mean(losses)))
        history['val'].append(float(_nll(model, y_val, c_val)) if n_val else float('nan'))
        if progress:
            print("band {} epoch {}: train {:.4f}, val {:.4f}".format(flow.band.index, epoch, history['train'][-1],
                                                                     history['val'][-1]), flush=True)
    flow.model = model
    return history


class BandedFlowEmulator:
    '''
    One BandFlow per band; the population-conditional density of (N_res per band, S_gw per bin)
    is the product over bands. Same interface as flows.BandedFlowEmulator, plus jax_log_prob.
    '''

    def __init__(self, bands, fbins, seed=0, **flow_kwargs):
        self.bands = bands
        self.fbins = np.asarray(fbins)
        self.flow_kwargs = flow_kwargs
        self.seed = seed
        self.flows = [BandFlow(b, seed=seed + j, **flow_kwargs) for j, b in enumerate(bands)]
        self.key = jax.random.key(seed + 12345)

    @property
    def fs(self):
        return self.fbins[1:]

    def n_params(self):
        return [f.n_params() for f in self.flows]

    def jax_log_prob(self, context, N_bands, S, n_quad=32):
        '''
        Sum over bands of log P(N_band, S_band | context), shape (M,), as a pure JAX function of
        context (M, 8), N_bands (M, n_bands) and S (M, Nf'): usable inside jit / grad / vmap.
        '''
        total = 0.0
        for j, f in enumerate(self.flows):
            total = total + _log_prob_band(f.model, f.transform, f.context_transform, context, N_bands[:, j],
                                           S[:, f.band.slice], n_quad)
        return total

    def _broadcast(self, context, N_bands, S):
        context = np.atleast_2d(np.asarray(context, dtype=np.float64))
        M = context.shape[0]
        N_bands = np.array(np.broadcast_to(np.asarray(N_bands), (M, len(self.bands))), dtype=np.float64)
        S = np.array(np.broadcast_to(np.asarray(S, dtype=np.float64), (M, self.fs.size)))
        return context, N_bands, S

    def log_prob(self, context, N_bands, S, n_quad=32):
        '''
        As flows.BandedFlowEmulator.log_prob, returning numpy: context (M, 8); N_bands (n_bands,)
        or (M, n_bands); S (Nf',) or (M, Nf') on fbins[1:].
        '''
        context, N_bands, S = self._broadcast(context, N_bands, S)
        return np.asarray(self.jax_log_prob(jnp.asarray(context), jnp.asarray(N_bands), jnp.asarray(S), n_quad))

    def band_log_prob(self, j, context, N, S, n_quad=32):
        '''log P(N, S | context) for band j alone, shape (M,), as numpy.'''
        return np.asarray(self.flows[j].log_prob(np.atleast_2d(context), N, S, n_quad=n_quad))

    def sample(self, context):
        '''One draw per context row: N_res per band (M, n_bands) and S_gw (M, Nf'), NaN outside the bands.'''
        context = np.atleast_2d(context)
        N = np.zeros((context.shape[0], len(self.bands)))
        S = np.full((context.shape[0], self.fs.size), np.nan)
        for j, f in enumerate(self.flows):
            self.key, k = jax.random.split(self.key)
            Nj, Sj = f.sample(context, k)
            N[:, j] = np.asarray(Nj)
            S[:, f.band.slice] = np.asarray(Sj)
        return N, S

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        meta = dict(flow_base='flax', format=_FORMAT, fbins=self.fbins.tolist(),
                    bands=[b.to_dict() for b in self.bands], flow_kwargs=self.flow_kwargs, seed=self.seed,
                    context_names=CONTEXT_NAMES,
                    transforms=[{k: np.asarray(v).tolist() for k, v in f.transform._asdict().items()}
                                for f in self.flows],
                    context_transforms=[{k: np.asarray(v).tolist() for k, v in f.context_transform._asdict().items()}
                                        for f in self.flows])
        with open(os.path.join(directory, 'emulator.json'), 'w') as fh:
            json.dump(meta, fh, indent=1)
        params = {}
        for j, f in enumerate(self.flows):
            for path, v in _flat_params(f.model).items():
                params['band{}/{}'.format(j, path)] = v
        np.savez(os.path.join(directory, 'emulator.npz'), **params)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, 'emulator.json')) as fh:
            meta = json.load(fh)
        if meta.get('format') != _FORMAT:
            raise ValueError("{} is not a {} emulator".format(directory, _FORMAT))
        fbins = np.asarray(meta['fbins'])
        bands = [FrequencyBand(b['index'], b['start'], b['stop'], fbins[1:][b['start']:b['stop']])
                 for b in meta['bands']]
        em = cls(bands, fbins, seed=meta['seed'], **meta['flow_kwargs'])
        data = np.load(os.path.join(directory, 'emulator.npz'))
        for j, f in enumerate(em.flows):
            f.transform = BandTransform(**{k: jnp.asarray(v) for k, v in meta['transforms'][j].items()})
            f.context_transform = ContextTransform(**{k: jnp.asarray(v) for k, v in meta['context_transforms'][j].items()})
            prefix = 'band{}/'.format(j)
            _set_params(f.model, {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}, directory)
        return em


def _path_str(path):
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


def _flat_params(model):
    '''Parameters keyed by their path in the module tree, as numpy (ATLAS _flat_params).'''
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    return {_path_str(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(pure)[0]}


def _set_params(model, flat, source):
    '''Load flat path-keyed parameters into model, which must have the same architecture.'''
    state = nnx.state(model, nnx.Param)
    pure = nnx.to_pure_dict(state)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pure)
    want = {_path_str(p): np.shape(v) for p, v in leaves}
    got = {k: np.shape(v) for k, v in flat.items()}
    if want != got:
        differ = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError("{} holds a different architecture; mismatched parameters: {}".format(source, differ[:6]))
    new = [jnp.asarray(flat[_path_str(p)], dtype=np.asarray(v).dtype) for p, v in leaves]
    nnx.replace_by_pure_dict(state, jax.tree_util.tree_unflatten(treedef, new))
    nnx.update(model, state)


def train_emulator(ts, bands=None, flow_kwargs=None, seed=0, **train_kwargs):
    '''
    Train a BandedFlowEmulator on a TrainingSet; returns (emulator, per-band loss histories). Each
    band drops its rows with S_gw = 0 (flow_data.drop_zero_spectra); history['n_dropped'] counts them.
    '''
    bands = make_bands(ts.fs) if bands is None else bands
    em = BandedFlowEmulator(bands, ts.fbins, seed=seed, **(flow_kwargs or {}))
    histories = []
    for f in em.flows:
        context, N, S, n_dropped = drop_zero_spectra(*band_view(ts, f.band), band=f.band)
        h = train_band_flow(f, context, N, S, seed=seed, **train_kwargs)
        h['n_dropped'] = n_dropped
        histories.append(h)
    return em, histories
