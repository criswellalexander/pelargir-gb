"""
JAX forward model: draw each galaxy's binaries from the conditional population prior and
threshold them, in one jitted batch, without materializing the (4, N, Nreal, Nparallel) draw.

The samplers reproduce inference.GalacticBinaryPrior (distributions.truncnorm for the
masses, gaussian_exponential_mixture for d_L, powerlaw for a) as pure functions of a PRNG
key. Galaxy g of a call gets key fold_in(call_key, g), independent of batching.

Per-galaxy N: every galaxy in a batch is drawn at a static padded size n_pad >= its N
(a x1.25 geometric bucket), and entries at index >= N are given f = +inf, A = 0, i.e. out
of band, which every thresholder ignores exactly. With jax_threefry_partitionable (set in
backend.import_jax) draws are prefix-stable in n, so a galaxy's first N binaries, and hence
the results, do not depend on n_pad.

forward_galaxies runs many galaxies (training sets) at a few fixed shapes: one batch width per
padding bucket, sized to the GPU's memory, and fixed pre-filter capacity tiers.
"""
from collections import deque
from functools import partial
from typing import NamedTuple
import math
import os
import time

from . import backend

jax = backend.import_jax()
import jax.numpy as jnp

import numpy as np
from astropy import units as u

from . import jax_thresholding as jt

MSUN_KG = float((1*u.Msun).to(u.kg).value)
KPC_M = float((1*u.kpc).to(u.m).value)
AU_M = float((1*u.AU).to(u.m).value)
G = 6.6743e-11 ## m^3 kg^-1 s^-2
c = 2.99792458e8 ## m/s

MIN_PAD = 1024
PAD_GROWTH = 1.25


def prior_bounds(gbprior):
    '''Static sampler bounds (m_min, m_max, a_min, a_max, x0) from a GalacticBinaryPrior.'''
    return (float(gbprior.m_min), float(gbprior.m_max), float(gbprior.a_min), float(gbprior.a_max),
            float(gbprior.galactic_center))


class GBPriorParams(NamedTuple):
    '''Parameters of the conditional GB prior (see conditional_params).'''
    m_loc: object    ## truncnorm mean of m_1 and m_2 [Msun]
    m_scale: object  ## truncnorm standard deviation [Msun]
    m_min: float     ## truncation bounds [Msun]
    m_max: float
    x0: float        ## Galactic centre distance [kpc]
    r_bulge: object  ## bulge Gaussian scale [kpc]
    rh_disk: object  ## disk exponential scale [kpc]
    q_bd: object     ## bulge probability
    a_alpha: object  ## power-law index of p(a) ∝ (a - a_loc)^a_alpha
    a_loc: float     ## a_min [AU]
    a_scale: float   ## a_max - a_min [AU]


def conditional_params(theta, bounds):
    '''
    JAX counterpart of GalacticBinaryPrior.condition: the conditional prior's parameters
    for hyperparameters theta.

    Arguments
    -----------
    theta (array) : Shape (..., 6), GalacticBinaryPrior.pop_params order.
    bounds (tuple) : (m_min, m_max, a_min, a_max, x0), see prior_bounds.

    Returns
    -----------
    GBPriorParams; hyperparameter-dependent fields have shape theta.shape[:-1].
    '''
    m_min, m_max, a_min, a_max, x0 = bounds
    m_mu, m_sigma, rh_disk, r_bulge, q_bd, a_alpha = (theta[..., i] for i in range(6))
    return GBPriorParams(m_mu, m_sigma, m_min, m_max, x0, r_bulge, rh_disk, q_bd,
                         a_alpha, a_min, a_max - a_min)


def sample_theta(key, theta, n, bounds):
    '''
    Draw n binaries for one galaxy.

    Arguments
    -----------
    key : PRNG key for this galaxy.
    theta (array) : (m_mu, m_sigma, rh_disk, r_bulge, q_bd, a_alpha), GalacticBinaryPrior.pop_params order.
    n (int) : Number of binaries.
    bounds (tuple) : (m_min, m_max, a_min, a_max, x0), see prior_bounds.

    Returns
    -----------
    (4, n) array of (m_1 [Msun], m_2 [Msun], d_L [kpc], a [AU]).
    '''
    p = conditional_params(theta, bounds)
    k_m1, k_m2, k_mix, k_bulge, k_disk, k_dir, k_a = jax.random.split(key, 7)
    f64 = jnp.float64

    lo = (p.m_min - p.m_loc)/p.m_scale
    hi = (p.m_max - p.m_loc)/p.m_scale
    m_1 = p.m_loc + p.m_scale*jax.random.truncated_normal(k_m1, lo, hi, (n,), dtype=f64)
    m_2 = p.m_loc + p.m_scale*jax.random.truncated_normal(k_m2, lo, hi, (n,), dtype=f64)

    ## bulge with probability q_bd, else the disk on the far or near side with probability 1/2 each
    bulge = p.x0 + p.r_bulge*jax.random.normal(k_bulge, (n,), dtype=f64)
    side = jnp.where(jax.random.uniform(k_dir, (n,), dtype=f64) <= 0.5, 1.0, -1.0)
    disk = p.x0 + side*p.rh_disk*jax.random.exponential(k_disk, (n,), dtype=f64)
    d_L = jnp.abs(jnp.where(jax.random.uniform(k_mix, (n,), dtype=f64) <= p.q_bd, bulge, disk))

    ## p(a) ∝ (a - a_min)^a_alpha on [a_min, a_max], as distributions.powerlaw
    a = p.a_loc + p.a_scale*jax.random.uniform(k_a, (n,), dtype=f64)**(1.0/(p.a_alpha + 1.0))
    return jnp.stack([m_1, m_2, d_L, a])


def amp_freq(theta4):
    '''JAX port of utils.get_amp_freq: (amplitude, GW frequency [Hz]) from (m_1, m_2, d_L, a).'''
    m_1 = theta4[0]*MSUN_KG
    m_2 = theta4[1]*MSUN_KG
    d_L = theta4[2]*KPC_M
    a = theta4[3]*AU_M
    amp = (8/jnp.sqrt(5)) * (G**2/c**4) * (m_1*m_2)/(d_L*a)
    fgw = 1/jnp.pi * jnp.sqrt(G*(m_1+m_2)/a**3)
    return amp, fgw


def galaxy_obs(key, theta, N, n_pad, bounds):
    '''(f, A) of length n_pad for one galaxy of N binaries; entries at index >= N are out of band.'''
    A, f = amp_freq(sample_theta(key, theta, n_pad, bounds))
    valid = jnp.arange(n_pad) < N
    return jnp.where(valid, f, jnp.inf), jnp.where(valid, A, 0.0)


def pad_bucket(n):
    '''Padded size for n binaries: the smallest ceil(MIN_PAD*PAD_GROWTH**k) >= n.'''
    n = max(int(n), 1)
    k = max(0, math.ceil(math.log(n/MIN_PAD)/math.log(PAD_GROWTH))) if n > MIN_PAD else 0
    while math.ceil(MIN_PAD*PAD_GROWTH**k) < n:
        k += 1
    while k > 0 and math.ceil(MIN_PAD*PAD_GROWTH**(k-1)) >= n:
        k -= 1
    return math.ceil(MIN_PAD*PAD_GROWTH**k)


@partial(jax.jit, static_argnames=('bounds', 'n_pad', 'capacity', 'with_mask'))
def _forward_batch(keys, thetas, Ns, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh, cut,
                   bounds, n_pad, capacity=None, with_mask=True):
    '''
    Per-galaxy (Nres_f, fg_f, resolved, n_surv) for a batch: keys (B,), thetas (B,6), Ns (B,),
    and per-galaxy snr_thresh and pre-filter cut (B,). capacity None means no pre-filter
    (n_surv is then n_pad). with_mask False leaves out the per-binary mask (and its scatter).
    '''
    shared = (edges, noisePSD, LISA_rx, duration, duration_eff)

    def one(key, theta, N, rho, c):
        f, A = galaxy_obs(key, theta, N, n_pad, bounds)
        if capacity is None:
            out = (*jt._threshold_one(f, A, *shared, rho), jnp.int32(n_pad))
        else:
            out = jt._threshold_one_filtered(f, A, *shared, rho, c, capacity)
        return out if with_mask else (out[0], out[1], out[3])
    return jax.vmap(one)(keys, thetas, Ns, snr_thresh, cut)


@partial(jax.jit, static_argnames=('bounds', 'n_pad'))
def _forward_count_survivors(keys, thetas, Ns, edges, noisePSD, LISA_rx, duration, cut, bounds, n_pad):
    '''Per-galaxy count of binaries passing the per-bin pre-filter cut (B,), shape (B,).'''
    def count(key, theta, N, c):
        k, a = jt._prepare(*galaxy_obs(key, theta, N, n_pad, bounds), edges, LISA_rx)
        return jnp.sum(jt._survives(k, a, noisePSD, duration, c), dtype=jnp.int32)
    return jax.vmap(count)(keys, thetas, Ns, cut)


def _host(arr):
    '''Array (numpy, cupy, or jax) as a numpy array.'''
    if jt._is_cupy(arr):
        return arr.get()
    return np.asarray(arr)


def galaxy_keys(key, n_galaxies):
    '''Per-galaxy keys fold_in(key, g), g = 0..n_galaxies-1 (galaxy g = r*Nparallel + p).'''
    return jax.vmap(lambda g: jax.random.fold_in(key, g))(jnp.arange(n_galaxies))


def jax_forward_model(key, thetas, Ns, bounds, edges, noisePSD, LISA_rx, duration, duration_eff,
                      snr_thresh=7, batch_size=None, prefilter_snr=1.0, capacity_cache=None,
                      return_mask=False, return_nres_f=False, out_module=np):
    '''
    Draw and threshold every galaxy of a likelihood call.

    Arguments
    -----------
    key : PRNG key for this call.
    thetas (array) : Population hyperparameters, shape (Nparallel, 6), GalacticBinaryPrior.pop_params order.
    Ns (array)     : Binaries per galaxy, shape (Nrealz, Nparallel), non-negative ints.
    bounds (tuple) : Sampler bounds, see prior_bounds.
    edges, noisePSD, LISA_rx, duration, duration_eff : As in jax_thresholding.jax_threshold.
    snr_thresh (float or array) : Resolvability threshold, scalar or per parallel column (Nparallel,).
    batch_size (int) : Galaxies per jitted call. Default None (all at once).
    prefilter_snr (float) : Per-bin pre-filter cut (see jax_thresholding); None disables it. Default 1.
    capacity_cache (dict) : Survivor capacity carried between calls, keyed by (n_pad, batch).
    return_mask (bool) : Whether to also return the per-binary resolved mask.
    return_nres_f (bool) : Whether to also return the per-bin resolved counts.
    out_module : Array library for the outputs (numpy, or cupy via DLPack). Default numpy.

    Returns
    -----------
    Nres (Nrealz,Nparallel), foreground_amp (Nf,Nrealz,Nparallel)[, mask (n_max,Nrealz,Nparallel)]
    [, Nres_f (Nf,Nrealz,Nparallel)], squeezed over (Nrealz,Nparallel) when both are 1, as in the
    thresholders. mask rows past a galaxy's N are False; n_max is the largest padded size used.
    Nres_f includes bin 0, which Nres excludes.
    '''
    rho_min = float(np.min(_host(snr_thresh)))
    if prefilter_snr is not None and prefilter_snr > rho_min:
        raise ValueError("prefilter_snr ({}) must not exceed snr_thresh ({}); binaries between the "
                         "two could be resolved".format(prefilter_snr, rho_min))
    if capacity_cache is None:
        capacity_cache = {}
    cupy_module = out_module if out_module.__name__ == 'cupy' else None

    Ns = _host(Ns).astype(np.int64)
    thetas = _host(thetas).astype(np.float64).reshape(-1, 6)
    Nr, Np = Ns.shape
    if thetas.shape[0] != Np:
        raise ValueError("thetas has {} rows but Ns has {} parallel columns".format(thetas.shape[0], Np))
    G = Nr*Np
    B = G if batch_size is None else int(min(batch_size, G))
    Ns_g = Ns.reshape(G)
    thetas_g = np.tile(thetas, (Nr, 1))
    rhos_g = np.tile(np.broadcast_to(np.asarray(_host(snr_thresh), dtype=np.float64).reshape(-1), (Np,)), Nr)
    keys_g = galaxy_keys(key, G)

    consts = [jt._to_jax(np.ascontiguousarray(_host(c_), dtype=np.float64)) for c_ in (edges, noisePSD, LISA_rx)]
    scalars = [float(duration), float(duration_eff)]
    cut = 0.0 if prefilter_snr is None else float(prefilter_snr)

    Nres_f, fg_f, mask = [], [], []
    for g0 in range(0, G, B):
        n_real = min(B, G - g0)
        k_b = keys_g[g0:g0+n_real]
        th_b = thetas_g[g0:g0+n_real]
        N_b = Ns_g[g0:g0+n_real]
        rho_b = rhos_g[g0:g0+n_real]
        if n_real < B:
            ## padded galaxies have N = 0: all out of band, and their outputs are dropped
            k_b = jnp.concatenate([k_b, keys_g[:B-n_real]])
            th_b = np.concatenate([th_b, np.repeat(th_b[:1], B-n_real, axis=0)])
            N_b = np.concatenate([N_b, np.zeros(B-n_real, dtype=np.int64)])
            rho_b = np.concatenate([rho_b, np.repeat(rho_b[:1], B-n_real)])
        n_pad = pad_bucket(N_b.max())
        th_j, N_j, rho_j = jnp.asarray(th_b), jnp.asarray(N_b), jnp.asarray(rho_b)
        cut_j = jnp.full(B, cut)
        run = lambda capacity: _forward_batch(k_b, th_j, N_j, *consts, *scalars, rho_j, cut_j, bounds, n_pad,
                                              capacity=capacity, with_mask=return_mask)
        if prefilter_snr is None:
            out = run(None)
        else:
            out = jt._run_with_capacity(
                run, lambda: _forward_count_survivors(k_b, th_j, N_j, consts[0], consts[1], consts[2],
                                                      scalars[0], cut_j, bounds, n_pad),
                capacity_cache, (n_pad, B), n_pad)
        Nres_f.append(jt._from_jax(out[0], cupy_module)[:n_real])
        fg_f.append(jt._from_jax(out[1], cupy_module)[:n_real])
        if return_mask:
            mask.append(jt._from_jax(out[2], cupy_module)[:n_real])

    xp = out_module
    Nres_f = xp.moveaxis(xp.concatenate(Nres_f, axis=0).reshape(Nr, Np, -1), -1, 0).astype(xp.int64)
    fg_f = xp.moveaxis(xp.concatenate(fg_f, axis=0).reshape(Nr, Np, -1), -1, 0)
    Nres = xp.sum(Nres_f[1:, ...], axis=0)
    if Nr == 1 and Np == 1:
        Nres = Nres.squeeze()
        fg_f = fg_f.squeeze(axis=(1, 2))
    out = [Nres, fg_f]
    if return_mask:
        n_max = max(m.shape[1] for m in mask)
        mask = [xp.concatenate([m, xp.zeros((m.shape[0], n_max - m.shape[1]), dtype=bool)], axis=1) for m in mask]
        mask = xp.moveaxis(xp.concatenate(mask, axis=0).reshape(Nr, Np, n_max), -1, 0)
        if Nr == 1 and Np == 1:
            mask = mask.squeeze(axis=(1, 2))
        out.append(mask)
    if return_nres_f:
        out.append(Nres_f.squeeze(axis=(1, 2)) if Nr == 1 and Np == 1 else Nres_f)
    return tuple(out)


_sample_theta_jit = jax.jit(sample_theta, static_argnums=(2, 3))


def sample_galaxy_draw(key, g, theta, N, bounds, out_module=np):
    '''
    The (4, N) binaries (m_1, m_2, d_L, a) that jax_forward_model draws for galaxy g of the call
    with this key, materialized (for return_extras and tests).
    '''
    theta = jnp.asarray(_host(theta).astype(np.float64).reshape(6))
    draw = _sample_theta_jit(jax.random.fold_in(key, g), theta, int(N), bounds)
    cupy_module = out_module if out_module.__name__ == 'cupy' else None
    return jt._from_jax(draw, cupy_module)


# =============================================================================
# Many galaxies at fixed shapes (training sets)
# =============================================================================

## pre-filter capacity tiers as fractions of n_pad, ending with None (no pre-filter). Survivor
## fractions at a cut of 1 over 1000 hyperprior draws: median 0.049, 99th percentile 0.150, max 0.157
CAPACITY_FRACS = (0.16, 0.5, None)
## XLA memory per padded binary of _forward_batch(with_mask=False), with headroom (measured 57 and 75-81)
BYTES_PER_BINARY = {'filtered': 64, 'unfiltered': 96}
## padded binaries per batch when the device's free memory is unknown (e.g. on the CPU)
DEFAULT_MAX_BINARIES = 2.5e7
MEMORY_FRACTION = 0.8


def pad_buckets(Ns):
    '''pad_bucket for an array of sizes.'''
    Ns = np.maximum(np.asarray(Ns, dtype=np.int64), 1)
    table = [MIN_PAD]
    while table[-1] < Ns.max():
        table.append(math.ceil(MIN_PAD*PAD_GROWTH**len(table)))
    return np.asarray(table, dtype=np.int64)[np.searchsorted(table, Ns, side='left')]


def tier_capacity(frac, n_pad):
    '''Survivor capacity of a tier: frac*n_pad rounded up to a multiple of 1024, at most n_pad (None stays None).'''
    if frac is None:
        return None
    return int(min(n_pad, -(-math.ceil(frac*n_pad)//1024)*1024))


def device_free_bytes():
    '''Free memory of JAX's default device in bytes, or None if unknown (e.g. the CPU).'''
    dev = jax.devices()[0]
    if dev.platform != 'gpu':
        return None
    stats = dev.memory_stats()
    if stats and 'bytes_limit' in stats:
        return int(stats['bytes_limit'] - stats.get('bytes_in_use', 0))
    ## the platform allocator keeps no stats: ask NVML about the physical GPU behind the device
    try:
        import pynvml
        pynvml.nvmlInit()
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        ident = visible.split(',')[dev.local_hardware_id].strip() if visible else str(dev.local_hardware_id)
        handle = (pynvml.nvmlDeviceGetHandleByIndex(int(ident)) if ident.isdigit()
                  else pynvml.nvmlDeviceGetHandleByUUID(ident))
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    except Exception:
        return None


def batch_widths(Ns, max_binaries_per_batch=None, kernel='filtered'):
    '''
    Galaxies per batch for each padding bucket of Ns, {n_pad: width}: as many as fit in
    MEMORY_FRACTION of the device's free memory (or max_binaries_per_batch padded binaries), and
    at most the bucket's galaxy count rounded up to a power of 2.
    '''
    if max_binaries_per_batch is None:
        free = device_free_bytes()
        max_binaries_per_batch = (DEFAULT_MAX_BINARIES if free is None
                                  else MEMORY_FRACTION*free/BYTES_PER_BINARY[kernel])
    widths = {}
    for b, count in zip(*np.unique(pad_buckets(Ns), return_counts=True)):
        width = max(1, int(max_binaries_per_batch//b))
        widths[int(b)] = int(min(width, 1 << (int(count) - 1).bit_length()))
    return widths


@jax.jit
def _fold_keys(key, idx):
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(idx)


def forward_galaxies(key, idx, thetas, Ns, rhos, consts, bounds, widths=None, prefilter_snr=1.0,
                     capacity_fracs=CAPACITY_FRACS, max_binaries_per_batch=None, store=None, progress=False):
    '''
    Draw and threshold many galaxies at a few fixed shapes. Galaxies are grouped by padding
    bucket and run in batches of widths[n_pad], so each (n_pad, width, capacity) compiles once;
    batches are dispatched without waiting for the previous one. The first pass pre-filters with
    capacity tier_capacity(capacity_fracs[0], n_pad); galaxies whose survivors overflow it rerun at
    the next tier. Results don't depend on batching, padding or tier.

    Arguments
    -----------
    key : PRNG key; galaxy i's key is fold_in(key, idx[i]).
    idx (int array) : Per-galaxy key index, shape (G,).
    thetas (array) : Hyperparameters, (G, 6), GalacticBinaryPrior.pop_params order.
    Ns, rhos (array) : Binaries per galaxy and SNR thresholds, (G,).
    consts (tuple) : (edges, noisePSD, LISA_rx, duration, duration_eff), as in jax_forward_model.
    bounds (tuple) : prior_bounds.
    widths (dict) : {n_pad: galaxies per batch}. Default batch_widths(Ns, max_binaries_per_batch).
    prefilter_snr (float) : Per-bin pre-filter cut, at most every rho; None disables it.
    capacity_fracs (tuple) : Capacity tiers as fractions of n_pad, ending with None.
    max_binaries_per_batch (float) : Padded binaries per batch for the default widths; None sizes
        batches to the device's free memory.
    store : For resuming: get(n_pad, i, rows) -> (nres_f, fg, n_surv) or None, and
        put(n_pad, i, rows, nres_f, fg, n_surv), for the first pass's batches.
    progress (bool) : Print a line per bucket.

    Returns
    -----------
    nres_f (G, Nf) int and fg (G, Nf) per galaxy, bin 0 included, and a dict with the widths,
    compile count, simulated galaxies, overflow reruns and seconds.
    '''
    t0 = time.time()
    idx = np.asarray(idx, dtype=np.int64)
    Ns = np.asarray(Ns, dtype=np.int64)
    thetas = np.asarray(thetas, dtype=np.float64).reshape(-1, 6)
    rhos = np.broadcast_to(np.asarray(rhos, dtype=np.float64), Ns.shape)
    G = Ns.size
    if capacity_fracs[-1] is not None:
        raise ValueError("capacity_fracs must end with None, the unfiltered tier")
    if prefilter_snr is None:
        capacity_fracs = (None,)
    elif prefilter_snr > rhos.min():
        raise ValueError("prefilter_snr ({}) must not exceed snr_thresh ({}); binaries between the "
                         "two could be resolved".format(prefilter_snr, rhos.min()))
    edges, noisePSD, LISA_rx, duration, duration_eff = consts
    c_j = [jnp.asarray(_host(c_).astype(np.float64)) for c_ in (edges, noisePSD, LISA_rx)]
    scalars = (float(duration), float(duration_eff))
    cut = 0.0 if prefilter_snr is None else float(prefilter_snr)
    kernel = lambda frac: 'unfiltered' if frac is None else 'filtered'
    if widths is None:
        widths = batch_widths(Ns, max_binaries_per_batch, kernel(capacity_fracs[0]))

    Nf = c_j[0].shape[0]
    nres_f = np.zeros((G, Nf), dtype=np.int64)
    fg = np.zeros((G, Nf))
    n_surv = np.zeros(G, dtype=np.int64)
    n_pads = pad_buckets(Ns)
    n_compiled = _forward_batch._cache_size()
    info = dict(widths=dict(widths), simulated=0, overflow=0)

    def groups(rows, widths):
        '''(n_pad, width, [row batches]) per bucket of rows, in increasing n_pad.'''
        rows = rows[np.argsort(n_pads[rows], kind='stable')]
        b, start = np.unique(n_pads[rows], return_index=True)
        out = []
        for n_pad, lo, hi in zip(b, start, list(start[1:]) + [rows.size]):
            W = widths[int(n_pad)]
            out.append((int(n_pad), W, [rows[i:i+W] for i in range(lo, hi, W)]))
        return out

    def dispatch(rows, n_pad, W, capacity):
        pad = W - rows.size
        r = np.concatenate([rows, np.repeat(rows[:1], pad)])
        N_b = np.concatenate([Ns[rows], np.zeros(pad, dtype=np.int64)])
        return _forward_batch(_fold_keys(key, jnp.asarray(idx[r])), jnp.asarray(thetas[r]), jnp.asarray(N_b),
                              *c_j, *scalars, jnp.asarray(rhos[r]), jnp.full(W, cut), bounds, n_pad,
                              capacity=capacity, with_mask=False)

    def run(rows, tier, widths, use_store):
        '''Run rows at a capacity tier; returns the rows whose survivors overflowed it.'''
        pending, over = deque(), []
        bucket_groups = groups(rows, widths)
        total = sum(n_pad*len(batches) for n_pad, _, batches in bucket_groups)
        done = 0

        def finish():
            n_pad, i, rows, out, capacity = pending.popleft()
            k = rows.size
            res = tuple(np.asarray(o)[:k] for o in out)
            nres_f[rows], fg[rows], n_surv[rows] = res
            if use_store and store is not None:
                store.put(n_pad, i, rows, *res)
            if capacity is not None:
                over.append(rows[res[2] > capacity])

        for j, (n_pad, W, batches) in enumerate(bucket_groups):
            capacity = tier_capacity(capacity_fracs[tier], n_pad)
            for i, rows in enumerate(batches):
                got = store.get(n_pad, i, rows) if use_store and store is not None else None
                if got is not None:
                    nres_f[rows], fg[rows], n_surv[rows] = got
                    if capacity is not None:
                        over.append(rows[got[2] > capacity])
                    continue
                pending.append((n_pad, i, rows, dispatch(rows, n_pad, W, capacity), capacity))
                info['simulated'] += rows.size if tier == 0 else 0
                done += n_pad
                ## keep one batch in flight while the previous one's results come back
                if len(pending) > 1:
                    finish()
            if progress:
                el = time.time() - t0
                print("tier {} bucket {}/{} (n_pad {}): {} batches of {}; {:.0f} s elapsed, ~{:.0f} s left".format(
                    tier, j + 1, len(bucket_groups), n_pad, len(batches), W, el,
                    el/done*(total - done) if done else float('nan')), flush=True)
        while pending:
            finish()
        return np.concatenate(over) if over else np.zeros(0, dtype=np.int64)

    rerun = run(np.arange(G), 0, widths, True)
    for tier in range(1, len(capacity_fracs)):
        if rerun.size == 0:
            break
        info['overflow'] += int(rerun.size)
        rerun = run(rerun, tier, batch_widths(Ns[rerun], max_binaries_per_batch, kernel(capacity_fracs[tier])),
                    False)
    info.update(compiles=_forward_batch._cache_size() - n_compiled, seconds=time.time() - t0)
    return nres_f, fg, info
