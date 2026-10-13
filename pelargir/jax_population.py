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

forward_galaxies runs many galaxies (training sets) without padding them to n_pad: each galaxy is
drawn in fixed-size chunks, pre-filtered per bin chunk by chunk, and only its survivors are sorted
and thresholded; batch widths are sized to the GPU's memory.
"""
from collections import deque
from functools import partial
from typing import NamedTuple
import logging
import math
import os
import queue
import threading
import time

from . import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax import lax

import numpy as np
from astropy import units as u

from . import jax_thresholding as jt

logger = logging.getLogger('pelargir')

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

## binaries drawn per chunk of the fused training-set kernel; a galaxy's draw depends on it.
## 2^18 measured fastest of 2^18-2^20 (9-10 ns per binary on an RTX 4070) and wastes least on small galaxies
DRAW_CHUNK = 2**18
## growth of the chunk-count buckets (padding at most this factor plus one chunk)
CHUNK_GROWTH = 1.1
## pre-filter capacity tiers as fractions of the drawn size, ending with None (no pre-filter).
## Survivor fractions at a cut of 1 over 1000 hyperprior draws: median 0.049, 99th percentile 0.150, max 0.157
CAPACITY_FRACS = (0.16, 0.5, None)
## XLA memory of _forward_chunked per drawn-chunk binary and per survivor-buffer entry: at least 1.17x
## the compiled memory analyses on an RTX 4070 for 1-230 chunks at every tier (marginal ~25 B per
## survivor, a floor of up to ~33 MB per galaxy). Only a first guess: fit_widths checks every
## batch shape against XLA's own figure for the device.
BYTES_PER_DRAWN = 60
BYTES_PER_SURVIVOR = 29
## drawn binaries per batch when the device's free memory is unknown (e.g. on the CPU)
DEFAULT_MAX_BINARIES = 2.5e7
MEMORY_FRACTION = 0.8


def pad_buckets(Ns):
    '''pad_bucket for an array of sizes.'''
    Ns = np.maximum(np.asarray(Ns, dtype=np.int64), 1)
    table = [MIN_PAD]
    while table[-1] < Ns.max():
        table.append(math.ceil(MIN_PAD*PAD_GROWTH**len(table)))
    return np.asarray(table, dtype=np.int64)[np.searchsorted(table, Ns, side='left')]


def chunk_buckets(Ns, K=DRAW_CHUNK, growth=CHUNK_GROWTH):
    '''Chunks drawn per galaxy: ceil(N/K) (at least 1) rounded up to a geometric bucket of the given growth.'''
    n = np.maximum(-(-np.asarray(Ns, dtype=np.int64)//K), 1)
    table = [1]
    while table[-1] < n.max():
        table.append(max(table[-1] + 1, math.ceil(table[-1]*growth)))
    return np.asarray(table, dtype=np.int64)[np.searchsorted(table, n, side='left')]


def tier_capacity(frac, n_drawn):
    '''Survivor capacity of a tier: frac*n_drawn rounded up to a multiple of 1024, at most n_drawn (None stays None).'''
    if frac is None:
        return None
    return int(min(n_drawn, -(-math.ceil(frac*n_drawn)//1024)*1024))


def galaxy_bytes(n_chunks, K, capacity):
    '''Device memory of one galaxy in _forward_chunked (the survivor buffer is n_chunks*K without a pre-filter).'''
    M = n_chunks*K if capacity is None else capacity
    return BYTES_PER_DRAWN*K + BYTES_PER_SURVIVOR*(M + K)


def device_free_bytes(dev=None):
    '''Free memory of a device (default: JAX's default device) in bytes, or None if unknown (e.g. the CPU).'''
    dev = jax.devices()[0] if dev is None else dev
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


def resolve_devices(devices=None):
    '''Devices to run on: all local devices of JAX's default platform (None), the first n (an int), or a list.'''
    local = jax.local_devices()
    if devices is None:
        return local
    if isinstance(devices, int):
        if not 1 <= devices <= len(local):
            raise ValueError("asked for {} devices; {} are available: {}".format(devices, len(local), local))
        return local[:devices]
    return list(devices)


def device_budget(devices=None):
    '''MEMORY_FRACTION of the smallest free memory among devices (default: JAX's default device) in bytes, or None if unknown.'''
    frees = [device_free_bytes(d) for d in (devices or [None])]
    return None if any(f is None for f in frees) else int(MEMORY_FRACTION*min(frees))


def batch_widths(Ns, max_binaries_per_batch=None, capacity_frac=CAPACITY_FRACS[0], K=DRAW_CHUNK, devices=None):
    '''
    Galaxies per batch for each chunk-count bucket of Ns, {n_chunks: width}: as many as galaxy_bytes
    fits in device_budget(devices), or, if given, max_binaries_per_batch drawn binaries; and at most
    the bucket's galaxy count rounded up to a power of 2. A first guess; see fit_widths.
    '''
    budget = device_budget(devices) if max_binaries_per_batch is None else None
    widths = {}
    for nc, count in zip(*np.unique(chunk_buckets(Ns, K), return_counts=True)):
        nc = int(nc)
        if max_binaries_per_batch is not None:
            width = int(max_binaries_per_batch//(nc*K))
        elif budget is None:
            width = int(DEFAULT_MAX_BINARIES//(nc*K))
        else:
            width = int(budget//galaxy_bytes(nc, K, tier_capacity(capacity_frac, nc*K)))
        widths[nc] = int(min(max(width, 1), 1 << (int(count) - 1).bit_length()))
    return widths


def chunk_obs(key, theta, N, c, K, bounds):
    '''(f, A) of binaries c*K ... c*K + K - 1 of a galaxy of N; those at index >= N are out of band.'''
    A, f = amp_freq(sample_theta(jax.random.fold_in(key, c), theta, K, bounds))
    valid = c*K + jnp.arange(K) < N
    return jnp.where(valid, f, jnp.inf), jnp.where(valid, A, 0.0)


def sample_galaxy_draw_chunked(key, g, theta, N, K=DRAW_CHUNK, bounds=None, out_module=np):
    '''
    The (4, N) binaries (m_1, m_2, d_L, a) that forward_galaxies draws for galaxy g (key index)
    with this key, materialized (for tests).
    '''
    theta = jnp.asarray(_host(theta).astype(np.float64).reshape(6))
    kg = jax.random.fold_in(key, g)
    n_chunks = max(1, -(-int(N)//K))
    draw = jnp.concatenate([_sample_theta_jit(jax.random.fold_in(kg, c), theta, K, bounds)
                            for c in range(n_chunks)], axis=1)[:, :int(N)]
    cupy_module = out_module if out_module.__name__ == 'cupy' else None
    return jt._from_jax(draw, cupy_module)


@partial(jax.jit, static_argnames=('bounds', 'n_chunks', 'K', 'capacity'))
def _forward_chunked(keys, thetas, Ns, rhos, cuts, edges, noisePSD, LISA_rx, duration, duration_eff,
                     bounds, n_chunks, K, capacity=None):
    '''
    Per-galaxy (Nres_f, fg_f, n_surv) for a batch: keys (B,), thetas (B, 6), Ns, rhos and pre-filter
    cuts (B,). Each galaxy's binaries are drawn n_chunks chunks of K at a time; those failing the
    per-bin cut add their power to D (foreground and confusion noise), the rest are compacted into
    a survivor buffer of size capacity, which is sorted and thresholded. Results are only valid
    when n_surv <= capacity. capacity None keeps every in-band binary (no pre-filter).
    '''
    Nf = edges.shape[0]
    M = n_chunks*K if capacity is None else capacity

    def one(key, theta, N, rho, cut):
        def body(c, carry):
            D, buf_k, buf_a, n = carry
            k, a = jt._prepare(*chunk_obs(key, theta, N, c, K, bounds), edges, LISA_rx)
            if capacity is None:
                keep = k < Nf
            else:
                keep = jt._survives(k, a, noisePSD, duration, cut)
                D = D + jax.ops.segment_sum(jnp.where(keep, 0.0, a**2), k, num_segments=Nf+1)[:Nf]
            ## append this chunk's survivors at the running count; an overflowing galaxy is discarded
            pos = jnp.nonzero(keep, size=K, fill_value=K)[0]
            start = jnp.minimum(n, M)
            buf_k = lax.dynamic_update_slice(buf_k, k.at[pos].get(mode='fill', fill_value=Nf), (start,))
            buf_a = lax.dynamic_update_slice(buf_a, a.at[pos].get(mode='fill', fill_value=0.0), (start,))
            return D, buf_k, buf_a, n + jnp.sum(keep, dtype=jnp.int32)

        init = (jnp.zeros(Nf), jnp.full(M + K, Nf, dtype=jnp.int32), jnp.zeros(M + K), jnp.int32(0))
        D, buf_k, buf_a, n = lax.fori_loop(0, n_chunks, body, init)
        k_s, a_s = lax.sort((buf_k[:M], buf_a[:M]), num_keys=2)
        Nres_f, fg_f, _ = jt._threshold_core(k_s, a_s, jnp.arange(M, dtype=jnp.int32),
                                             None if capacity is None else D, noisePSD, duration, duration_eff,
                                             rho, M)
        return Nres_f, fg_f, n
    return jax.vmap(one)(keys, thetas, Ns, rhos, cuts)


## compiled _forward_chunked executables of training-set runs, by shape and device
_executables = {}
_executables_lock = threading.Lock()


def _compile(args, bounds, n_chunks, K, capacity, device=None):
    '''
    _forward_chunked compiled ahead of time for args (its ten arguments, arrays on device or
    jax.ShapeDtypeStruct specs of them), cached by shape and device; returns (executable, seconds
    spent compiling, 0 if cached).
    '''
    shape = (str(device), bounds, n_chunks, K, capacity) + tuple((np.shape(a), str(getattr(a, 'dtype', type(a))))
                                                                 for a in args)
    with _executables_lock:
        exe = _executables.get(shape)
    seconds = 0.0
    if exe is None:
        t = time.time()
        exe = _forward_chunked.lower(*args, bounds=bounds, n_chunks=n_chunks, K=K, capacity=capacity).compile()
        seconds = time.time() - t
        with _executables_lock:
            _executables[shape] = exe
    return exe, seconds


def _launch(args, bounds, n_chunks, K, capacity, device=None):
    '''
    _forward_chunked on args (its ten array arguments, placed on device), compiled on a shape's first
    use on that device, so the compile is timed on its own; returns (outputs, seconds spent compiling).
    '''
    exe, seconds = _compile(args, bounds, n_chunks, K, capacity, device)
    return exe(*args), seconds


def compiled_bytes(exe):
    '''Device memory of a compiled executable from XLA's memory analysis (temporaries, arguments, outputs), or None.'''
    try:
        m = exe.memory_analysis()
    except Exception:
        return None
    if m is None:
        return None
    return int(m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes)


def fit_widths(widths, capacity_frac, key, consts, bounds, budget, device, K=DRAW_CHUNK, progress=None):
    '''
    widths ({n_chunks: galaxies per batch}) shrunk until each bucket's _forward_chunked, compiled for
    device, needs at most budget bytes by XLA's memory analysis (memory is linear in the width).
    The compiled executables are cached for the run. Returns (widths, [per-bucket records]).

    consts : the five forward-model constants (three arrays on device and two floats).
    progress : a _Progress; the compiles are recorded as serial (the devices are idle).
    '''
    spec = lambda shape, dtype: jax.ShapeDtypeStruct(shape, dtype, sharding=jax.sharding.SingleDeviceSharding(device))
    f64 = jnp.float64
    out, records = {}, []
    for nc, W in sorted(widths.items(), reverse=True):
        capacity = tier_capacity(capacity_frac, nc*K)
        W0 = W
        while True:
            keys = jax.eval_shape(_fold_keys, key, jax.ShapeDtypeStruct((W,), jnp.int64))
            args = (spec(keys.shape, keys.dtype), spec((W, 6), f64), spec((W,), jnp.int64), spec((W,), f64),
                    spec((W,), f64), *consts)
            exe, seconds = _compile(args, bounds, nc, K, capacity, device)
            if seconds and progress is not None:
                progress.compiled(seconds, serial=True)
            need = compiled_bytes(exe)
            if need is None or need <= budget or W == 1:
                break
            W_new = max(1, min(W - 1, int(W*budget/need)))
            logger.info("bucket of %d chunks: width %d -> %d (compiled %.2f GiB > budget %.2f GiB)", nc, W, W_new,
                        need/2**30, budget/2**30)
            W = W_new
        if need is not None and need > budget:
            logger.warning("bucket of %d chunks needs %.2f GiB for one galaxy, over the budget of %.2f GiB", nc,
                           need/2**30, budget/2**30)
        out[nc] = W
        records.append(dict(n_chunks=nc, width_guess=W0, width=W, compiled_bytes=need, budget=budget))
    return out, records


@jax.jit
def _fold_keys(key, idx):
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(idx)


class _Progress:
    '''
    ETA bookkeeping for forward_galaxies, on wall time: compute time per binary (elapsed time less
    compile time, which _launch and fit_widths measure on their own; the workers' compiles on
    n_devices run side by side, fit_widths' run alone) plus a compile time for each shape still to
    run (the mean so far, or DEFAULT_COMPILE).
    '''
    DEFAULT_COMPILE = 3.0

    def __init__(self, n_devices=1, clock=time.time):
        self.clock, self.n_devices = clock, n_devices
        self.t0 = clock()
        self.total = self.done = self.compile_time = self.serial_compile_time = 0.0
        self.shapes_left = self.n_compiled = 0

    def add(self, binaries, shapes):
        '''Work about to start (a tier): its binaries and its distinct batch shapes.'''
        self.total += float(binaries)
        self.shapes_left += int(shapes)

    def compiled(self, seconds, serial=False):
        '''A shape took this long to compile (or load) on one device; serial if no other device was working.'''
        self.compile_time += seconds
        self.n_compiled += 1
        if serial:
            self.serial_compile_time += seconds

    def finished(self, binaries):
        self.done += float(binaries)

    def shapes_done(self, shapes):
        self.shapes_left = max(self.shapes_left - int(shapes), 0)

    def remaining(self):
        '''Seconds left: binaries left at the measured compute rate, plus the shapes still to compile.'''
        per_shape = self.compile_time/self.n_compiled if self.n_compiled else self.DEFAULT_COMPILE
        side_by_side = self.compile_time - self.serial_compile_time
        compute = max(self.clock() - self.t0 - side_by_side/self.n_devices - self.serial_compile_time, 0.0)
        per_binary = compute/self.done if self.done else float('nan')
        return max(self.total - self.done, 0.0)*per_binary + self.shapes_left*per_shape


## next_tier of a galaxy whose result is final
DONE = -1


class GalaxyState:
    '''
    Per-galaxy results of forward_galaxies and the capacity tier each galaxy runs at next (DONE
    once its result is final), in memory. flow_data._RunState keeps the same arrays on disk.
    '''

    def __init__(self, G, Nf):
        self.nres_f = np.zeros((G, Nf), dtype=np.int32)
        self.fg = np.zeros((G, Nf))
        self.n_surv = np.zeros(G, dtype=np.int64)
        self.next_tier = np.zeros(G, dtype=np.int8)

    def commit(self, rows, nres_f, fg, n_surv, next_tier):
        self.nres_f[rows], self.fg[rows], self.n_surv[rows] = nres_f, fg, n_surv
        self.next_tier[rows] = next_tier


def forward_galaxies(key, idx, thetas, Ns, rhos, consts, bounds, widths=None, prefilter_snr=1.0,
                     capacity_fracs=CAPACITY_FRACS, max_binaries_per_batch=None, state=None, progress=False,
                     K=DRAW_CHUNK, devices=None):
    '''
    Draw and threshold many galaxies at a few fixed shapes, with _forward_chunked, on one or more
    devices. Galaxies are grouped by chunk-count bucket (largest first) into batches of
    widths[n_chunks], so each (n_chunks, width, capacity) compiles once per device. One worker
    thread per device takes the next batch from a shared queue, so the devices run side by side
    whatever the allocator. Tier t pre-filters into a survivor buffer of
    tier_capacity(capacity_fracs[t], n_chunks*K); galaxies whose survivors overflow it move on to
    tier t + 1. Where the devices' free memory is known, every bucket's width (however chosen) is
    narrowed by fit_widths until its compiled kernel fits device_budget. Results don't depend on
    batching, padding, tier or device.

    Arguments
    -----------
    key : PRNG key; galaxy i's key is fold_in(key, idx[i]), and its chunk c of K binaries is drawn
        from fold_in(that, c).
    idx (int array) : Per-galaxy key index, shape (G,).
    thetas (array) : Hyperparameters, (G, 6), GalacticBinaryPrior.pop_params order.
    Ns, rhos (array) : Binaries per galaxy and SNR thresholds, (G,).
    consts (tuple) : (edges, noisePSD, LISA_rx, duration, duration_eff), as in jax_forward_model.
    bounds (tuple) : prior_bounds.
    widths (dict) : {n_chunks: galaxies per batch} for tier 0. Default batch_widths of the galaxies to run.
    prefilter_snr (float) : Per-bin pre-filter cut, at most every rho; None disables it.
    capacity_fracs (tuple) : Survivor-buffer tiers as fractions of the drawn size, ending with None.
    max_binaries_per_batch (float) : Drawn binaries per batch for the default widths (still capped by
        fit_widths); None sizes batches to the smallest device's free memory.
    state : A GalaxyState (default: a new one), or anything with its arrays and commit, e.g. one
        on disk. Galaxies with next_tier == DONE are skipped, and every finished batch is committed,
        so a state from an interrupted call resumes it (on any devices).
    progress (bool) : Print a line per bucket.
    K (int) : Binaries per drawn chunk.
    devices : Devices to run on (see resolve_devices): None for all local devices, a count, or a list.

    Returns
    -----------
    nres_f (G, Nf) int and fg (G, Nf) per galaxy (the state's arrays), bin 0 included, and a dict
    with the tier-0 widths, the width checks (fit_widths records), devices and batches per device,
    compile count, galaxies simulated and already done, overflow reruns, per-bucket timings and seconds.
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
    devices = resolve_devices(devices)
    edges, noisePSD, LISA_rx, duration, duration_eff = consts
    c_host = [_host(c_).astype(np.float64) for c_ in (edges, noisePSD, LISA_rx)]
    c_dev = {d: [jax.device_put(c_, d) for c_ in c_host] for d in devices}
    scalars = (float(duration), float(duration_eff))
    cut = 0.0 if prefilter_snr is None else float(prefilter_snr)
    Nf = c_host[0].shape[0]
    state = GalaxyState(G, Nf) if state is None else state
    n_chunks_g = chunk_buckets(Ns, K)
    n_compiled = len(_executables)
    todo = state.next_tier != DONE
    info = dict(widths={}, devices=[str(d) for d in devices], device_batches={str(d): 0 for d in devices},
                simulated=0, resumed=int(G - todo.sum()), overflow=0, buckets=[], width_checks=[])
    prog = _Progress(len(devices))
    lock = threading.Lock()

    def groups(rows, widths):
        '''(n_chunks, width, [row batches]) per bucket of rows, largest bucket first.'''
        rows = rows[np.argsort(-n_chunks_g[rows], kind='stable')]
        b, start = np.unique(-n_chunks_g[rows], return_index=True)
        out = []
        for nc, lo, hi in zip(-b, start, list(start[1:]) + [rows.size]):
            W = widths[int(nc)]
            out.append((int(nc), W, [rows[i:min(i + W, hi)] for i in range(lo, hi, W)]))
        return out

    def run_batch(dev, rows, nc, W, capacity):
        '''One batch on dev: (nres_f, fg, n_surv) for its rows, and the seconds spent compiling.'''
        ## a short last batch is padded only to the next power of 2 (N = 0 galaxies, dropped)
        W = min(W, 1 << (rows.size - 1).bit_length())
        pad = W - rows.size
        r = np.concatenate([rows, np.repeat(rows[:1], pad)])
        N_b = np.concatenate([Ns[rows], np.zeros(pad, dtype=np.int64)])
        put = lambda x: jax.device_put(x, dev)
        args = (_fold_keys(put(key), put(idx[r])), put(thetas[r]), put(N_b), put(rhos[r]), put(np.full(W, cut)),
                *c_dev[dev], *scalars)
        out, t_compile = _launch(args, bounds=bounds, n_chunks=nc, K=K, capacity=capacity, device=dev)
        return tuple(np.asarray(o)[:rows.size] for o in out), t_compile

    for tier in range(len(capacity_fracs)):
        rows_t = np.flatnonzero(state.next_tier == tier)
        if rows_t.size == 0:
            continue
        w = widths if tier == 0 and widths is not None else batch_widths(
            Ns[rows_t], max_binaries_per_batch, capacity_fracs[tier], K, devices)
        ## whatever chose the widths, no batch may need more than the device budget by XLA's own count
        budget = device_budget(devices)
        precompiled = set()
        if budget is not None:
            t_fit = time.time()
            w, checks = fit_widths(w, capacity_fracs[tier], key, c_dev[devices[0]] + list(scalars), bounds, budget,
                                   devices[0], K, prog)
            info['width_checks'] += [dict(c, tier=tier) for c in checks]
            precompiled = {(c['n_chunks'], c['width']) for c in checks}
            logger.info("tier %d: checked %d batch shapes against a budget of %.1f GiB in %.1f s; %d narrowed",
                        tier, len(checks), budget/2**30, time.time() - t_fit,
                        sum(c['width'] < c['width_guess'] for c in checks))
        bucket_groups = groups(rows_t, w)
        ## a bucket's shapes still to compile: the widths its batches dispatch at (a short last batch runs
        ## narrower), less the one fit_widths compiled
        shapes = [len({min(W, 1 << (b.size - 1).bit_length()) for b in batches} - {W if (nc, W) in precompiled else -1})
                  for nc, W, batches in bucket_groups]
        prog.add(Ns[rows_t].sum(), sum(shapes))
        logger.info("tier %d: %d galaxies in %d buckets to run on %d device(s)", tier, rows_t.size,
                    len(bucket_groups), len(devices))
        if tier == 0:
            info['widths'] = dict(w)
        else:
            info['overflow'] += int(rows_t.size)
        tasks = queue.Queue()
        buckets = []
        for j, (nc, W, batches) in enumerate(bucket_groups):
            capacity = tier_capacity(capacity_fracs[tier], nc*K)
            buckets.append(dict(left=len(batches), binaries=0, gpu_seconds=0.0, compile=0.0, t_first=None))
            for rows in batches:
                tasks.put((j, nc, W, capacity, rows))
        stop, errors = threading.Event(), []

        def worker(dev):
            while not stop.is_set():
                try:
                    j, nc, W, capacity, rows = tasks.get_nowait()
                except queue.Empty:
                    return
                b = buckets[j]
                with lock:
                    b['t_first'] = time.time() if b['t_first'] is None else b['t_first']
                t = time.time()
                try:
                    res, t_compile = run_batch(dev, rows, nc, W, capacity)
                except BaseException as err:
                    errors.append(err)
                    stop.set()
                    return
                sec = time.time() - t
                over = (res[2] > capacity) if capacity is not None else np.zeros(rows.size, dtype=bool)
                with lock:
                    state.commit(rows, *res, np.where(over, tier + 1, DONE).astype(np.int8))
                    n_b = int(Ns[rows].sum())
                    info['device_batches'][str(dev)] += 1
                    info['simulated'] += rows.size if tier == 0 else 0
                    b['left'] -= 1
                    b['binaries'] += n_b
                    b['gpu_seconds'] += sec - t_compile
                    b['compile'] += t_compile
                    if t_compile:
                        prog.compiled(t_compile)
                    prog.finished(n_b)
                    if b['left'] == 0:
                        prog.shapes_done(shapes[j])
                        bucket_report(tier, j, b)

        def bucket_report(tier, j, b):
            nc, W, batches = bucket_groups[j]
            wall = time.time() - b['t_first']
            ns = 1e9*b['gpu_seconds']/b['binaries'] if b['binaries'] else float('nan')
            info['buckets'].append(dict(tier=tier, n_chunks=nc, width=W, batches=len(batches), binaries=b['binaries'],
                                        seconds=wall, compile_seconds=b['compile'], ns_per_binary=ns,
                                        remaining=prog.remaining()))
            if progress:
                print("tier {} bucket {}/{} ({} chunks of {}): {} batches of {} in {:.1f} s ({:.1f} s compiling), "
                      "{:.1f} ns/binary; {:.0f} s elapsed, ~{:.0f} s left".format(
                          tier, j + 1, len(bucket_groups), nc, K, len(batches), W, wall, b['compile'], ns,
                          time.time() - t0, prog.remaining()), flush=True)

        threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devices]
        for th in threads:
            th.start()
        try:
            for th in threads:
                th.join()
        except BaseException:
            stop.set()
            raise
        if errors:
            raise errors[0]
    info.update(compiles=len(_executables) - n_compiled, seconds=time.time() - t0, draw_chunk=K)
    return state.nres_f, state.fg, info
