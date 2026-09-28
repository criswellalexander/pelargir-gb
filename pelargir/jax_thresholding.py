"""
Fixed-shape JAX implementation of the SNR thresholding in thresholding.SNR_Threshold.

Instead of padding each frequency bin to its most-populated count, all binaries of a
galaxy are sorted once by (bin, amplitude), so each bin is a contiguous segment. The
per-bin confusion cumsum is a segmented scan, the Eq. 17 boundary (arXiv:2604.03390) is
a segment max, and N_res and the foreground are segment sums. Every array has a static
shape, so the whole computation jits. Results match serial_array_sort/block_array_sort
on identical inputs (tests/test_jax_thresholding.py).

Per-bin pre-filter (prefilter_snr): a binary whose SNR against the bare
noise in its own bin, sqrt(duration*a^2/noisePSD[k]), is below prefilter_snr can never
be resolved (the confusion term only lowers its SNR), and since that bound grows with a,
the dropped binaries are always the lowest-amplitude prefix of their bin. Their summed
power D_k is folded into every survivor's confusion prefix and into the foreground (as
SNR_Threshold's extra_confusion_psd does). The survivors are compacted into a
static-capacity buffer of M entries before sorting, so only they go through the sort,
scan and segment reductions. Exact for any prefilter_snr <= snr_thresh.
The capacity M is a power-of-2 bucket chosen on the host; the kernels return each
galaxy's survivor count, and a batch that overflows M is rerun with a larger bucket.
"""
from functools import partial

import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax import lax

import numpy as np

MIN_CAPACITY = 1024


def _prepare(f, A, edges, LISA_rx):
    '''Bin index k (int32; k == Nf is out of band) and response-weighted amplitude a (0 out of band).'''
    Nf = edges.shape[0]
    ## same convention as SNR_Threshold.coarsegrain_bin
    k = jnp.digitize(f, edges).astype(jnp.int32)
    a = jnp.where(k < Nf, A*jnp.sqrt(LISA_rx[jnp.minimum(k, Nf-1)]), 0.0)
    return k, a


def _survives(k, a, noisePSD, duration, cut):
    '''
    True unless sqrt(duration*a^2/noisePSD[k]) < cut. Written like Nij in _threshold_core with
    zero confusion, so (IEEE rounding being monotone) this bound is never below Nij.
    '''
    Nf = noisePSD.shape[0]
    return (k < Nf) & (jnp.sqrt(duration*a**2/noisePSD[jnp.minimum(k, Nf-1)]) >= cut)


def _threshold_core(k_s, a_s, order_s, D, noisePSD, duration, duration_eff, snr_thresh, N):
    '''
    Thresholds M binaries sorted by (bin, amplitude). Entries with k_s == Nf (out of band,
    or capacity filler with a_s = 0) are never resolved and contribute nothing.

    Arguments
    -----------
    k_s, a_s, order_s (array) : Sorted bin indices, amplitudes, and positions in the
        galaxy's input order (N marks filler), shape (M,).
    D (array or None) : Per-bin power of pre-filtered binaries, shape (Nf,); None for no pre-filter.
    N (int) : Galaxy size, for the per-binary mask.
    Other arguments as in _threshold_one.
    '''
    Nf = noisePSD.shape[0]
    M = k_s.shape[0]
    pos = jnp.arange(M, dtype=jnp.int32)
    a2 = a_s**2
    k_in = jnp.minimum(k_s, Nf-1)

    ## per-bin cumsum of a^2 (restarting at each bin), i.e. calc_Nij's cumsum(A**2)
    seg_start = jnp.concatenate([jnp.ones(1, dtype=bool), k_s[1:] != k_s[:-1]])
    seg_add = lambda x, y: (jnp.where(y[1], y[0], x[0] + y[0]), x[1] | y[1])
    cum_a2, _ = lax.associative_scan(seg_add, (a2, seg_start))
    ## pre-filtered power joins the noise floor as in serial_array_sort's extra_confusion_psd
    noise = noisePSD[k_in] if D is None else noisePSD[k_in] + duration_eff*D[k_in]
    Nij = jnp.sqrt(duration*a2/(noise + duration_eff*(cum_a2 - a2)))

    ## Eq. 17: sources above the loudest sub-threshold source in their bin are resolved.
    ## Pre-filtered sources sit below every survivor of their bin and are all sub-threshold,
    ## so leaving them out cannot move this boundary past a survivor.
    last_sub = jax.ops.segment_max(jnp.where(Nij >= snr_thresh, -1, pos), k_s,
                                   num_segments=Nf+1, indices_are_sorted=True)
    resolved_s = pos > last_sub[k_s]

    Nres_f = jax.ops.segment_sum(resolved_s.astype(jnp.int32), k_s,
                                 num_segments=Nf+1, indices_are_sorted=True)[:Nf]
    fg_f = jax.ops.segment_sum(jnp.where(resolved_s, 0.0, a2), k_s,
                               num_segments=Nf+1, indices_are_sorted=True)[:Nf]
    if D is not None:
        fg_f = fg_f + D
    resolved = jnp.zeros(N, dtype=bool).at[order_s].set(resolved_s, mode='drop')
    return Nres_f, fg_f, resolved


def _threshold_one(f, A, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh):
    '''
    Threshold one galaxy, all N binaries (no pre-filter).

    Arguments
    -----------
    f, A (array)            : GW frequencies and response-free amplitudes, shape (N,).
    edges (array)           : Upper bin edges, fs + 0.5*delf, shape (Nf,).
    noisePSD, LISA_rx (array) : Per-bin noise PSD and LISA response, shape (Nf,).
    duration, duration_eff (float) : As in SNR_Threshold.
    snr_thresh (float)      : Resolvability threshold.

    Returns
    -----------
    Nres_f (int array) : Resolved count per bin, shape (Nf,), bin 0 included.
    fg_f (array)       : Unresolved response-weighted power per bin, shape (Nf,).
    resolved (bool array) : Per-binary resolved flag in input order, shape (N,).
    '''
    N = f.shape[0]
    k, a = _prepare(f, A, edges, LISA_rx)
    k_s, a_s, order = lax.sort((k, a, jnp.arange(N, dtype=jnp.int32)), num_keys=2)
    return _threshold_core(k_s, a_s, order, None, noisePSD, duration, duration_eff, snr_thresh, N)


def _threshold_one_filtered(f, A, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh,
                            cut, capacity):
    '''
    As _threshold_one, but only the (at most capacity) binaries passing the per-bin cut are
    gathered, sorted and thresholded. Also returns the survivor count; results are only
    valid when it is <= capacity.
    '''
    N = f.shape[0]
    Nf = edges.shape[0]
    k, a = _prepare(f, A, edges, LISA_rx)
    survive = _survives(k, a, noisePSD, duration, cut)
    D = jax.ops.segment_sum(jnp.where(survive, 0.0, a**2), k, num_segments=Nf+1)[:Nf]

    ## survivors' positions, padded with N; padding reads as out of band with no power
    idx = jnp.nonzero(survive, size=capacity, fill_value=N)[0].astype(jnp.int32)
    k_c = k.at[idx].get(mode='fill', fill_value=Nf)
    a_c = a.at[idx].get(mode='fill', fill_value=0.0)
    k_s, a_s, order = lax.sort((k_c, a_c, idx), num_keys=2)
    out = _threshold_core(k_s, a_s, order, D, noisePSD, duration, duration_eff, snr_thresh, N)
    return (*out, jnp.sum(survive, dtype=jnp.int32))


@partial(jax.jit, static_argnames=('capacity',))
def _threshold_batch(fA, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh, cut,
                     capacity=None):
    '''
    fA has shape (B,2,N): a batch of galaxies, each [frequencies, amplitudes]. Returns
    per-galaxy (Nres_f, fg_f, resolved, n_surv). capacity None means no pre-filter (cut is
    ignored and n_surv is N).
    '''
    shared = (edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh)
    if capacity is None:
        out = jax.vmap(_threshold_one, in_axes=(0, 0) + (None,)*6)(fA[:, 0], fA[:, 1], *shared)
        return (*out, jnp.full(fA.shape[0], fA.shape[2], dtype=jnp.int32))
    kernel = lambda f, A: _threshold_one_filtered(f, A, *shared, cut, capacity)
    return jax.vmap(kernel)(fA[:, 0], fA[:, 1])


@jax.jit
def _count_survivors(fA, edges, noisePSD, LISA_rx, duration, cut):
    '''Per-galaxy count of binaries passing the per-bin cut, shape (B,).'''
    def count(f, A):
        k, a = _prepare(f, A, edges, LISA_rx)
        return jnp.sum(_survives(k, a, noisePSD, duration, cut), dtype=jnp.int32)
    return jax.vmap(count)(fA[:, 0], fA[:, 1])


def _bucket(n, N):
    '''Capacity for n survivors: the next power of 2 >= max(n, MIN_CAPACITY), at most N.'''
    return int(min(N, max(MIN_CAPACITY, 1 << max(int(n) - 1, 0).bit_length())))


def _is_cupy(arr):
    return type(arr).__module__.split('.')[0] == 'cupy'


def _to_jax(arr):
    if _is_cupy(arr):
        return jax.dlpack.from_dlpack(arr)
    return jnp.asarray(arr)


def _from_jax(arr, cupy_module=None):
    if cupy_module is not None:
        return cupy_module.from_dlpack(arr)
    return np.asarray(arr)


def jax_threshold(binaries, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh=7,
                  batch_size=None, return_mask=False, prefilter_snr=1.0, capacity_cache=None):
    '''
    Threshold every galaxy in binaries with the JAX kernel, batch_size galaxies at a time.

    Arguments
    -----------
    binaries (array)   : Shape (2,Ndraws), (2,Ndraws,Nrealz), or (2,Ndraws,Nrealz,Nparallel),
        numpy or cupy; the first axis is (frequency, amplitude).
    edges (array)      : Upper bin edges, fs + 0.5*delf, shape (Nf,).
    noisePSD, LISA_rx (array) : Per-bin noise PSD and LISA response, shape (Nf,).
    duration, duration_eff (float) : As in SNR_Threshold.
    snr_thresh (float) : Resolvability threshold.
    batch_size (int)   : Galaxies (Nrealz*Nparallel) per jitted call. Default None (all at once).
    return_mask (bool) : Whether to also return the per-binary resolved mask.
    prefilter_snr (float) : Per-bin SNR cut below which binaries skip the scan (see module
        docstring); must be <= snr_thresh. None disables the pre-filter. Default 1.
    capacity_cache (dict) : Survivor capacity carried between calls, keyed by (N, batch).
        Default None (a fresh cache per call).

    Returns
    -----------
    Nres : Resolved count, excluding bin 0, shape (Nrealz,Nparallel).
    foreground_amp : Unresolved power per bin, shape (Nf,Nrealz,Nparallel).
    mask (only if return_mask) : Resolved flags, shape (Ndraws,Nrealz,Nparallel); bin-0 sources included.
    All three are squeezed over (Nrealz,Nparallel) when both are 1, as in block_array_sort,
    and are returned as the same array library as binaries.
    '''
    if prefilter_snr is not None and prefilter_snr > snr_thresh:
        raise ValueError("prefilter_snr ({}) must not exceed snr_thresh ({}); binaries between the "
                         "two could be resolved".format(prefilter_snr, snr_thresh))
    if capacity_cache is None:
        capacity_cache = {}

    xp = backend.xp
    cupy_module = xp if _is_cupy(binaries) else None
    if cupy_module is None:
        xp = np

    if binaries.ndim == 2:
        binaries_4d = binaries[:, :, None, None]
    elif binaries.ndim == 3:
        binaries_4d = binaries[:, :, :, None]
    elif binaries.ndim == 4:
        binaries_4d = binaries
    else:
        raise ValueError("Invalid shape. Binaries can be of shapes "
                         "(2,Ndraws), (2,Ndraws,Nrealz), or (2,Ndraws,Nrealz,Nparallel)")
    _, N, Nr, Np = binaries_4d.shape
    G = Nr*Np
    B = G if batch_size is None else int(min(batch_size, G))
    binaries_3d = binaries_4d.reshape(2, N, G)

    consts = [_to_jax(xp.ascontiguousarray(xp.asarray(c, dtype=xp.float64))) for c in (edges, noisePSD, LISA_rx)]
    scalars = [float(duration), float(duration_eff), float(snr_thresh)]
    cut = 0.0 if prefilter_snr is None else float(prefilter_snr)
    cache_key = (N, B)

    Nres_f, fg_f, mask = [], [], []
    for g0 in range(0, G, B):
        chunk = xp.ascontiguousarray(xp.moveaxis(binaries_3d[:, :, g0:g0+B], -1, 0).astype(xp.float64))
        n_real = chunk.shape[0]
        if n_real < B:
            ## padded galaxies (f = A = 0) land in bin 0 with no power; their outputs are dropped
            chunk = xp.concatenate([chunk, xp.zeros((B - n_real, 2, N), dtype=xp.float64)], axis=0)
        fA = _to_jax(chunk)
        if prefilter_snr is None:
            out = _threshold_batch(fA, *consts, *scalars, cut)
        else:
            capacity = capacity_cache.get(cache_key)
            if capacity is None:
                capacity = _bucket(jnp.max(_count_survivors(fA, *consts, scalars[0], cut)), N)
            while True:
                out = _threshold_batch(fA, *consts, *scalars, cut, capacity=capacity)
                n_max = int(jnp.max(out[3]))
                if n_max <= capacity:
                    break
                ## overflow: some survivors fell outside the buffer, so this result is discarded
                capacity = _bucket(n_max, N)
            ## carry the capacity forward, shrinking only once it is 4x too large
            capacity_cache[cache_key] = _bucket(n_max, N) if 4*n_max < capacity else capacity
        Nres_f.append(_from_jax(out[0], cupy_module)[:n_real])
        fg_f.append(_from_jax(out[1], cupy_module)[:n_real])
        if return_mask:
            mask.append(_from_jax(out[2], cupy_module)[:n_real])

    Nres_f = xp.moveaxis(xp.concatenate(Nres_f, axis=0).reshape(Nr, Np, -1), -1, 0).astype(xp.int64)
    fg_f = xp.moveaxis(xp.concatenate(fg_f, axis=0).reshape(Nr, Np, -1), -1, 0)
    Nres = xp.sum(Nres_f[1:, ...], axis=0)
    if Nr == 1 and Np == 1:
        Nres = Nres.squeeze()
        fg_f = fg_f.squeeze(axis=(1, 2))
    if not return_mask:
        return Nres, fg_f

    mask = xp.moveaxis(xp.concatenate(mask, axis=0).reshape(Nr, Np, N), -1, 0)
    if Nr == 1 and Np == 1:
        mask = mask.squeeze(axis=(1, 2))
    return Nres, fg_f, mask
