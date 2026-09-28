"""
Fixed-shape JAX implementation of the SNR thresholding in thresholding.SNR_Threshold.

Instead of padding each frequency bin to its most-populated count, all N binaries of a
galaxy are sorted once by (bin, amplitude), so each bin is a contiguous segment. The
per-bin confusion cumsum is a segmented scan, the Eq. 17 boundary (arXiv:2604.03390) is
a segment max, and N_res and the foreground are segment sums. Every array has static
shape N, so the whole computation jits. Results match serial_array_sort/block_array_sort
on identical inputs (tests/test_jax_thresholding.py).
"""
import backend

jax = backend.import_jax()
import jax.numpy as jnp
from jax import lax

import numpy as np


def _threshold_one(f, A, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh):
    '''
    Threshold one galaxy.

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
    Nf = edges.shape[0]
    N = f.shape[0]
    ## same convention as SNR_Threshold.coarsegrain_bin; k == Nf is out of band
    k = jnp.digitize(f, edges).astype(jnp.int32)
    k_in = jnp.minimum(k, Nf-1)
    a = jnp.where(k < Nf, A*jnp.sqrt(LISA_rx[k_in]), 0.0)

    pos = jnp.arange(N, dtype=jnp.int32)
    k_s, a_s, order = lax.sort((k, a, pos), num_keys=2)
    a2 = a_s**2

    ## per-bin cumsum of a^2 (restarting at each bin), i.e. calc_Nij's cumsum(A**2)
    seg_start = jnp.concatenate([jnp.ones(1, dtype=bool), k_s[1:] != k_s[:-1]])
    seg_add = lambda x, y: (jnp.where(y[1], y[0], x[0] + y[0]), x[1] | y[1])
    cum_a2, _ = lax.associative_scan(seg_add, (a2, seg_start))
    Nij = jnp.sqrt(duration*a2/(noisePSD[jnp.minimum(k_s, Nf-1)] + duration_eff*(cum_a2 - a2)))

    ## Eq. 17: sources above the loudest sub-threshold source in their bin are resolved
    last_sub = jax.ops.segment_max(jnp.where(Nij >= snr_thresh, -1, pos), k_s,
                                   num_segments=Nf+1, indices_are_sorted=True)
    resolved_s = pos > last_sub[k_s]

    Nres_f = jax.ops.segment_sum(resolved_s.astype(jnp.int32), k_s,
                                 num_segments=Nf+1, indices_are_sorted=True)[:Nf]
    fg_f = jax.ops.segment_sum(jnp.where(resolved_s, 0.0, a2), k_s,
                               num_segments=Nf+1, indices_are_sorted=True)[:Nf]
    resolved = jnp.zeros(N, dtype=bool).at[order].set(resolved_s)
    return Nres_f, fg_f, resolved


@jax.jit
def _threshold_batch(fA, edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh):
    '''fA has shape (B,2,N): a batch of galaxies, each [frequencies, amplitudes].'''
    return jax.vmap(_threshold_one, in_axes=(0, 0, None, None, None, None, None, None))(
        fA[:, 0], fA[:, 1], edges, noisePSD, LISA_rx, duration, duration_eff, snr_thresh)


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
                  batch_size=None, return_mask=False):
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

    Returns
    -----------
    Nres : Resolved count, excluding bin 0, shape (Nrealz,Nparallel).
    foreground_amp : Unresolved power per bin, shape (Nf,Nrealz,Nparallel).
    mask (only if return_mask) : Resolved flags, shape (Ndraws,Nrealz,Nparallel); bin-0 sources included.
    All three are squeezed over (Nrealz,Nparallel) when both are 1, as in block_array_sort,
    and are returned as the same array library as binaries.
    '''
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

    Nres_f, fg_f, mask = [], [], []
    for g0 in range(0, G, B):
        chunk = xp.ascontiguousarray(xp.moveaxis(binaries_3d[:, :, g0:g0+B], -1, 0).astype(xp.float64))
        n_real = chunk.shape[0]
        if n_real < B:
            ## padded galaxies (f = A = 0) land in bin 0 with no power; their outputs are dropped
            chunk = xp.concatenate([chunk, xp.zeros((B - n_real, 2, N), dtype=xp.float64)], axis=0)
        out = _threshold_batch(_to_jax(chunk), *consts, *scalars)
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
