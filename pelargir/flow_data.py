"""
Torch-free data layer of the flow emulators (flows.py: zuko; flax_flows.py: flax): hyperpriors,
frequency grids and bands, training sets from the JAX forward model, and helpers shared by both
flow bases. With Poisson N_tot and per-bin thresholding, disjoint bins are independent given the
context, so an emulator's joint density is the product of per-band densities.
"""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as ss

CONTEXT_NAMES = ['m_mu', 'm_sigma', 'rh_disk', 'r_bulge', 'q_bd', 'a_alpha', 'rho_thresh', 'log10_lambda_tot']
N_POP = 6

## Table 1 hyperpriors (as inference.PopulationHyperPrior), plus rho_thresh and lambda_tot
RHO_MIN = 1.0
LAMBDA_RANGE = (5e5, 5e7)
HYPERPRIOR = {'m_mu': ss.uniform(loc=0.2, scale=0.9),
              'm_sigma': ss.invgamma(7),
              'rh_disk': ss.uniform(loc=1, scale=9),
              'r_bulge': ss.uniform(loc=0.05, scale=1.95),
              'q_bd': ss.uniform(loc=0.01, scale=0.98),
              'a_alpha': ss.uniform(loc=-0.5, scale=2.0),
              ## N(8, 2) truncated at the pre-filter cut
              'rho_thresh': ss.truncnorm((RHO_MIN - 8)/2, np.inf, loc=8, scale=2),
              'log10_lambda_tot': ss.uniform(loc=np.log10(LAMBDA_RANGE[0]),
                                             scale=np.log10(LAMBDA_RANGE[1]/LAMBDA_RANGE[0]))}

DEFAULT_FMIN, DEFAULT_FMAX, DEFAULT_FBIN = 1e-4, 1e-3, 2e-5
DEFAULT_BINS_PER_BAND = 5
DURATION = 1.262e8  ## SNR_Threshold's default observation time [s]


def sample_context(rng, n):
    '''n draws from HYPERPRIOR, shape (n, 8) in CONTEXT_NAMES order.'''
    return np.column_stack([HYPERPRIOR[k].rvs(size=n, random_state=rng) for k in CONTEXT_NAMES])


# =============================================================================
# Grids and bands
# =============================================================================

def model_fbins(fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX, fbin=DEFAULT_FBIN):
    '''Model bin centres as in run_pelargir; bin 0 is the below-band catch-all, so fbins[1:] is modelled.'''
    return np.arange(fmin - fbin/2, fmax + fbin/2, fbin)


@dataclass
class FrequencyBand:
    '''A contiguous run of modelled bins: fs[start:stop] of the returned grid fbins[1:].'''
    index: int
    start: int
    stop: int
    fs: np.ndarray = field(repr=False)

    @property
    def slice(self):
        return slice(self.start, self.stop)

    @property
    def nf(self):
        return self.stop - self.start

    def to_dict(self):
        return dict(index=self.index, start=self.start, stop=self.stop)


def make_bands(fs, bins_per_band=DEFAULT_BINS_PER_BAND, edges=None):
    '''
    Partition the modelled grid fs into bands: consecutive runs of bins_per_band bins (the last
    may be shorter), or, if edges (Hz, increasing) is given, band j holds the bins with
    edges[j] <= f < edges[j+1]. Bins outside [edges[0], edges[-1]) are not modelled.
    '''
    fs = np.asarray(fs)
    if edges is None:
        bounds = list(range(0, fs.size, bins_per_band)) + [fs.size]
    else:
        bounds = [int(np.searchsorted(fs, e, side='left')) for e in edges]
    bands = []
    for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b <= a:
            raise ValueError("band {} ({} to {}) contains no bins".format(j, bounds[j], bounds[j+1]))
        bands.append(FrequencyBand(j, a, b, fs[a:b]))
    return bands


# =============================================================================
# Training sets (JAX forward model)
# =============================================================================

@dataclass
class TrainingSet:
    '''
    Simulator realizations: one row per galaxy. context (R, 8) in CONTEXT_NAMES order; nres_f and
    psd (R, Nf') are the per-bin resolved counts and foreground PSD on fbins[1:]; ntot (R,) the
    galaxy's number of binaries.
    '''
    context: np.ndarray
    nres_f: np.ndarray
    psd: np.ndarray
    ntot: np.ndarray
    fbins: np.ndarray
    meta: dict

    def save(self, path):
        np.savez(path, context=self.context, nres_f=self.nres_f, psd=self.psd, ntot=self.ntot, fbins=self.fbins,
                 meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d['context'], d['nres_f'], d['psd'], d['ntot'], d['fbins'], json.loads(str(d['meta'])))

    @property
    def fs(self):
        return self.fbins[1:]

    def subset(self, rows):
        return TrainingSet(self.context[rows], self.nres_f[rows], self.psd[rows], self.ntot[rows], self.fbins,
                           self.meta)


def band_view(ts, band):
    '''(context, N_res of the band, S_gw in the band's bins) for every row of ts.'''
    return ts.context, ts.nres_f[:, band.slice].sum(axis=1), ts.psd[:, band.slice]


def simulate(key, contexts, n_real, fbins, prefilter_snr=1.0, max_binaries_per_batch=2.5e7, capacity_cache=None):
    '''
    n_real realizations of the forward model at each context row, via jax_population.jax_forward_model.

    Arguments
    -----------
    key : JAX PRNG key.
    contexts (array) : Shape (P, 8), CONTEXT_NAMES order.
    n_real (int) : Realizations per context.
    fbins (array) : Model bin centres (model_fbins).
    prefilter_snr (float) : Per-bin pre-filter cut (see jax_thresholding); must be <= every rho_thresh.
    max_binaries_per_batch (float) : Bound on galaxies*padded size per jitted batch (GPU memory).
    capacity_cache (dict) : Pre-filter survivor capacity carried between calls.

    Returns
    -----------
    nres_f (P, n_real, Nf'), psd (P, n_real, Nf') on fbins[1:], and N_tot (P, n_real).
    '''
    from . import backend
    jax = backend.import_jax()
    import jax.numpy as jnp
    import legwork as lw
    import astropy.units as u
    from . import jax_population as jp
    from .inference import GalacticBinaryPrior
    from .utils import lisa_noise_psd

    fbins = np.asarray(fbins, dtype=np.float64)
    fbin = fbins[1] - fbins[0]
    rx = lw.psd.approximate_response_function(fbins*u.Hz, 19.09*u.mHz).value
    Sn = lisa_noise_psd(fbins, cpu=True)
    bounds = jp.prior_bounds(GalacticBinaryPrior(np.random.default_rng(0)))
    capacity_cache = {} if capacity_cache is None else capacity_cache

    contexts = np.asarray(contexts, dtype=np.float64)
    P = contexts.shape[0]
    k_N, k_draw = jax.random.split(key)
    lam = 10**contexts[:, 7]
    Ns = np.asarray(jax.random.poisson(k_N, jnp.asarray(lam), (n_real, P)), dtype=np.int64)

    ## group columns of similar size so each batch's padded size (and memory) stays bounded
    order = np.argsort(Ns.max(axis=0), kind='stable')
    buckets = np.array([jp.pad_bucket(n) for n in Ns.max(axis=0)[order]])
    nres_f = np.zeros((P, n_real, fbins.size - 1), dtype=np.int32)
    psd = np.zeros((P, n_real, fbins.size - 1))
    for gi, b in enumerate(np.unique(buckets)):
        cols = order[buckets == b]
        B = int(max(1, min(len(cols)*n_real, max_binaries_per_batch//b)))
        Nres, fg, Nres_f = jp.jax_forward_model(
            jax.random.fold_in(k_draw, gi), contexts[cols, :N_POP], Ns[:, cols], bounds, fbins + 0.5*fbin,
            Sn, rx, DURATION, 1/fbin, snr_thresh=contexts[cols, 6], batch_size=B, prefilter_snr=prefilter_snr,
            capacity_cache=capacity_cache, return_nres_f=True, out_module=np)
        Nres_f = Nres_f.reshape(fbins.size, n_real, len(cols))
        fg = fg.reshape(fbins.size, n_real, len(cols))
        nres_f[cols] = np.moveaxis(Nres_f[1:], (0, 1, 2), (2, 1, 0))
        psd[cols] = np.moveaxis(fg[1:]/fbin, (0, 1, 2), (2, 1, 0))
    return nres_f, psd, Ns.T


def draw_training_set(n_draws, n_real, fbins, seed, chunk=256, prefilter_snr=1.0, max_binaries_per_batch=2.5e7,
                      chunk_dir=None, progress=True):
    '''
    TrainingSet of n_draws hyperprior draws x n_real realizations (rows ordered draw-major).
    Reproducible for a fixed (seed, chunk, max_binaries_per_batch). With chunk_dir, each chunk is
    saved there as it completes and chunks already on disk are reused, so an interrupted run
    resumes with the same result.
    '''
    from . import backend
    jax = backend.import_jax()

    rng = np.random.default_rng(seed)
    contexts = sample_context(rng, n_draws)
    key = jax.random.key(seed)
    if chunk_dir is not None:
        os.makedirs(chunk_dir, exist_ok=True)
    nres_f, psd, Ntot = [], [], []
    cache = {}
    t0 = time.time()
    n_new = 0
    for c0 in range(0, n_draws, chunk):
        ctx = contexts[c0:c0+chunk]
        path = None if chunk_dir is None else os.path.join(chunk_dir, 'chunk_{:05d}.npz'.format(c0//chunk))
        if path is not None and os.path.exists(path):
            d = np.load(path)
            if not (np.array_equal(d['context'], ctx) and int(d['n_real']) == n_real
                    and np.array_equal(d['fbins'], fbins)):
                raise ValueError("{} was made with different settings; use a new chunk_dir".format(path))
            out = (d['nres_f'], d['psd'], d['ntot'])
        else:
            out = simulate(jax.random.fold_in(key, c0//chunk), ctx, n_real, fbins, prefilter_snr=prefilter_snr,
                           max_binaries_per_batch=max_binaries_per_batch, capacity_cache=cache)
            n_new += len(ctx)
            if path is not None:
                tmp = path[:-4] + '.tmp.npz'
                np.savez(tmp, context=ctx, n_real=n_real, fbins=fbins, nres_f=out[0], psd=out[1], ntot=out[2])
                os.replace(tmp, path)
        for lst, v in zip((nres_f, psd, Ntot), out):
            lst.append(v)
        if progress:
            done = min(c0 + chunk, n_draws)
            el = time.time() - t0
            left = el/n_new*(n_draws - done) if n_new else float('nan')
            print("draws {}/{}: {:.0f} s elapsed, ~{:.0f} s left".format(done, n_draws, el, left), flush=True)
    nres_f = np.concatenate(nres_f).reshape(n_draws*n_real, -1)
    psd = np.concatenate(psd).reshape(n_draws*n_real, -1)
    meta = dict(n_draws=n_draws, n_real=n_real, seed=seed, chunk=chunk, prefilter_snr=prefilter_snr,
                max_binaries_per_batch=max_binaries_per_batch, context_names=CONTEXT_NAMES,
                rho_min=RHO_MIN, lambda_range=LAMBDA_RANGE, duration=DURATION, seconds=time.time() - t0,
                simulated_draws=n_new)
    return TrainingSet(np.repeat(contexts, n_real, axis=0), nres_f, psd, np.concatenate(Ntot).reshape(-1),
                       np.asarray(fbins), meta)



# =============================================================================
# Helpers shared by the flow bases
# =============================================================================

def _standardizer(x):
    '''(median, half-range) per column, as in the notebook; a constant column gets half-range 1.'''
    med = np.median(x, axis=0)
    half = (x.max(axis=0) - x.min(axis=0))/2
    return med, np.where(half > 0, half, 1.0)


def gauss_legendre_01(n):
    '''Gauss-Legendre nodes and log weights on [0, 1].'''
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1)/2, np.log(w/2)


def quadrature_convergence(emulator, context, N_bands, S, n_quads=(2, 4, 8, 16, 32, 64), tol=1e-3):
    '''
    Per band and n_quad, the largest |log P(n_quad) - log P(max n_quad)| over rows; and the smallest
    n_quad within tol nats of the reference for every band. Works for any emulator with
    band_log_prob(j, context, N, S, n_quad).
    '''
    context = np.atleast_2d(context)
    ref_n = max(n_quads)
    err = np.zeros((len(emulator.bands), len(n_quads)))
    for j, b in enumerate(emulator.bands):
        sl = b.slice
        ref = emulator.band_log_prob(j, context, N_bands[:, j], S[:, sl], n_quad=ref_n)
        for i, n in enumerate(n_quads):
            err[j, i] = np.max(np.abs(emulator.band_log_prob(j, context, N_bands[:, j], S[:, sl], n_quad=n) - ref))
    ok = [n for i, n in enumerate(n_quads) if n < ref_n and np.all(err[:, i] < tol)]
    return err, (min(ok) if ok else None)


def load_emulator(directory, device=None):
    '''A saved emulator of either flow base ('flow_base' in emulator.json; saves without it are zuko).'''
    with open(os.path.join(directory, 'emulator.json')) as fh:
        base = json.load(fh).get('flow_base', 'zuko')
    if base == 'zuko':
        from .flows import BandedFlowEmulator
        return BandedFlowEmulator.load(directory, device='cpu' if device is None else device)
    if base == 'flax':
        from .flax_flows import BandedFlowEmulator
        return BandedFlowEmulator.load(directory)
    raise ValueError("unknown flow_base {!r} in {}".format(base, directory))
