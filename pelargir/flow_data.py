"""
Torch-free data layer of the flow emulators (flows.py: zuko; flax_flows.py: flax): hyperpriors,
frequency grids and bands, training sets from the JAX forward model, and helpers shared by both
flow bases. With Poisson N_tot and per-bin thresholding, disjoint bins are independent given the
context, so an emulator's joint density is the product of per-band densities.
"""
import json
import os
import time
import warnings
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


def drop_zero_spectra(context, N, S, band=None):
    '''
    (context, N, S, n_dropped) without the rows that have S_gw = 0 in any bin, as new arrays; warns
    if any were dropped. A flow trained on what remains learns p(N, S | context, S > 0 in every
    bin), so it leaves out the probability of a zero-foreground bin; that matters where it isn't
    small (bins above ~1 mHz, flows Stage 3).
    '''
    S = np.asarray(S)
    keep = np.all(S > 0, axis=1)
    n_dropped = int(keep.size - keep.sum())
    where = "" if band is None else " in band {}".format(band.index)
    if not keep.any():
        raise ValueError("every row has S_gw = 0 in some bin{}; nothing to train on".format(where))
    if n_dropped:
        warnings.warn("dropped {} of {} rows ({:.2%}) with S_gw = 0 in some bin{}".format(
            n_dropped, keep.size, n_dropped/keep.size, where), stacklevel=2)
    return np.asarray(context)[keep], np.asarray(N)[keep], S[keep], n_dropped


def _forward_consts(fbins):
    '''(edges, noisePSD, LISA_rx, duration, duration_eff) of the forward model on fbins, and the prior bounds.'''
    import legwork as lw
    import astropy.units as u
    from . import jax_population as jp
    from .inference import GalacticBinaryPrior
    from .utils import lisa_noise_psd
    fbins = np.asarray(fbins, dtype=np.float64)
    fbin = fbins[1] - fbins[0]
    rx = lw.psd.approximate_response_function(fbins*u.Hz, 19.09*u.mHz).value
    consts = (fbins + 0.5*fbin, lisa_noise_psd(fbins, cpu=True), rx, DURATION, 1/fbin)
    return consts, jp.prior_bounds(GalacticBinaryPrior(np.random.default_rng(0)))


def _row_plan(key, rows):
    '''N_tot ~ Poisson(10**log10_lambda_tot) per context row, and the galaxy key (row i's galaxy is fold_in(it, i)).'''
    from . import backend
    jax = backend.import_jax()
    import jax.numpy as jnp
    k_N, k_gal = jax.random.split(key)
    return np.asarray(jax.random.poisson(k_N, jnp.asarray(10**rows[:, 7])), dtype=np.int64), k_gal


def _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch, widths=None, store=None,
                   progress=False):
    '''Per-row N_res and S_gw on fbins[1:] via jax_population.forward_galaxies, and its info dict.'''
    from . import jax_population as jp
    consts, bounds = _forward_consts(fbins)
    nres_f, fg, info = jp.forward_galaxies(k_gal, np.arange(len(Ns)), rows[:, :N_POP], Ns, rows[:, 6], consts, bounds,
                                           widths=widths, prefilter_snr=prefilter_snr,
                                           max_binaries_per_batch=max_binaries_per_batch, store=store,
                                           progress=progress)
    return nres_f[:, 1:].astype(np.int32), fg[:, 1:]*consts[4], info


def simulate(key, contexts, n_real, fbins, prefilter_snr=1.0, max_binaries_per_batch=None):
    '''
    n_real realizations of the forward model at each context row, via jax_population.forward_galaxies.

    Arguments
    -----------
    key : JAX PRNG key.
    contexts (array) : Shape (P, 8), CONTEXT_NAMES order.
    n_real (int) : Realizations per context.
    fbins (array) : Model bin centres (model_fbins).
    prefilter_snr (float) : Per-bin pre-filter cut (see jax_thresholding); must be <= every rho_thresh.
    max_binaries_per_batch (float) : Padded binaries per jitted batch; None fills the GPU's free memory.

    Returns
    -----------
    nres_f (P, n_real, Nf'), psd (P, n_real, Nf') on fbins[1:], and N_tot (P, n_real).
    '''
    contexts = np.asarray(contexts, dtype=np.float64)
    P = contexts.shape[0]
    rows = np.repeat(contexts, n_real, axis=0)
    Ns, k_gal = _row_plan(key, rows)
    nres_f, psd, _ = _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch)
    return nres_f.reshape(P, n_real, -1), psd.reshape(P, n_real, -1), Ns.reshape(P, n_real)


def _save_atomic(path, **arrays):
    '''np.savez to path via a temporary file, so a killed job never leaves a partial file.'''
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class _BatchStore:
    '''draw_training_set's plan and per-batch results in a directory, for resuming.'''

    def __init__(self, directory):
        self.dir = str(directory)
        os.makedirs(self.dir, exist_ok=True)

    def plan(self, settings, make_widths):
        '''Batch widths: from plan.npz if it exists (settings must match), else make_widths(), saved with settings.'''
        path = os.path.join(self.dir, 'plan.npz')
        if os.path.exists(path):
            d = np.load(path)
            for k, v in settings.items():
                if k not in d.files or not np.array_equal(d[k], np.asarray(v), equal_nan=True):
                    raise ValueError("{} was made with different settings ({}); use a new chunk_dir".format(self.dir, k))
            return {int(b): int(w) for b, w in zip(d['width_npad'], d['width'])}
        widths = make_widths()
        _save_atomic(path, width_npad=np.array(list(widths), dtype=np.int64),
                     width=np.array(list(widths.values()), dtype=np.int64), **settings)
        return widths

    def _path(self, n_pad, i):
        return os.path.join(self.dir, 'b{}_{:05d}.npz'.format(n_pad, i))

    def get(self, n_pad, i, rows):
        path = self._path(n_pad, i)
        if not os.path.exists(path):
            return None
        d = np.load(path)
        if not np.array_equal(d['rows'], rows):
            raise ValueError("{} holds other rows than this plan's; use a new chunk_dir".format(path))
        return d['nres_f'], d['fg'], d['n_surv']

    def put(self, n_pad, i, rows, nres_f, fg, n_surv):
        _save_atomic(self._path(n_pad, i), rows=rows, nres_f=nres_f, fg=fg, n_surv=n_surv)


def draw_training_set(n_draws, n_real, fbins, seed, prefilter_snr=1.0, max_binaries_per_batch=None, chunk_dir=None,
                      capacity_fracs=None, progress=True):
    '''
    TrainingSet of n_draws hyperprior draws x n_real realizations (rows ordered draw-major). Row i
    has N_tot ~ Poisson(lambda_tot) and galaxy key fold_in(k_gal, i), both from the seed, so the set
    depends only on the seed and settings, not on batching or the GPU. Galaxies are run grouped by
    padding bucket in batches sized to the GPU's free memory (or max_binaries_per_batch padded
    binaries); see jax_population.forward_galaxies. With chunk_dir, the plan (including the batch
    widths) and every batch are saved there as they complete, and a rerun resumes from them.
    '''
    from . import backend
    from . import jax_population as jp
    jax = backend.import_jax()

    t0 = time.time()
    capacity_fracs = jp.CAPACITY_FRACS if capacity_fracs is None else tuple(capacity_fracs)
    rng = np.random.default_rng(seed)
    contexts = sample_context(rng, n_draws)
    rows = np.repeat(contexts, n_real, axis=0)
    Ns, k_gal = _row_plan(jax.random.key(seed), rows)
    first = 'unfiltered' if prefilter_snr is None or capacity_fracs[0] is None else 'filtered'
    make_widths = lambda: jp.batch_widths(Ns, max_binaries_per_batch, first)
    store, widths = None, None
    if chunk_dir is not None:
        store = _BatchStore(chunk_dir)
        settings = dict(context=contexts, ntot=Ns, n_real=n_real, seed=seed, fbins=np.asarray(fbins, dtype=np.float64),
                        prefilter_snr=np.nan if prefilter_snr is None else float(prefilter_snr),
                        capacity_fracs=np.array([np.nan if f is None else f for f in capacity_fracs]))
        widths = store.plan(settings, make_widths)
    nres_f, psd, info = _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch, widths=widths,
                                       store=store, progress=progress)
    meta = dict(n_draws=n_draws, n_real=n_real, seed=seed, prefilter_snr=prefilter_snr,
                capacity_fracs=list(capacity_fracs), max_binaries_per_batch=max_binaries_per_batch,
                widths={str(k): v for k, v in info['widths'].items()}, compiles=info['compiles'],
                simulated_galaxies=info['simulated'], overflow_reruns=info['overflow'], context_names=CONTEXT_NAMES,
                rho_min=RHO_MIN, lambda_range=LAMBDA_RANGE, duration=DURATION, seconds=time.time() - t0)
    return TrainingSet(rows, nres_f, psd, Ns, np.asarray(fbins), meta)



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
