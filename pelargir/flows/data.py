"""
Torch-free data layer of the flow emulators (flows/zuko_base.py: zuko; flows/flax_base.py: flax): hyperpriors,
frequency grids and bands, training sets from the JAX forward model, and helpers shared by both
flow bases. With Poisson N_tot and per-bin thresholding, disjoint bins are independent given the
context, so an emulator's joint density is the product of per-band densities.
"""
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as ss

from ..log_utils import progress_bar, format_seconds

logger = logging.getLogger('pelargir')

CONTEXT_NAMES = ['m_mu', 'm_sigma', 'rh_disk', 'r_bulge', 'q_bd', 'a_alpha', 'rho_thresh', 'log10_lambda_tot']
N_POP = 6

## Table 1 hyperpriors (as analytic_inference.PopulationHyperPrior), plus rho_thresh and lambda_tot
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

    @property
    def draw_index(self):
        '''

        Hyperprior draw of each row: row // n_real for a full set from draw_training_set (rows are
        draw-major), otherwise the index of the row's distinct context.

        Arguments
        ------------------------------
        None

        Returns
        ------------------------------
        draws : numpy.ndarray
        	(R,) int draw index per row.

        '''
        R = self.context.shape[0]
        n_draws, n_real = self.meta.get('n_draws'), self.meta.get('n_real')
        if n_draws is not None and n_real is not None and n_draws*n_real == R:
            return np.arange(R)//n_real
        return context_draws(self.context)

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
    from ..sim import jax_population as jp
    from ..inference.analytic_inference import GalacticBinaryPrior
    from ..utils import lisa_noise_psd
    fbins = np.asarray(fbins, dtype=np.float64)
    fbin = fbins[1] - fbins[0]
    rx = lw.psd.approximate_response_function(fbins*u.Hz, 19.09*u.mHz).value
    consts = (fbins + 0.5*fbin, lisa_noise_psd(fbins, cpu=True), rx, DURATION, 1/fbin)
    return consts, jp.prior_bounds(GalacticBinaryPrior(np.random.default_rng(0)))


def _row_plan(key, rows):
    '''N_tot ~ Poisson(10**log10_lambda_tot) per context row, and the galaxy key (row i's galaxy is fold_in(it, i)).'''
    from .. import backend
    jax = backend.import_jax()
    import jax.numpy as jnp
    k_N, k_gal = jax.random.split(key)
    return np.asarray(jax.random.poisson(k_N, jnp.asarray(10**rows[:, 7])), dtype=np.int64), k_gal


def _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch, widths=None, state=None,
                   progress=False, capacity_fracs=None, devices=None):
    '''Per-row N_res and S_gw on fbins[1:] via jax_population.forward_galaxies, and its info dict.'''
    from ..sim import jax_population as jp
    consts, bounds = _forward_consts(fbins)
    nres_f, fg, info = jp.forward_galaxies(k_gal, np.arange(len(Ns)), rows[:, :N_POP], Ns, rows[:, 6], consts, bounds,
                                           widths=widths, prefilter_snr=prefilter_snr,
                                           capacity_fracs=jp.CAPACITY_FRACS if capacity_fracs is None else capacity_fracs,
                                           max_binaries_per_batch=max_binaries_per_batch, state=state,
                                           progress=progress, devices=devices)
    return nres_f[:, 1:].astype(np.int32), fg[:, 1:]*consts[4], info


def simulate(key, contexts, n_real, fbins, prefilter_snr=1.0, max_binaries_per_batch=None, devices=None, progress=False):
    '''
    n_real realizations of the forward model at each context row, via jax_population.forward_galaxies.

    Arguments
    -----------
    key : JAX PRNG key.
    contexts (array) : Shape (P, 8), CONTEXT_NAMES order.
    n_real (int) : Realizations per context.
    fbins (array) : Model bin centres (model_fbins).
    prefilter_snr (float) : Per-bin pre-filter cut (see jax_thresholding); must be <= every rho_thresh.
    max_binaries_per_batch (float) : Drawn binaries per jitted batch; None fills the GPU's free memory.
    devices : Devices to run on (jax_population.resolve_devices); default all local devices.
    progress (bool) : Show forward_galaxies' progress bar and log the run's size and time.

    Returns
    -----------
    nres_f (P, n_real, Nf'), psd (P, n_real, Nf') on fbins[1:], and N_tot (P, n_real).
    '''
    contexts = np.asarray(contexts, dtype=np.float64)
    P = contexts.shape[0]
    rows = np.repeat(contexts, n_real, axis=0)
    Ns, k_gal = _row_plan(key, rows)
    if progress:
        logger.info("simulating %d galaxies (%d contexts x %d realizations, %.3g binaries)", rows.shape[0], P, n_real,
                    float(Ns.sum()))
    nres_f, psd, info = _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch, devices=devices,
                                       progress=progress)
    if progress:
        logger.info("simulated %d galaxies in %s", rows.shape[0], format_seconds(info['seconds']))
    return nres_f.reshape(P, n_real, -1), psd.reshape(P, n_real, -1), Ns.reshape(P, n_real)


def _save_atomic(path, **arrays):
    '''np.savez to path via a temporary file, so a killed job never leaves a partial file.'''
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class _RunState:
    '''
    draw_training_set's resumable state in a directory: plan.npz (the settings, validated on
    resume) and jax_population.GalaxyState's arrays as .npy files on disk (memory-mapped), written
    batch by batch. A batch's results are flushed before its next_tier flags, so a run killed at any
    point recomputes at most the batches it had in flight.
    '''
    ARRAYS = (('nres_f', np.int32, True), ('fg', np.float64, True), ('n_surv', np.int64, False),
              ('next_tier', np.int8, False))

    def __init__(self, directory, G, Nf, settings):
        self.dir = str(directory)
        os.makedirs(self.dir, exist_ok=True)
        plan = os.path.join(self.dir, 'plan.npz')
        self.new = not os.path.exists(plan)
        if not self.new:
            d = np.load(plan)
            for k, v in settings.items():
                if k not in d.files or not np.array_equal(d[k], np.asarray(v), equal_nan=True):
                    raise ValueError("{} was made with different settings ({}); use a new chunk_dir".format(self.dir, k))
        for name, dtype, per_bin in self.ARRAYS:
            path = os.path.join(self.dir, name + '.npy')
            shape = (G, Nf) if per_bin else (G,)
            if self.new:
                arr = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            else:
                arr = np.lib.format.open_memmap(path, mode='r+')
                if arr.shape != shape or arr.dtype != dtype:
                    raise ValueError("{} has shape {} and dtype {}; expected {} and {}".format(
                        path, arr.shape, arr.dtype, shape, np.dtype(dtype)))
            setattr(self, name, arr)
        if self.new:
            ## the arrays start zeroed (next_tier 0: every galaxy to run); the plan, written last, marks them valid
            for name, _, _ in self.ARRAYS:
                getattr(self, name).flush()
            _save_atomic(plan, **settings)

    def commit(self, rows, nres_f, fg, n_surv, next_tier):
        self.nres_f[rows], self.fg[rows], self.n_surv[rows] = nres_f, fg, n_surv
        for a in (self.nres_f, self.fg, self.n_surv):
            a.flush()
        self.next_tier[rows] = next_tier
        self.next_tier.flush()


def draw_training_set(n_draws, n_real, fbins, seed, prefilter_snr=1.0, max_binaries_per_batch=None, chunk_dir=None,
                      capacity_fracs=None, progress=True, devices=None):
    '''
    TrainingSet of n_draws hyperprior draws x n_real realizations (rows ordered draw-major). Row i
    has N_tot ~ Poisson(lambda_tot) and galaxy key fold_in(k_gal, i), both from the seed, so the set
    depends only on the seed and settings (including jax_population.DRAW_CHUNK), not on batching or
    the GPU. Galaxies are drawn in chunks, pre-filtered per bin at prefilter_snr (at most RHO_MIN,
    the bottom of rho_thresh's prior), and run grouped by chunk count in batches sized to the GPU's
    free memory (or max_binaries_per_batch drawn binaries); see jax_population.forward_galaxies.
    With chunk_dir, results are written there batch by batch (_RunState), and a rerun with the same
    settings resumes where the last one stopped, on any GPUs. devices (jax_population.resolve_devices)
    selects the GPUs to spread batches over; the default is every visible one.
    '''
    from .. import backend
    from ..sim import jax_population as jp
    jax = backend.import_jax()

    t0 = time.time()
    if prefilter_snr is not None and prefilter_snr > RHO_MIN:
        raise ValueError("prefilter_snr ({}) must not exceed RHO_MIN ({}), the bottom of rho_thresh's prior".format(
            prefilter_snr, RHO_MIN))
    capacity_fracs = jp.CAPACITY_FRACS if capacity_fracs is None else tuple(capacity_fracs)
    rng = np.random.default_rng(seed)
    contexts = sample_context(rng, n_draws)
    rows = np.repeat(contexts, n_real, axis=0)
    Ns, k_gal = _row_plan(jax.random.key(seed), rows)
    G, Nf = Ns.size, np.asarray(fbins).size
    state = None
    if chunk_dir is None:
        logger.info("no chunk_dir: this run keeps results in memory and cannot be resumed")
    else:
        settings = dict(context=contexts, ntot=Ns, n_real=n_real, seed=seed, fbins=np.asarray(fbins, dtype=np.float64),
                        prefilter_snr=np.nan if prefilter_snr is None else float(prefilter_snr),
                        capacity_fracs=np.array([np.nan if f is None else f for f in capacity_fracs]),
                        draw_chunk=jp.DRAW_CHUNK, chunk_growth=jp.CHUNK_GROWTH)
        state = _RunState(chunk_dir, G, Nf, settings)
        todo = np.asarray(state.next_tier) != jp.DONE
        if state.new:
            logger.info("starting a new run: %d galaxies (%d draws x %d realizations), state in %s",
                        G, n_draws, n_real, chunk_dir)
        elif not todo.any():
            logger.info("%s is complete: assembling the training set without simulating", chunk_dir)
        else:
            k = int(G - todo.sum())
            logger.info("resuming from %s: %d of %d galaxies done (%.1f%% of binaries); %d overflow reruns pending; "
                        "continuing at tier %d", chunk_dir, k, G, 100*(1 - Ns[todo].sum()/max(Ns.sum(), 1)),
                        int(np.sum(np.asarray(state.next_tier) > 0)), int(np.asarray(state.next_tier)[todo].min()))
    nres_f, psd, info = _simulate_rows(k_gal, rows, Ns, fbins, prefilter_snr, max_binaries_per_batch,
                                       state=state, progress=progress, capacity_fracs=capacity_fracs, devices=devices)
    meta = dict(n_draws=n_draws, n_real=n_real, seed=seed, prefilter_snr=prefilter_snr,
                capacity_fracs=list(capacity_fracs), max_binaries_per_batch=max_binaries_per_batch,
                widths={str(k): v for k, v in info['widths'].items()}, width_checks=info['width_checks'],
                compiles=info['compiles'],
                simulated_galaxies=info['simulated'], resumed_galaxies=info['resumed'], devices=info['devices'],
                device_batches=info['device_batches'],
                overflow_reruns=info['overflow'], draw_chunk=info['draw_chunk'],
                chunk_growth=jp.CHUNK_GROWTH, buckets=info['buckets'], context_names=CONTEXT_NAMES,
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


def split_by_draw(draws, val_frac, rng):
    '''

    Train/validation split that keeps every realization of a hyperprior draw on one side, so the
    validation loss (and the checkpoint it selects) measures generalization to new contexts.

    Arguments
    ------------------------------
    draws : numpy.ndarray
    	(M,) draw index of each row (e.g. TrainingSet.draw_index); equal contexts share a draw.
    val_frac : float
    	Fraction of the distinct draws held out for validation.
    rng : numpy.random.Generator
    	Generator for the permutation of the draws.

    Returns
    ------------------------------
    val_rows : numpy.ndarray
    	Sorted row indices of the validation set (empty if val_frac rounds to no draws).
    trn_rows : numpy.ndarray
    	Sorted row indices of the training set.

    '''
    draws = np.asarray(draws)
    distinct = np.unique(draws)
    n_val = int(round(val_frac*distinct.size))
    is_val = np.isin(draws, rng.permutation(distinct)[:n_val])
    return np.flatnonzero(is_val), np.flatnonzero(~is_val)


def context_draws(context):
    '''

    Draw index per row from the contexts alone (rows with identical contexts share a draw), for
    training data without a TrainingSet.

    Arguments
    ------------------------------
    context : numpy.ndarray
    	(M, C) context rows.

    Returns
    ------------------------------
    draws : numpy.ndarray
    	(M,) int draw index per row.

    '''
    return np.unique(np.asarray(context), axis=0, return_inverse=True)[1].reshape(-1)


def quadrature_convergence(emulator, context, N_bands, S, n_quads=(2, 4, 8, 16, 32, 64), tol=1e-3, progress=False):
    '''
    Per band and n_quad, the largest |log P(n_quad) - log P(max n_quad)| over rows; and the smallest
    n_quad within tol nats of the reference for every band. Works for any emulator with
    band_log_prob(j, context, N, S, n_quad). progress shows a bar over bands x n_quads (each new
    n_quad compiles on the flax base).
    '''
    context = np.atleast_2d(context)
    ref_n = max(n_quads)
    err = np.zeros((len(emulator.bands), len(n_quads)))
    bar = progress_bar(total=len(emulator.bands)*(len(n_quads) + 1), desc='quadrature', unit='eval', enabled=progress,
                       leave=False)
    for j, b in enumerate(emulator.bands):
        sl = b.slice
        bar.set_postfix_str(f"band {j}", refresh=False)
        ref = emulator.band_log_prob(j, context, N_bands[:, j], S[:, sl], n_quad=ref_n)
        bar.update()
        for i, n in enumerate(n_quads):
            err[j, i] = np.max(np.abs(emulator.band_log_prob(j, context, N_bands[:, j], S[:, sl], n_quad=n) - ref))
            bar.update()
    bar.close()
    ok = [n for i, n in enumerate(n_quads) if n < ref_n and np.all(err[:, i] < tol)]
    return err, (min(ok) if ok else None)


def load_emulator(directory, device=None, weights='best'):
    '''

    A saved emulator of either flow base ('flow_base' in emulator.json).

    Arguments
    ------------------------------
    directory : str
    	Emulator directory.
    device : str or None
    	torch device for a zuko emulator; ignored for flax. (Default: None, the CPU)
    weights : str
    	'best' (best-validation weights) or 'final' (last epoch). (Default: 'best')

    Returns
    ------------------------------
    emulator : zuko_base.BandedFlowEmulator or flax_base.BandedFlowEmulator
    	The loaded emulator.

    '''
    with open(os.path.join(directory, 'emulator.json')) as fh:
        base = json.load(fh).get('flow_base', 'zuko')
    if base == 'zuko':
        from .zuko_base import BandedFlowEmulator
        return BandedFlowEmulator.load(directory, device='cpu' if device is None else device, weights=weights)
    if base == 'flax':
        from .flax_base import BandedFlowEmulator
        return BandedFlowEmulator.load(directory, weights=weights)
    raise ValueError("unknown flow_base {!r} in {}".format(base, directory))
