"""
Per-band normalizing-flow emulator of the pelargir population model (arXiv:2604.03390, §6.2).

For each frequency band, a flow models the joint density of the band's resolved-binary count
N_res and its foreground spectrum S_gw (one value per bin), conditioned on the context: the
GalacticBinaryPrior hyperparameters, the SNR threshold rho_thresh, and log10 of the rate
lambda_tot of N_tot ~ Poisson(lambda_tot). With Poisson N_tot and per-bin thresholding,
disjoint bins are independent given the context, so the joint density is the product over bands.

Layers: hyperpriors, grids and bands, and training-set generation (the JAX forward model) use
numpy and JAX only; torch and zuko appear only in the flow layer (BandTransform onwards), so the
data layer carries over unchanged to a JAX flow.

Flow choices follow prototype-notebooks/pelargir-v1.0.1-flow-toy-model.ipynb: zuko NSF with 3
transforms and two hidden layers of 10*features, N_res dequantized with U(0,1), N_res, S_gw and
lambda_tot in log10, and every dimension standardized by (x - median)/half-range.
"""
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as ss
import torch
import zuko

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
# Flow layer (torch, zuko)
# =============================================================================

_LN10 = math.log(10.0)


def _standardizer(x):
    '''(median, half-range) per column, as in the notebook; a constant column gets half-range 1.'''
    med = np.median(x, axis=0)
    half = (x.max(axis=0) - x.min(axis=0))/2
    return med, np.where(half > 0, half, 1.0)


class BandTransform(torch.nn.Module):
    '''
    Maps (N_res dequantized, S_gw per bin) to the flow's standardized space: log10 of each, then
    (x - median)/half-range. forward also returns log|det dz/dx| for densities in physical units.
    '''

    def __init__(self, nf):
        super().__init__()
        self.register_buffer('loc', torch.zeros(1 + nf, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(1 + nf, dtype=torch.float64))

    def fit(self, N, S, rng):
        S = np.asarray(S, dtype=np.float64)
        if np.any(S <= 0):
            raise ValueError("S_gw must be positive in every bin; {} of {} values are not (zero-foreground bins "
                             "are not supported yet)".format(int(np.sum(S <= 0)), S.size))
        x = np.column_stack([np.log10(np.asarray(N) + rng.uniform(size=len(N))), np.log10(S)])
        med, half = _standardizer(x)
        self.loc.copy_(torch.as_tensor(med))
        self.scale.copy_(torch.as_tensor(half))
        return self

    def forward(self, N_deq, S):
        '''z (..., 1+nf) and log|det dz/dx| (...,) for dequantized counts N_deq (...,) and S (..., nf).'''
        x = torch.cat([N_deq[..., None], S], dim=-1).to(torch.float64)
        z = (torch.log10(x) - self.loc)/self.scale
        logdet = -torch.sum(torch.log(x*_LN10*self.scale), dim=-1)
        return z, logdet

    def inverse(self, z):
        '''(N_res, S_gw) from z; N_res is the floor of the dequantized count.'''
        x = 10**(z.to(torch.float64)*self.scale + self.loc)
        return torch.floor(x[..., 0]), x[..., 1:]


class ContextTransform(torch.nn.Module):
    '''Standardizes the context (log10 lambda_tot already in log space) by median and half-range.'''

    def __init__(self, n_context=len(CONTEXT_NAMES)):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_context, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(n_context, dtype=torch.float64))

    def fit(self, context):
        med, half = _standardizer(np.asarray(context, dtype=np.float64))
        self.loc.copy_(torch.as_tensor(med))
        self.scale.copy_(torch.as_tensor(half))
        return self

    def forward(self, context):
        return (torch.as_tensor(context, dtype=torch.float64, device=self.loc.device) - self.loc)/self.scale


def gauss_legendre_01(n):
    '''Gauss-Legendre nodes and log weights on [0, 1].'''
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1)/2, np.log(w/2)


class BandFlow(torch.nn.Module):
    '''
    Flow for one band: zuko NSF over z = BandTransform(N_res + u, S_gw), conditioned on the
    standardized context. Runs in float32; transforms are float64.
    '''

    def __init__(self, band, n_context=len(CONTEXT_NAMES), transforms=3, hidden_factor=10):
        super().__init__()
        self.band = band
        features = 1 + band.nf
        self.features = features
        self.hidden_factor = hidden_factor
        self.n_transforms = transforms
        self.flow = zuko.flows.NSF(features, n_context, transforms=transforms,
                                   hidden_features=[hidden_factor*features]*2)
        self.transform = BandTransform(band.nf)
        self.context_transform = ContextTransform(n_context)

    @property
    def device(self):
        return self.transform.loc.device

    def fit_transforms(self, context, N, S, rng):
        self.transform.fit(N, S, rng)
        self.context_transform.fit(context)
        return self

    def _context(self, context):
        return self.context_transform(context).to(torch.float32)

    def log_prob_z(self, context_z, z):
        '''Flow log density in the standardized space.'''
        return self.flow(context_z).log_prob(z.to(torch.float32))

    def log_prob(self, context, N, S, n_quad=32):
        '''
        log P(N_res = N, S_gw = S | context), shape (M,), for context (M, 8), integer N (M,), S (M, nf):
        log int_0^1 q(N + u, S) du by Gauss-Legendre over u, with the transform Jacobian (S_gw per Hz^-1).
        The spline integrand has kinks, so the error falls only as ~n_quad^-2 in the worst cases
        (see quadrature_convergence); 32 nodes gave <4e-5 nats on a test problem.
        '''
        dev = self.device
        c = self._context(context)
        N = torch.as_tensor(N, dtype=torch.float64, device=dev)
        S = torch.as_tensor(S, dtype=torch.float64, device=dev)
        u, logw = gauss_legendre_01(n_quad)
        M = N.shape[0]
        N_deq = N[None, :] + torch.as_tensor(u, device=dev)[:, None]               ## (K, M)
        z, logdet = self.transform(N_deq, S.expand(n_quad, *S.shape))
        lq = self.log_prob_z(c.expand(n_quad, *c.shape).reshape(n_quad*M, -1), z.reshape(n_quad*M, -1))
        lq = lq.to(torch.float64).reshape(n_quad, M) + logdet + torch.as_tensor(logw, device=dev)[:, None]
        return torch.logsumexp(lq, dim=0)

    def sample(self, context):
        '''One (N_res, S_gw) draw per context row.'''
        z = self.flow(self._context(context)).sample()
        return self.transform.inverse(z)

    def config(self):
        return dict(band=self.band.to_dict(), features=self.features, hidden_factor=self.hidden_factor,
                    transforms=self.n_transforms)


def train_band_flow(flow, context, N, S, n_epochs=8, batch_size=64, lr=1e-3, val_frac=0.1, seed=0, progress=True):
    '''
    Fit flow's transforms and train it by maximum likelihood in the standardized space (as in the
    notebook), re-dequantizing N every epoch. Returns the per-epoch train and validation losses.
    '''
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = flow.device
    rows = rng.permutation(len(N))
    n_val = int(round(val_frac*len(N)))
    val, trn = rows[:n_val], rows[n_val:]
    flow.fit_transforms(context[trn], N[trn], S[trn], rng)

    c_all = flow._context(context)
    N_t = torch.as_tensor(np.asarray(N), dtype=torch.float64, device=dev)
    S_t = torch.as_tensor(np.asarray(S), dtype=torch.float64, device=dev)
    ## fixed dequantization for the validation loss, so it is comparable across epochs
    z_val = flow.transform(N_t[val] + torch.as_tensor(rng.uniform(size=n_val), device=dev), S_t[val])[0]

    opt = torch.optim.Adam(flow.flow.parameters(), lr=lr)
    history = dict(train=[], val=[])
    for epoch in range(n_epochs):
        u = torch.as_tensor(rng.uniform(size=len(trn)), device=dev)
        z = flow.transform(N_t[trn] + u, S_t[trn])[0].to(torch.float32)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z, c_all[trn]), batch_size=batch_size,
                                             shuffle=True, generator=gen)
        losses = []
        flow.train()
        for zb, cb in loader:
            loss = -flow.flow(cb).log_prob(zb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        flow.eval()
        with torch.no_grad():
            v = -flow.log_prob_z(c_all[val], z_val).mean().item() if n_val else float('nan')
        history['train'].append(torch.stack(losses).mean().item())
        history['val'].append(v)
        if progress:
            print("band {} epoch {}: train {:.4f}, val {:.4f}".format(flow.band.index, epoch, history['train'][-1], v),
                  flush=True)
    return history


class BandedFlowEmulator(torch.nn.Module):
    '''
    One BandFlow per band; the population-conditional density of (N_res per band, S_gw per bin)
    is the product over bands.
    '''

    def __init__(self, bands, fbins, **flow_kwargs):
        super().__init__()
        self.bands = bands
        self.fbins = np.asarray(fbins)
        self.flow_kwargs = flow_kwargs
        self.flows = torch.nn.ModuleList([BandFlow(b, **flow_kwargs) for b in bands])

    @property
    def fs(self):
        return self.fbins[1:]

    def log_prob(self, context, N_bands, S, n_quad=32):
        '''
        Sum over bands of log P(N_band, S_band | context), shape (M,).
        context (M, 8); N_bands (n_bands,) or (M, n_bands); S (Nf',) or (M, Nf') on fbins[1:].
        '''
        context = np.atleast_2d(context)
        M = context.shape[0]
        N_bands = np.array(np.broadcast_to(np.asarray(N_bands), (M, len(self.bands))))
        S = np.array(np.broadcast_to(np.asarray(S, dtype=np.float64), (M, self.fs.size)))
        total = 0
        with torch.no_grad():
            for j, f in enumerate(self.flows):
                total = total + f.log_prob(context, N_bands[:, j], S[:, f.band.slice], n_quad=n_quad)
        return total.cpu().numpy()

    def sample(self, context):
        '''One draw per context row: N_res per band (M, n_bands) and S_gw (M, Nf'), NaN outside the bands.'''
        context = np.atleast_2d(context)
        N = np.zeros((context.shape[0], len(self.bands)))
        S = np.full((context.shape[0], self.fs.size), np.nan)
        with torch.no_grad():
            for j, f in enumerate(self.flows):
                Nj, Sj = f.sample(context)
                N[:, j] = Nj.cpu().numpy()
                S[:, f.band.slice] = Sj.cpu().numpy()
        return N, S

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        meta = dict(fbins=self.fbins.tolist(), bands=[b.to_dict() for b in self.bands], flow_kwargs=self.flow_kwargs,
                    context_names=CONTEXT_NAMES)
        with open(os.path.join(directory, 'emulator.json'), 'w') as fh:
            json.dump(meta, fh, indent=1)
        torch.save(self.state_dict(), os.path.join(directory, 'emulator.pt'))

    @classmethod
    def load(cls, directory, device='cpu'):
        with open(os.path.join(directory, 'emulator.json')) as fh:
            meta = json.load(fh)
        fbins = np.asarray(meta['fbins'])
        bands = [FrequencyBand(b['index'], b['start'], b['stop'], fbins[1:][b['start']:b['stop']])
                 for b in meta['bands']]
        em = cls(bands, fbins, **meta['flow_kwargs'])
        em.load_state_dict(torch.load(os.path.join(directory, 'emulator.pt'), map_location=device))
        return em.to(device).eval()


def train_emulator(ts, bands=None, device='cpu', flow_kwargs=None, **train_kwargs):
    '''Train a BandedFlowEmulator on a TrainingSet; returns (emulator, per-band loss histories).'''
    bands = make_bands(ts.fs) if bands is None else bands
    em = BandedFlowEmulator(bands, ts.fbins, **(flow_kwargs or {})).to(device)
    histories = []
    for f in em.flows:
        context, N, S = band_view(ts, f.band)
        histories.append(train_band_flow(f, context, N, S, **train_kwargs))
    return em.eval(), histories


def quadrature_convergence(emulator, context, N_bands, S, n_quads=(2, 4, 8, 16, 32, 64), tol=1e-3):
    '''
    Per band and n_quad, the largest |log P(n_quad) - log P(max n_quad)| over rows; and the smallest
    n_quad within tol nats of the reference for every band.
    '''
    context = np.atleast_2d(context)
    ref_n = max(n_quads)
    err = np.zeros((len(emulator.bands), len(n_quads)))
    with torch.no_grad():
        for j, f in enumerate(emulator.flows):
            sl = f.band.slice
            ref = f.log_prob(context, N_bands[:, j], S[:, sl], n_quad=ref_n)
            for i, n in enumerate(n_quads):
                err[j, i] = (f.log_prob(context, N_bands[:, j], S[:, sl], n_quad=n) - ref).abs().max().item()
    ok = [n for i, n in enumerate(n_quads) if n < ref_n and np.all(err[:, i] < tol)]
    return err, (min(ok) if ok else None)
