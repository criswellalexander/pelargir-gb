"""
Per-band normalizing-flow emulator of the pelargir population model (arXiv:2604.03390, §6.2).

For each frequency band, a flow models the joint density of the band's resolved-binary count
N_res and its foreground spectrum S_gw (one value per bin), conditioned on the context: the
GalacticBinaryPrior hyperparameters, the SNR threshold rho_thresh, and log10 of the rate
lambda_tot of N_tot ~ Poisson(lambda_tot). With Poisson N_tot and per-bin thresholding,
disjoint bins are independent given the context, so the joint density is the product over bands.

The torch-free data layer (hyperpriors, grids and bands, training sets) lives in flows/data.py and
is re-exported here; flows/flax_base.py is the JAX flow base built on the same data layer.

Transforms follow prototype-notebooks/pelargir-v1.0.1-flow-toy-model.ipynb: N_res dequantized with
U(0,1), N_res and lambda_tot in log10, S_gw as x = log10(S_gw + C) (flows.data: C = NOISE_FLOOR; a bin
with S_gw = 0 is dequantized into the slab below log10 C, and sampling returns max(10**x - C, 0);
C = 0 for emulators saved before Stage 3), and every dimension standardized by (x - median)/half-range. The default zuko NSF has 8 transforms, each conditioner two hidden layers of
512 (as the flax base). Training keeps each band's best-validation weights, with the validation set
held out by hyperprior draw.
"""
import json
import math
import time
import logging
import os

import numpy as np
import torch
import zuko

from ..log_utils import progress_bar, format_seconds

from .data import (CONTEXT_NAMES, N_POP, RHO_MIN, LAMBDA_RANGE, HYPERPRIOR, DEFAULT_FMIN, DEFAULT_FMAX,
                        DEFAULT_FBIN, DEFAULT_BINS_PER_BAND, DURATION, sample_context, model_fbins, FrequencyBand,
                        make_bands, TrainingSet, band_view, simulate, draw_training_set,
                        _standardizer, gauss_legendre_01, quadrature_convergence, load_emulator, split_by_draw,
                        context_draws, sobol_01, qmc_convergence, log10_shifted, unshift, NOISE_FLOOR, SLAB_DEX,
                        QMC_POINTS)

_FORMAT = "pelargir-zuko-v3"
## v2 saves (log10 S_gw, no floor) load as floor = slab = 0
_FORMATS_LOADABLE = (_FORMAT, "pelargir-zuko-v2")

# =============================================================================
# Flow layer (torch, zuko)
# =============================================================================

_LN10 = math.log(10.0)
logger = logging.getLogger('pelargir')


class BandTransform(torch.nn.Module):
    '''
    Maps (N_res dequantized, S_gw per bin) to the flow's standardized space: log10 of N_res and
    x = log10(S_gw + floor), then (x - median)/half-range; bins with S_gw = 0 take a slab position
    v in [0, 1) to x = log10(floor) - slab*v. forward also returns the log Jacobian for densities in
    physical units (log|dz/dv| for zero bins, the measure of P(S_gw = 0)). floor = 0 is the
    pre-Stage-3 log10 S_gw.
    '''

    def __init__(self, nf):
        super().__init__()
        self.register_buffer('loc', torch.zeros(1 + nf, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(1 + nf, dtype=torch.float64))
        self.register_buffer('floor', torch.zeros((), dtype=torch.float64))
        self.register_buffer('slab', torch.zeros((), dtype=torch.float64))

    def fit(self, N, S, rng, floor=NOISE_FLOOR, slab=SLAB_DEX):
        S = np.asarray(S, dtype=np.float64)
        zero = S == 0
        if np.any(S < 0) or (floor <= 0 and zero.any()):
            raise ValueError("S_gw must be positive in every bin with floor = 0 (and non-negative with floor > 0); "
                             "{} of {} values are not".format(int(np.sum(S < 0) + (floor <= 0)*np.sum(zero)), S.size))
        lN = np.log10(np.asarray(N) + rng.uniform(size=len(N)))
        lS = np.log10(S + floor)
        if zero.any():
            lS[zero] = np.log10(floor) - slab*rng.uniform(size=int(zero.sum()))
        med, half = _standardizer(np.column_stack([lN, lS]))
        self.loc.copy_(torch.as_tensor(med))
        self.scale.copy_(torch.as_tensor(half))
        self.floor.fill_(float(floor))
        self.slab.fill_(float(slab))
        return self

    def forward(self, N_deq, S, v=None):
        '''
        z (..., 1+nf) and log|det| (...,) for dequantized counts N_deq (...,) and S (..., nf); v (..., nf)
        are the slab positions of the zero bins (None when there are none).
        '''
        N_deq, S = N_deq.to(torch.float64), S.to(torch.float64)
        w = S + self.floor
        if v is None:
            lS = torch.log10(w)
            ldS = -torch.log(w*_LN10*self.scale[1:])
        else:
            zero = S == 0
            w_safe = torch.where(zero, torch.ones_like(w), w)
            lf = torch.log10(torch.where(self.floor > 0, self.floor, torch.ones_like(self.floor)))
            lS = torch.where(zero, lf - self.slab*v.to(torch.float64), torch.log10(w_safe))
            ldS = torch.where(zero, torch.log(self.slab/self.scale[1:]).expand_as(w), -torch.log(w_safe*_LN10*self.scale[1:]))
        z = (torch.cat([torch.log10(N_deq)[..., None], lS], dim=-1) - self.loc)/self.scale
        logdet = -torch.log(N_deq*_LN10*self.scale[0]) + torch.sum(ldS, dim=-1)
        return z, logdet

    def inverse(self, z):
        '''(N_res, S_gw) from z; N_res is the floor of the dequantized count, S_gw = max(10**x - floor, 0).'''
        x = 10**(z.to(torch.float64)*self.scale + self.loc)
        return torch.floor(x[..., 0]), torch.clamp(x[..., 1:] - self.floor, min=0.0)


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
        return (torch.as_tensor(np.array(context, dtype=np.float64), device=self.loc.device) - self.loc)/self.scale


class BandFlow(torch.nn.Module):
    '''

    Flow for one band: zuko NSF over z = BandTransform(N_res + u, S_gw), conditioned on the
    standardized context. Runs in float32; transforms are float64.

    Arguments
    ------------------------------
    band : flows.data.FrequencyBand
    	The band this flow models.
    n_context : int
    	Context dimension. (Default: len(CONTEXT_NAMES))
    transforms : int
    	Number of NSF transforms. (Default: 8)
    hidden_factor : int
    	Conditioner width as a multiple of the features, used only if hidden_size is None. (Default: 10)
    hidden_size : int or None
    	Width of each conditioner hidden layer. (Default: 512)
    mlp_num_layers : int
    	Hidden layers per conditioner. (Default: 2)

    Returns
    ------------------------------
    None

    '''

    def __init__(self, band, n_context=len(CONTEXT_NAMES), transforms=8, hidden_factor=10, hidden_size=512,
                 mlp_num_layers=2):
        super().__init__()
        self.band = band
        features = 1 + band.nf
        self.features = features
        self.n_context = n_context
        self.hidden_factor = hidden_factor
        self.hidden_size = hidden_size
        self.n_transforms = transforms
        self.mlp_num_layers = mlp_num_layers
        ## mlp_num_layers hidden layers of hidden_size, or of hidden_factor*features (the notebook's choice)
        width = hidden_factor*features if hidden_size is None else hidden_size
        self.flow = zuko.flows.NSF(features, n_context, transforms=transforms, hidden_features=[width]*mlp_num_layers)
        self.transform = BandTransform(band.nf)
        self.context_transform = ContextTransform(n_context)
        ## last-epoch weights of flow.flow, set by train_band_flow (the module holds the best ones)
        self.final_state = None

    @property
    def device(self):
        return self.transform.loc.device

    def fit_transforms(self, context, N, S, rng, floor=NOISE_FLOOR, slab=SLAB_DEX):
        self.transform.fit(N, S, rng, floor, slab)
        self.context_transform.fit(context)
        return self

    def _context(self, context):
        return self.context_transform(context).to(torch.float32)

    def log_prob_z(self, context_z, z):
        '''Flow log density in the standardized space.'''
        return self.flow(context_z).log_prob(z.to(torch.float32))

    def log_prob(self, context, N, S, n_quad=32, n_qmc=QMC_POINTS):
        '''
        log P(N_res = N, S_gw = S | context), shape (M,), for context (M, 8) or (1, 8), integer N (M,), S (M, nf),
        with a point mass at S_gw = 0. Rows with S_gw > 0 in every bin: log int_0^1 q(N + u, S) du by
        Gauss-Legendre over u, with the transform Jacobian (S_gw per Hz^-1); the spline integrand has kinks,
        so the error falls only as ~n_quad^-2 in the worst cases (see quadrature_convergence). Rows with a
        zero bin: the count and the zero bins' slab positions integrated jointly by n_qmc scrambled Sobol
        points (qmc_convergence).
        '''
        dev = self.device
        c = self._context(context)
        N = torch.as_tensor(np.asarray(N), dtype=torch.float64, device=dev)
        S = torch.as_tensor(np.asarray(S, dtype=np.float64), device=dev)
        M = N.shape[0]
        c = c.expand(M, -1) if c.shape[0] == 1 else c
        zero_rows = (S == 0).any(dim=-1)
        if bool(zero_rows.any()) and float(self.transform.floor) <= 0:
            raise ValueError("S_gw = 0 in band {}, which this emulator (floor 0, saved before Stage 3) can't "
                             "represent".format(self.band.index))
        out = torch.empty(M, dtype=torch.float64, device=dev)
        pos = ~zero_rows
        if bool(pos.any()):
            u, logw = gauss_legendre_01(n_quad)
            Np, Sp, cp, Mp = N[pos], S[pos], c[pos], int(pos.sum())
            N_deq = Np[None, :] + torch.as_tensor(u, device=dev)[:, None]               ## (K, M)
            z, logdet = self.transform(N_deq, Sp.expand(n_quad, *Sp.shape))
            lq = self.log_prob_z(cp.expand(n_quad, *cp.shape).reshape(n_quad*Mp, -1), z.reshape(n_quad*Mp, -1))
            lq = lq.to(torch.float64).reshape(n_quad, Mp) + logdet + torch.as_tensor(logw, device=dev)[:, None]
            out[pos] = torch.logsumexp(lq, dim=0)
        if bool(zero_rows.any()):
            pts = torch.as_tensor(sobol_01(n_qmc, 1 + self.band.nf), device=dev)
            Nz, Sz, cz, Mz = N[zero_rows], S[zero_rows], c[zero_rows], int(zero_rows.sum())
            K = pts.shape[0]
            z, logdet = self.transform(Nz[None, :] + pts[:, :1], Sz.expand(K, *Sz.shape),
                                       pts[:, None, 1:].expand(K, Mz, self.band.nf))
            lq = self.log_prob_z(cz.expand(K, *cz.shape).reshape(K*Mz, -1), z.reshape(K*Mz, -1))
            out[zero_rows] = torch.logsumexp(lq.to(torch.float64).reshape(K, Mz) + logdet, dim=0) - math.log(K)
        return out

    def sample(self, context):
        '''One (N_res, S_gw) draw per context row.'''
        z = self.flow(self._context(context)).sample()
        return self.transform.inverse(z)

    def config(self):
        return dict(band=self.band.to_dict(), features=self.features, **self.arch())

    def arch(self):
        '''

        The resolved architecture: the constructor arguments other than band, as saved with an emulator.

        Arguments
        ------------------------------
        None

        Returns
        ------------------------------
        arch : dict
        	n_context, transforms, hidden_factor, hidden_size and mlp_num_layers.

        '''
        return dict(n_context=self.n_context, transforms=self.n_transforms, hidden_factor=self.hidden_factor,
                    hidden_size=self.hidden_size, mlp_num_layers=self.mlp_num_layers)


def train_band_flow(flow, context, N, S, n_epochs=500, batch_size=4096, lr=1e-4, val_frac=0.1, seed=0, progress=True,
                    draws=None, floor=NOISE_FLOOR):
    '''

    Fit flow's transforms and train it by maximum likelihood in the standardized space (as in the
    notebook), re-dequantizing N every epoch. The validation set holds out whole hyperprior draws
    (flows.data.split_by_draw) with a fixed dequantization; the weights with the lowest validation
    loss are restored at the end, and the last epoch's are kept in flow.final_state.

    Arguments
    ------------------------------
    flow : BandFlow
    	Flow to train, on its device.
    context : numpy.ndarray
    	(M, C) context rows.
    N : numpy.ndarray
    	(M,) resolved counts of the band.
    S : numpy.ndarray
    	(M, nf) foreground PSD in the band's bins.
    n_epochs : int
    	Training epochs. (Default: 500)
    batch_size : int
    	Rows per Adam step. (Default: 4096)
    lr : float
    	Adam learning rate. (Default: 1e-4)
    val_frac : float
    	Fraction of the draws held out for validation. (Default: 0.1)
    seed : int
    	Seed of the split, dequantization and batch order. (Default: 0)
    progress : bool
    	Show an epoch progress bar and log the result (each epoch at DEBUG). (Default: True)
    draws : numpy.ndarray or None
    	(M,) hyperprior draw of each row; None treats rows with equal contexts as one draw. (Default: None)
    floor : float
    	C of x = log10(S_gw + C); bins with S_gw = 0 are dequantized into the slab below log10 C, afresh
    	each epoch (fixed for the validation set). 0 is the pre-Stage-3 log10 S_gw. (Default: NOISE_FLOOR)

    Returns
    ------------------------------
    history : dict
    	Per-epoch 'train' and 'val' losses, 'best_epoch' and 'best_val' (NaN val losses without a validation set).

    '''
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = flow.device
    val, trn = split_by_draw(context_draws(context) if draws is None else draws, val_frac, rng)
    n_val = val.size
    flow.fit_transforms(context[trn], N[trn], S[trn], rng, floor=floor)

    c_all = flow._context(context)
    N_t = torch.as_tensor(np.asarray(N), dtype=torch.float64, device=dev)
    S_t = torch.as_tensor(np.asarray(S), dtype=torch.float64, device=dev)
    ## slab positions for the zero bins (drawn only when there are zeros, so zero-free training is unchanged)
    has_zero = bool(np.any(np.asarray(S) == 0))
    slab_v = lambda n: torch.as_tensor(rng.uniform(size=(n, flow.band.nf)), device=dev) if has_zero else None
    ## fixed dequantization for the validation loss, so it is comparable across epochs
    z_val = flow.transform(N_t[val] + torch.as_tensor(rng.uniform(size=n_val), device=dev), S_t[val], slab_v(n_val))[0]

    opt = torch.optim.Adam(flow.flow.parameters(), lr=lr)
    history = dict(train=[], val=[])
    snapshot = lambda: {k: v.detach().clone() for k, v in flow.flow.state_dict().items()}
    best, best_epoch, best_val = None, None, np.inf
    t0 = time.time()
    bar = progress_bar(range(n_epochs), desc=f"band {flow.band.index}", unit='epoch', enabled=progress, leave=False)
    for epoch in bar:
        u = torch.as_tensor(rng.uniform(size=len(trn)), device=dev)
        z = flow.transform(N_t[trn] + u, S_t[trn], slab_v(len(trn)))[0].to(torch.float32)
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
        ## NaN never compares lower, so without a validation set no epoch is "best"
        if v < best_val:
            best, best_epoch, best_val = snapshot(), epoch, v
        if progress:
            bar.set_postfix(train=f"{history['train'][-1]:.4f}", val=f"{v:.4f}", best=f"{best_val:.4f}", refresh=False)
            logger.debug("band %d epoch %d: train %.4f, val %.4f (best %.4f)", flow.band.index, epoch,
                         history['train'][-1], v, best_val)
    flow.final_state = snapshot()
    if best is None:
        best_epoch, best_val = n_epochs - 1, float('nan')
    else:
        flow.flow.load_state_dict(best)
    history.update(best_epoch=int(best_epoch), best_val=float(best_val))
    if progress:
        logger.info("band %d: %d epochs on %d rows in %s; best val %.4f at epoch %d", flow.band.index, n_epochs,
                    len(trn), format_seconds(time.time() - t0), best_val, best_epoch)
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

    @property
    def floor(self):
        '''C of x = log10(S_gw + C), shared by every band (0 for emulators saved before Stage 3).'''
        floors = {float(f.transform.floor) for f in self.flows}
        if len(floors) != 1:
            raise ValueError("bands have different floors: {}".format(sorted(floors)))
        return floors.pop()

    def log_prob(self, context, N_bands, S, n_quad=32, n_qmc=QMC_POINTS):
        '''
        Sum over bands of log P(N_band, S_band | context), shape (M,), with a point mass at S_gw = 0
        (BandFlow.log_prob). context (M, 8); N_bands (n_bands,) or (M, n_bands); S (Nf',) or (M, Nf') on fbins[1:].
        '''
        context = np.atleast_2d(context)
        M = context.shape[0]
        N_bands = np.array(np.broadcast_to(np.asarray(N_bands), (M, len(self.bands))))
        S = np.array(np.broadcast_to(np.asarray(S, dtype=np.float64), (M, self.fs.size)))
        total = 0
        with torch.no_grad():
            for j, f in enumerate(self.flows):
                total = total + f.log_prob(context, N_bands[:, j], S[:, f.band.slice], n_quad=n_quad, n_qmc=n_qmc)
        return total.cpu().numpy()

    def band_log_prob(self, j, context, N, S, n_quad=32, n_qmc=QMC_POINTS):
        '''log P(N, S | context) for band j alone, shape (M,), as numpy.'''
        with torch.no_grad():
            return self.flows[j].log_prob(np.atleast_2d(context), np.asarray(N), np.asarray(S, dtype=np.float64),
                                          n_quad=n_quad, n_qmc=n_qmc).cpu().numpy()

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
        '''

        Save the emulator: emulator.json (bands, resolved architecture), emulator.pt (the current,
        i.e. best-validation, weights) and, after training, emulator_final.pt (last-epoch weights).

        Arguments
        ------------------------------
        directory : str
        	Output directory, created if needed.

        Returns
        ------------------------------
        None

        '''
        os.makedirs(directory, exist_ok=True)
        meta = dict(flow_base='zuko', format=_FORMAT, fbins=self.fbins.tolist(), bands=[b.to_dict() for b in self.bands],
                    flow_kwargs=self.flows[0].arch(), context_names=CONTEXT_NAMES)
        with open(os.path.join(directory, 'emulator.json'), 'w') as fh:
            json.dump(meta, fh, indent=1)
        state = self.state_dict()
        torch.save(state, os.path.join(directory, 'emulator.pt'))
        final_path = os.path.join(directory, 'emulator_final.pt')
        if all(f.final_state is not None for f in self.flows):
            ## the best-weights state with each band's flow parameters replaced by its last epoch's
            for j, f in enumerate(self.flows):
                state.update({'flows.{}.flow.{}'.format(j, k): v for k, v in f.final_state.items()})
            torch.save(state, final_path)
        elif os.path.exists(final_path):
            ## never leave final weights from another emulator beside these
            os.remove(final_path)

    @classmethod
    def load(cls, directory, device='cpu', weights='best'):
        '''

        Load an emulator saved by save().

        Arguments
        ------------------------------
        directory : str
        	Emulator directory.
        device : str
        	torch device. (Default: 'cpu')
        weights : str
        	'best' (emulator.pt) or 'final' (emulator_final.pt). (Default: 'best')

        Returns
        ------------------------------
        emulator : BandedFlowEmulator
        	The emulator, in eval mode.

        '''
        with open(os.path.join(directory, 'emulator.json')) as fh:
            meta = json.load(fh)
        if meta.get('format') not in _FORMATS_LOADABLE:
            raise ValueError("{} is not a {} emulator (saved by an older pelargir?); retrain it".format(
                directory, _FORMAT))
        name = {'best': 'emulator.pt', 'final': 'emulator_final.pt'}[weights]
        fbins = np.asarray(meta['fbins'])
        bands = [FrequencyBand(b['index'], b['start'], b['stop'], fbins[1:][b['start']:b['stop']])
                 for b in meta['bands']]
        em = cls(bands, fbins, **meta['flow_kwargs'])
        state = torch.load(os.path.join(directory, name), map_location=device)
        ## v2 saves predate the floor and slab buffers: they are the floor = 0 (log10 S_gw) transform
        for j in range(len(bands)):
            for buf in ('floor', 'slab'):
                state.setdefault('flows.{}.transform.{}'.format(j, buf), torch.zeros((), dtype=torch.float64))
        em.load_state_dict(state)
        return em.to(device).eval()


def train_emulator(ts, bands=None, device='cpu', flow_kwargs=None, **train_kwargs):
    '''

    Train a BandedFlowEmulator on a TrainingSet, one band at a time, on x = log10(S_gw + C) with
    zero bins dequantized (train_band_flow's floor); each band's validation set holds out whole
    hyperprior draws (TrainingSet.draw_index).

    Arguments
    ------------------------------
    ts : flows.data.TrainingSet
    	Training set.
    bands : list of flows.data.FrequencyBand or None
    	Bands; None uses make_bands(ts.fs). (Default: None)
    device : str
    	torch device. (Default: 'cpu')
    flow_kwargs : dict or None
    	BandFlow arguments (architecture). (Default: None)
    **train_kwargs
    	Passed to train_band_flow.

    Returns
    ------------------------------
    emulator : BandedFlowEmulator
    	The trained emulator (best-validation weights), in eval mode.
    histories : list of dict
    	Per-band train_band_flow histories, plus 'zero_fraction' (the fraction of the band's S_gw values
    	that are 0).

    '''
    bands = make_bands(ts.fs) if bands is None else bands
    em = BandedFlowEmulator(bands, ts.fbins, **(flow_kwargs or {})).to(device)
    draws = ts.draw_index
    histories = []
    for f in progress_bar(em.flows, desc='bands', unit='band', enabled=train_kwargs.get('progress', True)):
        context, N, S = band_view(ts, f.band)
        h = train_band_flow(f, context, N, S, draws=draws, **train_kwargs)
        h['zero_fraction'] = float(np.mean(np.asarray(S) == 0))
        histories.append(h)
    return em.eval(), histories
