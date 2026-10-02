"""
Per-band normalizing-flow emulator of the pelargir population model (arXiv:2604.03390, §6.2).

For each frequency band, a flow models the joint density of the band's resolved-binary count
N_res and its foreground spectrum S_gw (one value per bin), conditioned on the context: the
GalacticBinaryPrior hyperparameters, the SNR threshold rho_thresh, and log10 of the rate
lambda_tot of N_tot ~ Poisson(lambda_tot). With Poisson N_tot and per-bin thresholding,
disjoint bins are independent given the context, so the joint density is the product over bands.

The torch-free data layer (hyperpriors, grids and bands, training sets) lives in flow_data.py and
is re-exported here; flax_flows.py is the JAX flow base built on the same data layer.

Flow choices follow prototype-notebooks/pelargir-v1.0.1-flow-toy-model.ipynb: zuko NSF with 3
transforms and two hidden layers of 10*features, N_res dequantized with U(0,1), N_res, S_gw and
lambda_tot in log10, and every dimension standardized by (x - median)/half-range.
"""
import json
import math
import os

import numpy as np
import torch
import zuko

from .flow_data import (CONTEXT_NAMES, N_POP, RHO_MIN, LAMBDA_RANGE, HYPERPRIOR, DEFAULT_FMIN, DEFAULT_FMAX,
                        DEFAULT_FBIN, DEFAULT_BINS_PER_BAND, DURATION, sample_context, model_fbins, FrequencyBand,
                        make_bands, TrainingSet, band_view, drop_zero_spectra, simulate, draw_training_set,
                        _standardizer, gauss_legendre_01, quadrature_convergence, load_emulator)

# =============================================================================
# Flow layer (torch, zuko)
# =============================================================================

_LN10 = math.log(10.0)


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
        return (torch.as_tensor(np.array(context, dtype=np.float64), device=self.loc.device) - self.loc)/self.scale


class BandFlow(torch.nn.Module):
    '''
    Flow for one band: zuko NSF over z = BandTransform(N_res + u, S_gw), conditioned on the
    standardized context. Runs in float32; transforms are float64.
    '''

    def __init__(self, band, n_context=len(CONTEXT_NAMES), transforms=3, hidden_factor=10, hidden_size=None):
        super().__init__()
        self.band = band
        features = 1 + band.nf
        self.features = features
        self.hidden_factor = hidden_factor
        self.hidden_size = hidden_size
        self.n_transforms = transforms
        ## two hidden layers of hidden_size, or of hidden_factor*features (the notebook's choice)
        width = hidden_factor*features if hidden_size is None else hidden_size
        self.flow = zuko.flows.NSF(features, n_context, transforms=transforms, hidden_features=[width]*2)
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
                    hidden_size=self.hidden_size, transforms=self.n_transforms)


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

    def band_log_prob(self, j, context, N, S, n_quad=32):
        '''log P(N, S | context) for band j alone, shape (M,), as numpy.'''
        with torch.no_grad():
            return self.flows[j].log_prob(np.atleast_2d(context), np.asarray(N), np.asarray(S, dtype=np.float64),
                                          n_quad=n_quad).cpu().numpy()

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
        meta = dict(flow_base='zuko', fbins=self.fbins.tolist(), bands=[b.to_dict() for b in self.bands],
                    flow_kwargs=self.flow_kwargs, context_names=CONTEXT_NAMES)
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
    '''
    Train a BandedFlowEmulator on a TrainingSet; returns (emulator, per-band loss histories). Each
    band drops its rows with S_gw = 0 (flow_data.drop_zero_spectra); history['n_dropped'] counts them.
    '''
    bands = make_bands(ts.fs) if bands is None else bands
    em = BandedFlowEmulator(bands, ts.fbins, **(flow_kwargs or {})).to(device)
    histories = []
    for f in em.flows:
        context, N, S, n_dropped = drop_zero_spectra(*band_view(ts, f.band), band=f.band)
        h = train_band_flow(f, context, N, S, **train_kwargs)
        h['n_dropped'] = n_dropped
        histories.append(h)
    return em.eval(), histories
