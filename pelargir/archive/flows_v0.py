#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flows.py

Normalizing flow utilities for pelargir.

Author: Alexander W. Criswell
"""
import os
import pickle

import numpy as np
import torch
import zuko
from tqdm import tqdm

from utils import get_amp_freq, to_numpy

## flows are run in double precision throughout: the foreground PSD spans
## ~1e-42 to 1e-33, which is subnormal (and so underflows to zero) in float32.
DTYPE = torch.float64

## natural log of ln(10), for the log10 change-of-variables Jacobian
_LOG_LN10 = float(np.log(np.log(10.0)))

## default test configuration: 3 linear bands of 5 bins each on [1e-4,4e-4] Hz
DEFAULT_FMIN = 1e-4
DEFAULT_FMAX = 4e-4
DEFAULT_BIN_WIDTH = 2e-5
DEFAULT_NBANDS = 3

## quadrature nodes used to marginalize the count dequantization out of the
## joint flow when evaluating its log density
DEFAULT_NQUAD = 5

## resolved GB parameter names and astrophysical bounds, in [Msun,Msun,kpc,AU].
## these match the 'default' bounds of utils.apply_theta_lims().
GB_PARAM_NAMES = ['m_1', 'm_2', 'd_L', 'a']
GB_PARAM_BOUNDS = np.array([[0.17, 1.44], [0.17, 1.44], [1e-3, 100.0], [1e-4, 1e-2]])
## which GB parameters are modelled in log10 space
GB_PARAM_LOG10 = np.array([False, False, True, True])


def make_fbins(fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX, bin_width=DEFAULT_BIN_WIDTH):
    """
    Build a coarse-grained frequency bin array for the population model.

    One extra bin is appended beyond fmax, because PopModel.run_model() discards
    its lowest foreground bin and so returns one fewer bin than len(fbins).
    This makes the usable (returned) bins span exactly [fmin,fmax).

    Parameters
    ----------
    fmin : float, optional
        Lowest bin centre frequency in Hz. The default is 1e-4.
    fmax : float, optional
        Upper edge of the usable band in Hz. The default is 4e-4.
    bin_width : float, optional
        Coarse-grained bin width in Hz. The default is 2e-5.

    Returns
    -------
    fbins : array
        Frequency bin centres, of length (fmax-fmin)/bin_width + 1.

    """

    n_usable = int(round((fmax - fmin) / bin_width))

    return fmin + bin_width * np.arange(n_usable + 1)


def model_frequencies(fbins):
    """
    Physical bin centre frequencies of the foreground PSD returned by
    PopModel.run_model().

    The thresher bins binaries with xp.digitize() against the bin edges, so a
    binary at fbins[k] lands in f_idx = k+1, and element ii of the thresher's
    foreground array holds the power at fbins[ii-1]. run_model() discards
    element 0 (which collects everything below the band) and so returns an
    array whose element j is the power at fbins[j] -- NOT at fbins[j+1], which
    is what run_model() returns as its accompanying frequency array. Bands are
    therefore defined against these frequencies rather than run_model()'s.

    Parameters
    ----------
    fbins : array
        Frequency bin centres handed to PopModel.

    Returns
    -------
    fs : array
        Physical bin centres of the returned foreground PSD, of length len(fbins)-1.

    """

    return np.asarray(to_numpy(fbins), dtype=float)[:-1]


class FrequencyBand:
    """
    A contiguous block of coarse-grained frequency bins.
    """

    def __init__(self, index, fs, start, stop):
        """
        Parameters
        ----------
        index : int
            Band number.
        fs : array
            Physical bin centre frequencies of the full foreground PSD, as
            returned by model_frequencies().
        start : int
            First bin index of this band, into fs.
        stop : int
            One past the last bin index of this band, into fs.

        Returns
        -------
        None.

        """

        self.index = index
        self.start = int(start)
        self.stop = int(stop)
        self.fs = np.asarray(fs, dtype=float)[self.start:self.stop]
        self.n_bins = self.stop - self.start

        bin_width = fs[1] - fs[0]
        self.bin_width = bin_width
        self.f_low = self.fs[0] - 0.5 * bin_width
        self.f_high = self.fs[-1] + 0.5 * bin_width

        return

    def slice(self):
        """
        Returns
        -------
        band_slice : slice
            Slice selecting this band from a full-length frequency axis.

        """

        return slice(self.start, self.stop)

    def contains(self, fgw):
        """
        Test which binaries fall inside the band.

        Parameters
        ----------
        fgw : array
            GW frequencies in Hz.

        Returns
        -------
        mask : array
            Boolean mask, True where fgw lies within the band edges.

        """

        fgw = np.asarray(fgw, dtype=float)

        return (fgw >= self.f_low) & (fgw < self.f_high)

    def __repr__(self):
        return "FrequencyBand({}: {:.3e}-{:.3e} Hz, {} bins)".format(self.index, self.f_low,
                                                                     self.f_high, self.n_bins)


def make_bands(fbins, n_bands=DEFAULT_NBANDS):
    """
    Partition the usable frequency bins into contiguous bands of equal width.

    Parameters
    ----------
    fbins : array
        Frequency bin centres handed to PopModel.
    n_bands : int, optional
        Number of bands. The default is 3.

    Returns
    -------
    bands : list of FrequencyBand
        The frequency bands, in ascending frequency order.

    """

    fs = model_frequencies(fbins)
    n_usable = len(fs)

    if n_usable % n_bands != 0:
        raise ValueError("Cannot split {} usable frequency bins into {} equal bands. \
                          Choose an fmin/fmax/bin_width giving a multiple of {} bins.".format(n_usable,
                                                                                              n_bands,
                                                                                              n_bands))

    per_band = n_usable // n_bands

    return [FrequencyBand(ii, fs, ii * per_band, (ii + 1) * per_band) for ii in range(n_bands)]


class ElementwiseTransform(torch.nn.Module):
    """
    Invertible elementwise reparameterization with a tracked log-Jacobian.

    Maps physical quantities onto standardized, unbounded coordinates suitable
    for a spline flow, while retaining the Jacobian needed to report densities
    in the original physical units. This matters here because the flows are
    used as a population-informed prior, so their normalization is physical.

    The composition is, per dimension: an optional log10 (with an additive
    offset, so counts can use offset=1 and keep N=0 finite), an optional logit
    squash onto a bounded box, then an affine whitening fit from data.
    """

    def __init__(self, n_dim, log10=False, offset=0.0, bounds=None, pad=1e-6):
        """
        Parameters
        ----------
        n_dim : int
            Dimensionality of the quantity being transformed.
        log10 : {bool, array}, optional
            Per-dimension flag for log10 scaling. The default is False.
        offset : {float, array}, optional
            Per-dimension additive offset applied inside the log10. The default is 0.
        bounds : array, optional
            Shape (n_dim,2) array of (lower,upper) bounds in *physical* units,
            triggering a logit squash. The default is None (unbounded).
        pad : float, optional
            Fraction of each bound range by which to widen the box, so that
            samples sitting exactly on a boundary do not map to infinity.
            The default is 1e-6.

        Returns
        -------
        None.

        """

        super().__init__()

        self.n_dim = int(n_dim)

        log10_flag = np.broadcast_to(np.asarray(log10, dtype=bool), (self.n_dim,))
        offset_arr = np.broadcast_to(np.asarray(offset, dtype=float), (self.n_dim,))

        self.register_buffer('log10_flag', torch.as_tensor(np.array(log10_flag), dtype=torch.bool))
        self.register_buffer('offset', torch.as_tensor(np.array(offset_arr), dtype=DTYPE))

        if bounds is None:
            self.bounded = False
            self.register_buffer('lo', torch.zeros(self.n_dim, dtype=DTYPE))
            self.register_buffer('hi', torch.ones(self.n_dim, dtype=DTYPE))
        else:
            bounds = np.asarray(bounds, dtype=float).reshape(self.n_dim, 2)
            ## bounds are given physically, so push them through the log10 step
            lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
            lo[log10_flag] = np.log10(lo[log10_flag] + offset_arr[log10_flag])
            hi[log10_flag] = np.log10(hi[log10_flag] + offset_arr[log10_flag])
            span = hi - lo
            lo, hi = lo - pad * span, hi + pad * span
            self.bounded = True
            self.register_buffer('lo', torch.as_tensor(lo, dtype=DTYPE))
            self.register_buffer('hi', torch.as_tensor(hi, dtype=DTYPE))

        ## identity whitening until fit() is called
        self.register_buffer('mu', torch.zeros(self.n_dim, dtype=DTYPE))
        self.register_buffer('sigma', torch.ones(self.n_dim, dtype=DTYPE))

        return

    def _pre_whiten(self, x):
        """
        Apply the log10 and logit steps, accumulating their log-Jacobian.

        Parameters
        ----------
        x : Tensor
            Physical values, of shape (...,n_dim).

        Returns
        -------
        w : Tensor
            Values in pre-whitening coordinates.
        log_det : Tensor
            Summed log|dw/dx| over the trailing axis.

        """

        log_det = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

        shifted = x + self.offset
        ## torch.where evaluates both branches, so guard the log against
        ## non-positive entries in the dimensions that are not log-scaled
        safe = torch.where(self.log10_flag, shifted, torch.ones_like(shifted))
        w = torch.where(self.log10_flag, torch.log10(safe), x)
        log_det = log_det - torch.where(self.log10_flag,
                                        torch.log(safe) + _LOG_LN10,
                                        torch.zeros_like(safe)).sum(-1)

        if self.bounded:
            span = self.hi - self.lo
            s = (w - self.lo) / span
            w = torch.logit(s)
            log_det = log_det - (torch.log(span) + torch.log(s) + torch.log1p(-s)).sum(-1)

        return w, log_det

    def forward(self, x):
        """
        Map physical values to standardized flow coordinates.

        Parameters
        ----------
        x : Tensor
            Physical values, of shape (...,n_dim).

        Returns
        -------
        u : Tensor
            Standardized values.
        log_det : Tensor
            Summed log|du/dx| over the trailing axis.

        """

        w, log_det = self._pre_whiten(x)

        return (w - self.mu) / self.sigma, log_det - torch.log(self.sigma).sum()

    def inverse(self, u):
        """
        Map standardized flow coordinates back to physical values.

        Parameters
        ----------
        u : Tensor
            Standardized values, of shape (...,n_dim).

        Returns
        -------
        x : Tensor
            Physical values.

        """

        w = u * self.sigma + self.mu

        if self.bounded:
            w = self.lo + torch.sigmoid(w) * (self.hi - self.lo)

        return torch.where(self.log10_flag, torch.pow(10.0, w) - self.offset, w)

    def fit(self, x):
        """
        Fit the affine whitening to a sample of physical values.

        Parameters
        ----------
        x : Tensor
            Physical values, of shape (n_samples,n_dim).

        Returns
        -------
        None.

        """

        with torch.no_grad():
            w, _ = self._pre_whiten(x)
            self.mu.copy_(w.mean(0))
            ## guard against constant dimensions
            self.sigma.copy_(torch.clamp(w.std(0), min=1e-8))

        return


class BandFlow(torch.nn.Module):
    """
    The conditional flow model for a single frequency band.

    The population-informed prior is factorized into two conditional densities,
    both conditioned on the population hyperparameters and the SNR threshold,
    c = (Lambda,rho_thresh):

        p(S_gw,N_res,{theta_i} | c) = p(S_gw,N_res | c) prod_i p(theta_i | c)

    S_gw and N_res share a single flow over the concatenated vector
    [S_gw,N_res], rather than getting one each. They are strongly
    anti-correlated at fixed Lambda, since resolving a binary removes its power
    from the foreground; a factorized p(S_gw|c)p(N_res|c) would assert
    independence and so admit high-S_gw/high-N_res states the forward model
    never produces. The effect is negligible where the foreground is built from
    many faint sources and O(1) where it is built from a few loud ones.

    N_res is a count, so its dimension of the joint flow is trained on
    uniformly dequantized values; log_prob() recovers the mixed density
    (continuous in S_gw, a pmf in N_res) by quadrature over the dequantization.
    """

    def __init__(self, band, hpar_names, hidden_features=(64, 64), transforms=3, bins=8):
        """
        Parameters
        ----------
        band : FrequencyBand
            The frequency band this flow models.
        hpar_names : list of str
            Population hyperparameter names, in context order.
        hidden_features : tuple, optional
            Hidden layer widths of the autoregressive networks. The default is (64,64).
        transforms : int, optional
            Number of spline transformations per flow. The default is 3.
        bins : int, optional
            Number of spline bins per transformation. The default is 8.

        Returns
        -------
        None.

        """

        super().__init__()

        self.band = band
        self.hpar_names = list(hpar_names)
        ## context is the hyperparameters plus the SNR threshold
        self.context_names = self.hpar_names + ['snr_thresh']
        self.n_context = len(self.context_names)

        flow_kwargs = dict(context=self.n_context, hidden_features=hidden_features,
                           transforms=transforms, bins=bins)

        ## the joint flow spans [S_gw (n_bins), N_res]; the count dimension is
        ## the trailing one, and carries an offset of 1 so that N_res=0 stays
        ## finite under the log10
        self.n_joint = band.n_bins + 1
        self.count_mask = np.array([False] * band.n_bins + [True])

        self.joint_flow = zuko.flows.NSF(features=self.n_joint, **flow_kwargs).to(DTYPE)
        self.gb_flow = zuko.flows.NSF(features=len(GB_PARAM_NAMES), **flow_kwargs).to(DTYPE)

        self.context_transform = ElementwiseTransform(self.n_context)
        self.joint_transform = ElementwiseTransform(self.n_joint, log10=True,
                                                    offset=np.where(self.count_mask, 1.0, 0.0))
        self.gb_transform = ElementwiseTransform(len(GB_PARAM_NAMES), log10=GB_PARAM_LOG10,
                                                 bounds=GB_PARAM_BOUNDS)

        return

    def fit_transforms(self, context, fg_psd, n_res, gb_theta):
        """
        Fit the preprocessing whitening from a training set. Must be called
        before training.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).
        fg_psd : Tensor
            Foreground PSD in the band, of shape (n_draws,n_bins).
        n_res : Tensor
            Resolved binary counts in the band, of shape (n_draws,1).
        gb_theta : Tensor
            Resolved binary parameters, of shape (n_binaries,4).

        Returns
        -------
        None.

        """

        self.context_transform.fit(context)
        ## fit against the dequantization bin centre, as a representative value
        self.joint_transform.fit(self.stack_joint(fg_psd, n_res + 0.5))
        if len(gb_theta) > 0:
            self.gb_transform.fit(gb_theta)

        return

    def stack_joint(self, fg_psd, n_res):
        """
        Concatenate the foreground PSD and resolved count into the joint vector.

        Parameters
        ----------
        fg_psd : Tensor
            Foreground PSD in the band, of shape (...,n_bins).
        n_res : Tensor
            Resolved binary counts, of shape (...,1).

        Returns
        -------
        joint : Tensor
            Concatenated values, of shape (...,n_bins+1).

        """

        return torch.cat([fg_psd, n_res], dim=-1)

    def split_joint(self, joint):
        """
        Split the joint vector back into foreground PSD and resolved count.

        Parameters
        ----------
        joint : Tensor
            Concatenated values, of shape (...,n_bins+1).

        Returns
        -------
        fg_psd : Tensor
            Foreground PSD, of shape (...,n_bins).
        n_res : Tensor
            Resolved binary counts, of shape (...,1).

        """

        return joint[..., :self.band.n_bins], joint[..., self.band.n_bins:]

    def transform_context(self, context):
        """
        Whiten the conditioning vector.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (...,n_context).

        Returns
        -------
        c : Tensor
            Whitened conditioning values.

        """

        return self.context_transform(context)[0]

    def joint_log_prob(self, context, fg_psd, n_res, n_quad=DEFAULT_NQUAD):
        """
        Log density of the band foreground PSD and resolved count.

        This is a mixed density: continuous in S_gw, with units of Hz^-1 per
        bin, and a probability mass in N_res. The flow itself is trained on
        uniformly dequantized counts, so the count is marginalized back out by
        Gauss-Legendre quadrature over the dequantization interval,

            P(S_gw,N_res | c) = integral_0^1 q(S_gw,N_res+u | c) du

        which costs n_quad flow evaluations. n_quad=1 reduces to the midpoint
        rule.

        The quadrature is not converged at the default node count: the estimate
        moves by ~0.1 nats between 5 and 9 nodes, and by ~1 nat between 1 and 9,
        with the error depending on N_res. That is tolerable while the flow is
        a test case but should be fixed before the density is used as a prior in
        the hierarchical likelihood, since an N_res-dependent offset biases the
        resolved-count direction rather than shifting a constant.

        The fix is to drop the quadrature entirely. N_res is the trailing
        dimension of the joint vector and zuko's NSF is autoregressive with
        randperm=False, so its conditional map given S_gw is a composition of
        monotonic splines and the exact pmf is available as a CDF difference,

            P(S_gw,N_res | c) = p(S_gw | c) [F(N_res+1 | S_gw,c) - F(N_res | S_gw,c)]

        at the cost of two evaluations instead of n_quad, and with no tuning
        constant. It needs zuko's transform internals rather than the public
        log_prob, so it couples to the zuko version and wants its own test that
        the CDF is monotonic and agrees with a brute-force integral.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).
        fg_psd : Tensor
            Foreground PSD in the band, of shape (n_draws,n_bins).
        n_res : Tensor
            Resolved binary counts, of shape (n_draws,1).
        n_quad : int, optional
            Number of quadrature nodes. The default is 5.

        Returns
        -------
        ln_p : Tensor
            Log density, of shape (n_draws,).

        """

        dist = self.joint_flow(self.transform_context(context))

        ## Gauss-Legendre nodes and weights mapped from [-1,1] onto [0,1]
        nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
        nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights

        terms = []
        for node, weight in zip(nodes, weights):
            u, log_det = self.joint_transform(self.stack_joint(fg_psd, n_res + float(node)))
            terms.append(dist.log_prob(u) + log_det + float(np.log(weight)))

        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)

    def gb_log_prob(self, context, gb_theta):
        """
        Log density of resolved binary parameters, in physical units of
        [Msun,Msun,kpc,AU].

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_binaries,n_context). One row per
            binary, i.e. the parent draw's context repeated.
        gb_theta : Tensor
            Resolved binary parameters, of shape (n_binaries,4).

        Returns
        -------
        ln_p : Tensor
            Log density, of shape (n_binaries,).

        """

        u, log_det = self.gb_transform(gb_theta)

        return self.gb_flow(self.transform_context(context)).log_prob(u) + log_det

    def log_prob(self, context, fg_psd, n_res, gb_theta, gb_index=None,
                 n_quad=DEFAULT_NQUAD):
        """
        Total log prior density of the band observables.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).
        fg_psd : Tensor
            Foreground PSD in the band, of shape (n_draws,n_bins).
        n_res : Tensor
            Resolved binary counts, of shape (n_draws,1).
        gb_theta : Tensor
            Resolved binary parameters, of shape (n_binaries,4), flattened
            across draws.
        gb_index : Tensor, optional
            Integer index of shape (n_binaries,) giving the draw each binary
            belongs to. The default is None (all binaries belong to draw 0).
        n_quad : int, optional
            Number of quadrature nodes for the count dimension. The default is 5.

        Returns
        -------
        terms : dict
            The 'joint', 'gb' and 'total' log density contributions, each of
            shape (n_draws,).

        """

        n_draws = context.shape[0]

        ln_p_joint = self.joint_log_prob(context, fg_psd, n_res, n_quad=n_quad)

        ln_p_gb = torch.zeros(n_draws, dtype=context.dtype, device=context.device)
        if len(gb_theta) > 0:
            if gb_index is None:
                gb_index = torch.zeros(len(gb_theta), dtype=torch.long, device=context.device)
            per_binary = self.gb_log_prob(context[gb_index], gb_theta)
            ln_p_gb = ln_p_gb.index_add(0, gb_index, per_binary)

        return {'joint': ln_p_joint, 'gb': ln_p_gb, 'total': ln_p_joint + ln_p_gb}

    def sample_joint(self, context, n_samples=1):
        """
        Draw foreground PSD and resolved count samples, vectorized over
        conditioning vectors.

        The count is recovered from its dequantized value by taking the floor,
        which is the inverse of the U(0,1) dequantization used in training.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).
        n_samples : int, optional
            Number of samples to draw per conditioning vector. The default is 1.

        Returns
        -------
        fg_psd : Tensor
            Foreground PSD in Hz^-1, of shape (n_samples,n_draws,n_bins).
        n_res : Tensor
            Resolved binary counts, of shape (n_samples,n_draws).

        """

        with torch.no_grad():
            c = self.transform_context(context)
            joint = self.joint_transform.inverse(self.joint_flow(c).sample((int(n_samples),)))
            fg_psd, n_raw = self.split_joint(joint)

        return fg_psd, torch.clamp(torch.floor(n_raw.squeeze(-1)), min=0.0)

    def sample(self, context):
        """
        Draw a population-informed prior sample for each conditioning vector.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).

        Returns
        -------
        draws : list of dict
            One dict per conditioning vector, with keys 'fg_psd' (n_bins,) in
            Hz^-1, 'n_res' (int) and 'gb_theta' (n_res,4) in
            [Msun,Msun,kpc,AU].

        """

        with torch.no_grad():
            c = self.transform_context(context)
            fg_psd, n_res = self.sample_joint(context, n_samples=1)
            fg_psd, n_res = fg_psd[0], n_res[0]

            draws = []
            for ii in range(context.shape[0]):
                n_ii = int(n_res[ii].item())
                if n_ii > 0:
                    c_ii = c[ii].expand(n_ii, -1)
                    theta_ii = self.gb_transform.inverse(self.gb_flow(c_ii).sample())
                else:
                    theta_ii = torch.zeros((0, len(GB_PARAM_NAMES)), dtype=context.dtype,
                                           device=context.device)
                draws.append({'fg_psd': fg_psd[ii], 'n_res': n_ii, 'gb_theta': theta_ii})

        return draws


class BandedFlowPrior(torch.nn.Module):
    """
    A population-informed prior built from one BandFlow per frequency band.

    Maps a draw of the population hyperparameters and the SNR threshold onto a
    density over the band observables (S_gw, N_res, theta_GB).
    """

    def __init__(self, bands, hpar_names, **flow_kwargs):
        """
        Parameters
        ----------
        bands : list of FrequencyBand
            The frequency bands to model.
        hpar_names : list of str
            Population hyperparameter names, in context order.
        flow_kwargs : dict, optional
            Keyword arguments passed to each BandFlow.

        Returns
        -------
        None.

        """

        super().__init__()

        self.bands = list(bands)
        self.hpar_names = list(hpar_names)
        self.context_names = self.hpar_names + ['snr_thresh']
        self.n_context = len(self.context_names)
        self.band_flows = torch.nn.ModuleList([BandFlow(band, self.hpar_names, **flow_kwargs)
                                               for band in self.bands])

        return

    def build_context(self, pop_theta, snr_thresh):
        """
        Assemble the conditioning vector from hyperparameters and SNR threshold.

        Parameters
        ----------
        pop_theta : {dict, array}
            Population hyperparameters. A dict of {name:value}, or an array of
            shape (n_draws,n_hyper) ordered as hpar_names.
        snr_thresh : {float, array}
            SNR threshold(s) used to define resolved vs. unresolved.

        Returns
        -------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).

        """

        if isinstance(pop_theta, dict):
            cols = [np.atleast_1d(to_numpy(pop_theta[name])).ravel() for name in self.hpar_names]
            hyper = np.stack(cols, axis=-1)
        else:
            hyper = np.atleast_2d(np.asarray(to_numpy(pop_theta), dtype=float))

        if hyper.shape[-1] != len(self.hpar_names):
            raise ValueError("Expected {} hyperparameters ordered as {}, got shape {}.".format(
                len(self.hpar_names), self.hpar_names, hyper.shape))

        rho = np.broadcast_to(np.atleast_1d(np.asarray(to_numpy(snr_thresh), dtype=float)),
                              (hyper.shape[0],))

        return torch.as_tensor(np.concatenate([hyper, np.array(rho)[:, None]], axis=-1), dtype=DTYPE)

    def log_prob(self, context, band_data, n_quad=DEFAULT_NQUAD):
        """
        Total log prior density summed over bands.

        Parameters
        ----------
        context : Tensor
            Conditioning values, of shape (n_draws,n_context).
        band_data : list of dict
            Per-band observables, each with keys 'fg_psd', 'n_res', 'gb_theta'
            and optionally 'gb_index', as returned by draw_training_set().
        n_quad : int, optional
            Number of quadrature nodes for the count dimension. The default is 5.

        Returns
        -------
        ln_p : Tensor
            Total log density, of shape (n_draws,).
        per_band : list of dict
            The per-band term breakdown from BandFlow.log_prob().

        """

        if len(band_data) != len(self.band_flows):
            raise ValueError("Got data for {} bands but this prior models {}.".format(
                len(band_data), len(self.band_flows)))

        per_band = [flow.log_prob(context, data['fg_psd'], data['n_res'], data['gb_theta'],
                                  gb_index=data.get('gb_index'), n_quad=n_quad)
                    for flow, data in zip(self.band_flows, band_data)]

        return sum(terms['total'] for terms in per_band), per_band

    def sample(self, pop_theta, snr_thresh):
        """
        Draw a population-informed prior sample in every band.

        Parameters
        ----------
        pop_theta : {dict, array}
            Population hyperparameters.
        snr_thresh : {float, array}
            SNR threshold(s).

        Returns
        -------
        band_draws : list of list of dict
            Per-band lists of draws, as returned by BandFlow.sample().

        """

        context = self.build_context(pop_theta, snr_thresh)

        return [flow.sample(context) for flow in self.band_flows]

    def save(self, path):
        """
        Write the trained prior to disk.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        None.

        """

        config = {'hpar_names': self.hpar_names,
                  'bands': [(b.index, b.start, b.stop, b.fs.tolist(), b.bin_width)
                            for b in self.bands],
                  'state_dict': self.state_dict()}
        with open(path, 'wb') as f:
            pickle.dump(config, f)

        return


def _draw_realizations(popmodel, bands, pop_theta, n_realizations, sink, row_offset, draw):
    """
    Run the forward model n_realizations times at fixed hyperparameters and
    append one row per realization to an accumulator.

    Parameters
    ----------
    popmodel : PopModel
        The population model to sample. popmodel.thresh_val must already be set.
    bands : list of FrequencyBand
        The frequency bands to record.
    pop_theta : dict
        The population hyperparameter draw, as {name:value}.
    n_realizations : int
        Number of independent realizations to draw at this pop_theta.
    sink : dict
        Accumulator of lists, as built by _new_sink().
    row_offset : int
        Index of the first row this call will write.
    draw : int
        Hyperparameter draw number, recorded in sink['draw_index'].

    Returns
    -------
    n_rows : int
        Number of rows appended.

    """

    n_par = len(GB_PARAM_NAMES)

    for realization in range(n_realizations):
        _, fg_psd, _, res_idx, galaxy_draw = popmodel.run_model(pop_theta, return_extras=True)

        fg_psd = np.asarray(to_numpy(fg_psd), dtype=float).ravel()
        galaxy_draw = np.asarray(to_numpy(galaxy_draw), dtype=float)
        if galaxy_draw.shape[0] != n_par:
            raise RuntimeError("Expected galaxy_draw with leading axis {} ({}), got shape {}.".format(
                n_par, GB_PARAM_NAMES, galaxy_draw.shape))
        galaxy_draw = galaxy_draw.reshape(n_par, -1)

        ## recover the resolved binaries and assign them to bands by their own
        ## GW frequency, which avoids relying on the thresher's flattened
        ## per-bin index bookkeeping
        res_idx = np.asarray([int(to_numpy(ii)) for ii in res_idx], dtype=int)
        res_theta = galaxy_draw[:, res_idx]
        _, res_fgw = get_amp_freq(res_theta)
        res_fgw = np.asarray(to_numpy(res_fgw), dtype=float).ravel()

        row = row_offset + realization
        sink['draw_index'].append(draw)

        for jj, band in enumerate(bands):
            sink['fg'][jj].append(fg_psd[band.slice()])
            in_band = band.contains(res_fgw)
            sink['n'][jj].append([in_band.sum()])
            if in_band.any():
                sink['gb'][jj].append(res_theta[:, in_band].T)
                sink['gb_index'][jj].append(np.full(int(in_band.sum()), row, dtype=int))

    return n_realizations


def _new_sink(bands):
    """
    Build an empty accumulator for _draw_realizations().

    Parameters
    ----------
    bands : list of FrequencyBand
        The frequency bands to record.

    Returns
    -------
    sink : dict
        Empty accumulator lists.

    """

    return {'context': [], 'draw_index': [],
            'fg': [[] for band in bands], 'n': [[] for band in bands],
            'gb': [[] for band in bands], 'gb_index': [[] for band in bands]}


def _finalize_sink(sink, bands, hpar_names):
    """
    Convert an accumulator into the tensor dict returned by the draw functions.

    Parameters
    ----------
    sink : dict
        Populated accumulator.
    bands : list of FrequencyBand
        The frequency bands recorded.
    hpar_names : list of str
        Population hyperparameter names, in context order.

    Returns
    -------
    dataset : dict
        With keys 'context', 'context_names', 'draw_index' and 'bands'.

    """

    n_par = len(GB_PARAM_NAMES)

    band_sets = []
    for jj, band in enumerate(bands):
        gb = sink['gb'][jj]
        idx = sink['gb_index'][jj]
        band_sets.append({
            'fg_psd': torch.as_tensor(np.array(sink['fg'][jj]), dtype=DTYPE),
            'n_res': torch.as_tensor(np.array(sink['n'][jj], dtype=float), dtype=DTYPE),
            'gb_theta': torch.as_tensor(np.concatenate(gb, axis=0) if len(gb)
                                        else np.zeros((0, n_par)), dtype=DTYPE),
            'gb_index': torch.as_tensor(np.concatenate(idx) if len(idx) else np.zeros(0),
                                        dtype=torch.long),
        })

    return {'context': torch.as_tensor(np.array(sink['context']), dtype=DTYPE),
            'context_names': list(hpar_names) + ['snr_thresh'],
            'draw_index': torch.as_tensor(np.array(sink['draw_index']), dtype=torch.long),
            'bands': band_sets}


def draw_training_set(popmodel, n_draws, bands, n_realizations=1, snr_thresh_range=(3.0, 10.0),
                      rng=None, progress=True):
    """
    Run the population forward model repeatedly to build a flow training set.

    Each draw samples the population hyperparameters from the model's attached
    hyperprior and the SNR threshold from a uniform range, then records
    n_realizations independent realizations of the band observables at that
    fixed context. Rows are flattened to n_rows = n_draws*n_realizations, with
    the context row repeated verbatim across a draw's realizations, so the
    training code needs no special handling.

    n_realizations=1 is the right default for training. Conditional maximum
    likelihood is already consistent with one realization per context, so
    repeats do not improve estimation of the conditional; they only change the
    design measure, and at fixed compute they trade coverage of the context
    space for clustered draws. They also cost a proportional number of forward
    model calls with no vectorization saving, since realizations are generated
    by looping rather than via PopModel's own Nreal (the thresher loses
    resolved binary identity for Nreal>1). Their use is
    draw_diagnostic_set(), which needs many realizations at one context.

    Per-row log densities are deliberately not weighted by 1/n_realizations:
    every row is a genuine draw from p(observables|context) and the training
    loss already averages over rows. That holds only while n_realizations is
    constant across draws, which it is here.

    Requires Nreal=1 on the model itself, since the thresher only tracks
    resolved binary indices for a single realization.

    Parameters
    ----------
    popmodel : PopModel
        The population model to sample. Passed in rather than constructed here,
        so that this module does not depend on the GPU/Eryn stack.
    n_draws : int
        Number of hyperparameter draws.
    bands : list of FrequencyBand
        The frequency bands to record.
    n_realizations : int, optional
        Number of forward model realizations per hyperparameter draw. The
        default is 1.
    snr_thresh_range : tuple, optional
        Uniform (low,high) range for the SNR threshold. The default is (3,10).
    rng : Generator, optional
        RNG for the SNR threshold draws. The default is None.
    progress : bool, optional
        Whether to show a progress bar. The default is True.

    Returns
    -------
    training_set : dict
        With keys 'context' of shape (n_rows,n_context), 'context_names',
        'draw_index' of shape (n_rows,), and 'bands', a list of per-band dicts
        holding 'fg_psd' (n_rows,n_bins), 'n_res' (n_rows,1), 'gb_theta'
        (n_binaries,4) and 'gb_index' (n_binaries,) indexing into n_rows.

    """

    if popmodel.Nreal != 1:
        raise ValueError("Training data generation requires Nreal=1, got {}. The thresher \
                          only tracks resolved binary indices for a single realization.".format(popmodel.Nreal))

    if rng is None:
        rng = np.random.default_rng()

    hpar_names = popmodel.hpar_names
    original_thresh = popmodel.thresh_val

    sink = _new_sink(bands)
    row = 0

    iterator = tqdm(range(n_draws), desc='drawing training set') if progress else range(n_draws)

    try:
        for draw in iterator:
            rho = float(rng.uniform(*snr_thresh_range))
            popmodel.thresh_val = rho
            pop_theta = popmodel.hyperprior.sample(1)

            context_row = [float(to_numpy(np.atleast_1d(pop_theta[name])).ravel()[0])
                           for name in hpar_names] + [rho]
            sink['context'].extend([context_row] * n_realizations)

            row += _draw_realizations(popmodel, bands, pop_theta, n_realizations, sink, row, draw)
    finally:
        popmodel.thresh_val = original_thresh

    return _finalize_sink(sink, bands, hpar_names)


def draw_diagnostic_set(popmodel, bands, pop_theta_grid, snr_thresh, n_realizations=200,
                        progress=True):
    """
    Build a fixed-hyperparameter diagnostic set for validating the learned
    conditional spread.

    Unlike draw_training_set(), the hyperparameters are given explicitly rather
    than sampled, so many realizations share exactly one context and the
    empirical scatter at that context can be measured directly. Pair with
    compare_conditional_scatter().

    Parameters
    ----------
    popmodel : PopModel
        The population model to sample.
    bands : list of FrequencyBand
        The frequency bands to record.
    pop_theta_grid : {array, list of dict}
        Hyperparameter values to evaluate at. An array of shape
        (n_contexts,n_hyper) ordered as popmodel.hpar_names, or a list of
        {name:value} dicts.
    snr_thresh : {float, array}
        SNR threshold, either shared or one per context.
    n_realizations : int, optional
        Number of realizations per context. The default is 200.
    progress : bool, optional
        Whether to show a progress bar. The default is True.

    Returns
    -------
    diagnostic_set : dict
        Same structure as draw_training_set(), with n_rows =
        n_contexts*n_realizations.

    """

    if popmodel.Nreal != 1:
        raise ValueError("Diagnostic data generation requires Nreal=1, got {}.".format(popmodel.Nreal))

    hpar_names = popmodel.hpar_names
    original_thresh = popmodel.thresh_val

    if isinstance(pop_theta_grid, (list, tuple)) and len(pop_theta_grid) \
            and isinstance(pop_theta_grid[0], dict):
        grid = [[float(to_numpy(np.atleast_1d(d[name])).ravel()[0]) for name in hpar_names]
                for d in pop_theta_grid]
        grid = np.array(grid)
    else:
        grid = np.atleast_2d(np.asarray(to_numpy(pop_theta_grid), dtype=float))

    if grid.shape[-1] != len(hpar_names):
        raise ValueError("Expected {} hyperparameters ordered as {}, got shape {}.".format(
            len(hpar_names), hpar_names, grid.shape))

    rho_grid = np.broadcast_to(np.atleast_1d(np.asarray(to_numpy(snr_thresh), dtype=float)),
                               (grid.shape[0],))

    sink = _new_sink(bands)
    row = 0

    iterator = tqdm(range(grid.shape[0]), desc='drawing diagnostic set') if progress \
        else range(grid.shape[0])

    try:
        for draw in iterator:
            rho = float(rho_grid[draw])
            popmodel.thresh_val = rho
            pop_theta = {name: np.atleast_1d(grid[draw, ii])
                         for ii, name in enumerate(hpar_names)}

            sink['context'].extend([list(grid[draw]) + [rho]] * n_realizations)

            row += _draw_realizations(popmodel, bands, pop_theta, n_realizations, sink, row, draw)
    finally:
        popmodel.thresh_val = original_thresh

    return _finalize_sink(sink, bands, hpar_names)


def train_component(flow, transform, context, values, n_epochs=200, batch_size=128, lr=1e-3,
                    weight_decay=0.0, dequantize=False, rng=None, progress=True, desc='flow',
                    val_values=None, val_context=None):
    """
    Fit a single conditional flow by stochastic maximum likelihood.

    The loss is the negative mean log density in *standardized* coordinates;
    the preprocessing Jacobian is a constant offset in the flow parameters and
    so does not affect the optimum.

    Parameters
    ----------
    flow : LazyDistribution
        The zuko flow to train.
    transform : ElementwiseTransform
        Preprocessing for the target values. Must already be fit.
    context : Tensor
        Whitened conditioning values, of shape (n_samples,n_context).
    values : Tensor
        Physical target values, of shape (n_samples,n_features).
    n_epochs : int, optional
        Number of passes over the training set. The default is 200.
    batch_size : int, optional
        Minibatch size. The default is 128.
    lr : float, optional
        Adam learning rate. The default is 1e-3.
    weight_decay : float, optional
        Adam weight decay. The default is 0.
    dequantize : {bool, array}, optional
        Which dimensions to add U(0,1) noise to, for discrete counts. Either a
        scalar flag or a boolean array over features. The default is False.
    rng : Generator, optional
        RNG for dequantization and shuffling. The default is None.
    progress : bool, optional
        Whether to show a progress bar. The default is True.
    desc : str, optional
        Progress bar label. The default is 'flow'.
    val_values : Tensor, optional
        Held-out target values. The default is None (no validation).
    val_context : Tensor, optional
        Held-out whitened conditioning values. The default is None.

    Returns
    -------
    losses : array
        Mean negative log density per epoch.
    val_losses : array
        Mean held-out negative log density per epoch, empty if no validation
        data was given.

    """

    if len(values) == 0:
        raise ValueError("No training values given for {}.".format(desc))

    if rng is None:
        rng = np.random.default_rng()

    ## a scalar flag applies to every dimension; an array selects dimensions,
    ## which is what the joint flow needs since only its count dimension is
    ## discrete
    deq_mask = np.broadcast_to(np.asarray(dequantize, dtype=bool), (values.shape[-1],))
    deq_mask = torch.as_tensor(np.array(deq_mask), dtype=values.dtype)
    any_deq = bool(deq_mask.any())

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    n_samples = len(values)
    validate = val_values is not None and len(val_values) > 0
    losses, val_losses = [], []

    iterator = tqdm(range(n_epochs), desc=desc) if progress else range(n_epochs)

    for epoch in iterator:
        order = torch.as_tensor(rng.permutation(n_samples), dtype=torch.long)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]
            target = values[idx]
            if any_deq:
                target = target + deq_mask * torch.as_tensor(rng.uniform(size=tuple(target.shape)),
                                                             dtype=target.dtype)

            u, _ = transform(target)
            loss = -flow(context[idx]).log_prob(u).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

        if validate:
            with torch.no_grad():
                target = val_values
                if any_deq:
                    target = target + deq_mask * torch.as_tensor(
                        rng.uniform(size=tuple(target.shape)), dtype=target.dtype)
                u, _ = transform(target)
                val_losses.append(float(-flow(val_context).log_prob(u).mean().item()))

        if progress:
            postfix = {'loss': '{:.4f}'.format(losses[-1])}
            if validate:
                postfix['val'] = '{:.4f}'.format(val_losses[-1])
            iterator.set_postfix(**postfix)

    return np.array(losses), np.array(val_losses)


def train_banded_prior(prior, training_set, n_epochs=200, batch_size=128, lr=1e-3, rng=None,
                       progress=True, val_fraction=0.0):
    """
    Fit every component flow of a BandedFlowPrior to a training set.

    Parameters
    ----------
    prior : BandedFlowPrior
        The prior to train, modified in place.
    training_set : dict
        Training data, as returned by draw_training_set().
    n_epochs : int, optional
        Number of passes over the training set per component. The default is 200.
    batch_size : int, optional
        Minibatch size. The default is 128.
    lr : float, optional
        Adam learning rate. The default is 1e-3.
    rng : Generator, optional
        RNG for dequantization, shuffling and the validation split. The default
        is None.
    progress : bool, optional
        Whether to show progress bars. The default is True.
    val_fraction : float, optional
        Fraction of hyperparameter draws to hold out for validation. The split
        is on unique draw_index values rather than on rows, since rows sharing a
        draw share a context exactly and would otherwise leak across the split.
        The default is 0 (no validation). Note that NLL curves are not
        comparable across different n_realizations, which change the design
        measure over the context space.

    Returns
    -------
    history : list of dict
        Per-band dicts with 'joint' and 'gb' training loss curves, and
        'joint_val'/'gb_val' held-out curves when val_fraction > 0.

    """

    if training_set['context_names'] != prior.context_names:
        raise ValueError("Training set context {} does not match the prior's {}.".format(
            training_set['context_names'], prior.context_names))

    if rng is None:
        rng = np.random.default_rng()

    context = training_set['context']
    n_rows = context.shape[0]

    ## hold out whole hyperparameter draws, so repeated realizations of one
    ## context never straddle the split
    if val_fraction > 0:
        draw_index = training_set.get('draw_index')
        if draw_index is None:
            raise ValueError("val_fraction > 0 requires a 'draw_index' in the training set.")
        draw_index = np.asarray(to_numpy(draw_index), dtype=int)
        unique = np.unique(draw_index)
        n_val = max(int(round(val_fraction * len(unique))), 1)
        val_draws = rng.choice(unique, size=n_val, replace=False)
        is_val = np.isin(draw_index, val_draws)
    else:
        is_val = np.zeros(n_rows, dtype=bool)

    train_rows = torch.as_tensor(np.flatnonzero(~is_val), dtype=torch.long)
    val_rows = torch.as_tensor(np.flatnonzero(is_val), dtype=torch.long)
    validate = len(val_rows) > 0

    history = []

    for band_flow, data in zip(prior.band_flows, training_set['bands']):
        ## fit the whitening on the training rows only
        band_flow.fit_transforms(context[train_rows], data['fg_psd'][train_rows],
                                 data['n_res'][train_rows], data['gb_theta'])
        c = band_flow.transform_context(context).detach()
        label = 'band {}'.format(band_flow.band.index)

        joint = band_flow.stack_joint(data['fg_psd'], data['n_res'])
        deq_mask = band_flow.count_mask

        band_history = {}
        band_history['joint'], band_history['joint_val'] = train_component(
            band_flow.joint_flow, band_flow.joint_transform, c[train_rows], joint[train_rows],
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, dequantize=deq_mask, rng=rng,
            progress=progress, desc=label + ' joint',
            val_values=joint[val_rows] if validate else None,
            val_context=c[val_rows] if validate else None)

        gb_index = data['gb_index']
        if len(data['gb_theta']) > 0:
            gb_is_val = torch.as_tensor(is_val, dtype=torch.bool)[gb_index]
            gb_train = torch.logical_not(gb_is_val)
            band_history['gb'], band_history['gb_val'] = train_component(
                band_flow.gb_flow, band_flow.gb_transform, c[gb_index[gb_train]],
                data['gb_theta'][gb_train], n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                rng=rng, progress=progress, desc=label + ' gb',
                val_values=data['gb_theta'][gb_is_val] if validate else None,
                val_context=c[gb_index[gb_is_val]] if validate else None)
        else:
            band_history['gb'], band_history['gb_val'] = np.array([]), np.array([])

        history.append(band_history)

    return history


def compare_conditional_scatter(prior, diagnostic_set, n_flow_samples=2000):
    """
    Compare the flow's conditional spread against the forward model's empirical
    spread, at each fixed context of a diagnostic set.

    This is the acceptance test for the joint (S_gw,N_res) flow. The
    correlation columns are the point of interest: the foreground and the
    resolved count are anti-correlated at fixed hyperparameters, since
    resolving a binary removes its power from the foreground, and the flow
    should reproduce that. Expect a correlation near zero where the foreground
    is built from many faint sources and clearly negative where it is built
    from a few loud ones.

    Parameters
    ----------
    prior : BandedFlowPrior
        The trained prior.
    diagnostic_set : dict
        Fixed-context data, as returned by draw_diagnostic_set().
    n_flow_samples : int, optional
        Number of flow samples to draw per context. The default is 2000.

    Returns
    -------
    rows : list of dict
        One entry per (band,context) pair, with the empirical and flow-sampled
        mean and standard deviation of N_res and of log10 S_gw, and the
        correlation between total band power and N_res under each.

    """

    context = diagnostic_set['context']
    draw_index = np.asarray(to_numpy(diagnostic_set['draw_index']), dtype=int)

    def _stats(fg_psd, n_res):
        ## fg_psd (n_samples,n_bins), n_res (n_samples,)
        log_psd = np.log10(np.clip(fg_psd, 1e-300, None))
        total = fg_psd.sum(axis=-1)
        if n_res.std() > 0 and total.std() > 0:
            corr = float(np.corrcoef(total, n_res)[0, 1])
        else:
            ## degenerate at this context; a correlation is not defined
            corr = float('nan')
        return {'n_res_mean': float(n_res.mean()), 'n_res_std': float(n_res.std()),
                'log10_psd_mean': log_psd.mean(axis=0), 'log10_psd_std': log_psd.std(axis=0),
                'corr_power_nres': corr}

    rows = []

    for draw in np.unique(draw_index):
        mask = draw_index == draw
        idx = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
        context_row = context[idx[:1]]

        for band_flow, data in zip(prior.band_flows, diagnostic_set['bands']):
            empirical = _stats(np.asarray(to_numpy(data['fg_psd'][idx]), dtype=float),
                               np.asarray(to_numpy(data['n_res'][idx]), dtype=float).ravel())

            fg_flow, n_flow = band_flow.sample_joint(context_row, n_samples=n_flow_samples)
            sampled = _stats(np.asarray(to_numpy(fg_flow[:, 0, :]), dtype=float),
                             np.asarray(to_numpy(n_flow[:, 0]), dtype=float))

            rows.append({'band': band_flow.band.index, 'draw': int(draw),
                         'n_realizations': int(mask.sum()),
                         'context': np.asarray(to_numpy(context_row[0]), dtype=float),
                         'empirical': empirical, 'flow': sampled})

    return rows


def load_banded_prior(path, **flow_kwargs):
    """
    Load a trained prior written by BandedFlowPrior.save().

    Parameters
    ----------
    path : str
        Input file path.
    flow_kwargs : dict, optional
        Keyword arguments passed to each BandFlow. Must match those used when
        the prior was built.

    Returns
    -------
    prior : BandedFlowPrior
        The restored prior.

    """

    with open(path, 'rb') as f:
        config = pickle.load(f)

    bands = []
    for index, start, stop, fs, bin_width in config['bands']:
        fs = np.asarray(fs, dtype=float)
        ## reconstruct the full-length frequency axis this band was cut from
        full_fs = fs[0] - bin_width * start + bin_width * np.arange(stop + len(fs))
        bands.append(FrequencyBand(index, full_fs, start, stop))

    prior = BandedFlowPrior(bands, config['hpar_names'], **flow_kwargs)
    prior.load_state_dict(config['state_dict'])

    return prior
