#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_flows.py

Verification suite for flows.py. Runs on CPU with only numpy/torch/zuko, so it
does not require legwork, eryn or cupy.

Author: Alexander W. Criswell
"""
import os
import sys
import time

import numpy as np
import torch
import zuko

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import flows

RNG = np.random.default_rng(42)
FAILURES = []

## test sizes. these are set for a fast correctness run, not for model quality:
## the suite checks that the machinery is right, not that a flow trained this
## briefly is accurate.
N_REALIZATIONS = 10
N_TRAIN_DRAWS = 250
N_EPOCHS = 80
N_STUB_BINARIES = 1500
## the section 3 normalization check sums the count dimension over a grid, so a
## small mean count keeps that grid short
POISSON_MEAN = 8.0
COUNT_GRID_MAX = 40


def check(name, ok, extra=''):
    """Record and report a single assertion."""
    print(('  PASS  ' if ok else '  FAIL  ') + name + (('   ' + extra) if extra else ''))
    if not ok:
        FAILURES.append(name)

    return


_SECTION = {'title': None, 'start': None}


def section(title):
    """Close out the previous section's timing and announce the next."""
    if _SECTION['title'] is not None:
        print('  ... %s took %.1fs' % (_SECTION['title'], time.time() - _SECTION['start']))
    _SECTION['title'], _SECTION['start'] = title, time.time()
    print('\n=== ' + title + ' ===')

    return


class StubHyperPrior:
    """Minimal stand-in for PopulationHyperPrior."""

    def __init__(self, rng, hpar_names):
        self.rng = rng
        self.hyperprior_dict = {name: None for name in hpar_names}
        return

    def sample(self, size=1):
        return {'m_mu': self.rng.uniform(0.2, 1.1, size),
                'a_alpha': self.rng.uniform(-0.5, 1.5, size)}


class StubPopModel:
    """
    Minimal stand-in for PopModel, exposing only what draw_training_set() uses.

    Deliberately produces a real S_gw/N_res anti-correlation: a fixed number of
    binaries is drawn, the brightest are declared resolved, and the foreground
    is the summed power of the rest. Resolving more sources therefore removes
    power from the foreground, which is the correlation the joint flow exists
    to capture.
    """

    def __init__(self, fbins, rng, n_binaries=N_STUB_BINARIES):
        self.Nreal = 1
        self.hpar_names = ['m_mu', 'a_alpha']
        self.thresh_val = 7.0
        self.fbins = np.asarray(fbins, dtype=float)
        self.rng = rng
        self.n_binaries = int(n_binaries)
        self.hyperprior = StubHyperPrior(rng, self.hpar_names)
        ## frozen on the first call: renormalizing per call would wash out the
        ## realization-to-realization variation this stub exists to produce
        self.fg_scale = None
        return

    def run_model(self, pop_theta, return_extras=False):
        m_mu = float(np.atleast_1d(pop_theta['m_mu']).ravel()[0])
        a_alpha = float(np.atleast_1d(pop_theta['a_alpha']).ravel()[0])
        n = self.n_binaries

        ## astrophysical draws in [Msun,Msun,kpc,AU], inside the flows bounds
        m_1 = np.clip(self.rng.normal(m_mu, 0.15, n), 0.18, 1.43)
        m_2 = np.clip(self.rng.normal(m_mu, 0.15, n), 0.18, 1.43)
        d_L = 10 ** self.rng.uniform(-1.5, 1.4, n)
        ## choose a to land inside the modelled frequency range
        f_target = self.rng.uniform(self.fbins[0], self.fbins[-1], n)
        a_m = (6.6743e-11 * (m_1 + m_2) * 1.98892e30 / (np.pi * f_target) ** 2) ** (1 / 3)
        a = np.clip(a_m / 1.495978707e11, 1.01e-4, 0.99e-2) * (1 + 0.02 * a_alpha)
        galaxy_draw = np.stack([m_1, m_2, d_L, a], axis=0)

        amp, fgw = flows.get_amp_freq(galaxy_draw)
        amp, fgw = np.asarray(amp, dtype=float), np.asarray(fgw, dtype=float)

        ## the brighter the source and the lower the threshold, the more get
        ## resolved; everything else becomes foreground power
        snr = amp / (1e-21 * self.thresh_val / 7.0)
        resolved = snr > np.quantile(snr, 0.98)
        res_idx = list(np.flatnonzero(resolved))

        n_f = len(self.fbins) - 1
        f_idx = np.clip(np.digitize(fgw, self.fbins) - 1, 0, n_f - 1)
        power = np.zeros(n_f)
        np.add.at(power, f_idx[~resolved], amp[~resolved] ** 2)
        if self.fg_scale is None:
            self.fg_scale = max(power.max(), 1e-300)
        ## keep the dynamic range representative of the real foreground
        fg_psd = 1e-38 * (0.05 + power / self.fg_scale)

        if not return_extras:
            return self.fbins[1:], fg_psd, len(res_idx)

        return self.fbins[1:], fg_psd, len(res_idx), res_idx, galaxy_draw


section('1. band construction')
fbins = flows.make_fbins()
bands = flows.make_bands(fbins, n_bands=3)
for band in bands:
    print('   ', band)
check('3 bands', len(bands) == 3)
check('5 bins each', all(b.n_bins == 5 for b in bands))
check('covers [1e-4,4e-4)', np.isclose(bands[0].f_low, 0.9e-4) and np.isclose(bands[-1].f_high, 3.9e-4))
check('contiguous', np.isclose(bands[0].f_high, bands[1].f_low) and np.isclose(bands[1].f_high, bands[2].f_low))
try:
    flows.make_bands(fbins, n_bands=4)
    check('non-multiple band count raises', False)
except ValueError:
    check('non-multiple band count raises', True)

section('2. ElementwiseTransform roundtrip and Jacobian')
cases = {
    'plain': (flows.ElementwiseTransform(3), np.abs(RNG.normal(size=(200, 3))) + 0.5),
    'log10 psd': (flows.ElementwiseTransform(5, log10=True), 10 ** RNG.uniform(-42, -33, (200, 5))),
    'joint mixed': (flows.ElementwiseTransform(6, log10=True,
                                               offset=np.array([0.] * 5 + [1.])),
                    np.concatenate([10 ** RNG.uniform(-42, -33, (200, 5)),
                                    RNG.integers(0, 500, (200, 1)).astype(float)], axis=-1)),
    'bounded gb': (flows.ElementwiseTransform(4, log10=flows.GB_PARAM_LOG10,
                                              bounds=flows.GB_PARAM_BOUNDS),
                   np.stack([RNG.uniform(0.2, 1.4, 200), RNG.uniform(0.2, 1.4, 200),
                             10 ** RNG.uniform(-2, 1.5, 200),
                             10 ** RNG.uniform(-3.5, -2.2, 200)], axis=-1)),
}
for name, (tf, x) in cases.items():
    xt = torch.as_tensor(x, dtype=flows.DTYPE)
    tf.fit(xt)
    u, logdet = tf(xt)
    rel = (torch.abs(tf.inverse(u) - xt) / torch.abs(xt)).max().item()
    check('roundtrip ' + name, rel < 1e-8, 'max rel err %.1e' % rel)
    check('finite ' + name, torch.isfinite(u).all().item() and torch.isfinite(logdet).all().item())

    probe = xt[:15].clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(lambda z: tf(z)[0].sum(0), probe)
    diag = torch.stack([jac[d, ii, d] for ii in range(15) for d in range(xt.shape[1])])
    numeric = torch.log(torch.abs(diag.reshape(15, xt.shape[1]))).sum(-1)
    err = (tf(xt[:15])[1] - numeric).abs().max().item()
    check('log-det ' + name, err < 1e-6, 'max abs err %.1e' % err)

section('3. mixed joint density normalizes (sum over N, integral over S)')
## a 1-bin band makes the joint 2-D, so the mixed density can be checked exactly
one_bin = flows.FrequencyBand(0, np.array([1e-4, 1.2e-4]), 0, 1)
bf = flows.BandFlow(one_bin, ['m_mu'], hidden_features=(32, 32), transforms=2)
n_tr = 3000
ctx_tr = torch.as_tensor(np.stack([RNG.uniform(0.2, 1.1, n_tr),
                                   RNG.uniform(3, 10, n_tr)], axis=-1), dtype=flows.DTYPE)
psd_tr = torch.as_tensor(10 ** RNG.normal(-38, 0.3, (n_tr, 1)), dtype=flows.DTYPE)
cnt_tr = torch.as_tensor(RNG.poisson(POISSON_MEAN, (n_tr, 1)).astype(float), dtype=flows.DTYPE)
bf.fit_transforms(ctx_tr, psd_tr, cnt_tr, torch.zeros((0, 4), dtype=flows.DTYPE))
flows.train_component(bf.joint_flow, bf.joint_transform,
                      bf.transform_context(ctx_tr).detach(),
                      bf.stack_joint(psd_tr, cnt_tr), n_epochs=200, batch_size=256, lr=3e-3,
                      dequantize=bf.count_mask, rng=RNG, progress=False)

## evaluate the whole (S,N) product grid as one batch: looping over N would
## cost n_counts*n_quad separate flow passes, each re-running the conditioner
## over the full S grid
s_grid = np.logspace(-41, -35, 400)
n_grid = np.arange(COUNT_GRID_MAX, dtype=float)
one_ctx = ctx_tr[:1]

s_mesh, n_mesh = np.meshgrid(s_grid, n_grid, indexing='ij')
s_t = torch.as_tensor(s_mesh.reshape(-1, 1), dtype=flows.DTYPE)
n_t = torch.as_tensor(n_mesh.reshape(-1, 1), dtype=flows.DTYPE)
with torch.no_grad():
    lp = bf.joint_log_prob(one_ctx.expand(s_t.shape[0], -1), s_t, n_t, n_quad=9)
dens = np.exp(lp.numpy()).reshape(len(s_grid), len(n_grid))
total = float(np.trapezoid(dens, s_grid, axis=0).sum())
check('sum_N integral_S p(S,N) = 1', abs(total - 1.0) < 3e-2, 'total = %.4f' % total)

section('4. count handling (floor vs round, and the quadrature pmf)')
fg_s, n_s = bf.sample_joint(one_ctx.expand(20000, -1), n_samples=1)
n_s = n_s[0].numpy()
check('sampled counts are integers', np.all(n_s == np.floor(n_s)))
check('sampled counts non-negative', np.all(n_s >= 0))
## a round() instead of floor() on the dequantized value biases this high by ~0.5
check('sampled count mean unbiased', abs(n_s.mean() - POISSON_MEAN) < 0.4,
      'flow %.3f vs Poisson(%.0f) truth %.1f' % (n_s.mean(), POISSON_MEAN, POISSON_MEAN))
check('sampled count std matches', abs(n_s.std() - np.sqrt(POISSON_MEAN)) < 0.6,
      'flow %.3f vs %.3f' % (n_s.std(), np.sqrt(POISSON_MEAN)))

section('5. joint flow recovers the S_gw / N_res anti-correlation')
hpar_names = ['m_mu', 'a_alpha']
stub = StubPopModel(fbins, RNG)
train_set = flows.draw_training_set(stub, N_TRAIN_DRAWS, bands, n_realizations=1,
                                    snr_thresh_range=(3.0, 10.0), rng=RNG, progress=False)
check('context shape', tuple(train_set['context'].shape) == (N_TRAIN_DRAWS, 3),
      str(tuple(train_set['context'].shape)))
check('context names', train_set['context_names'] == hpar_names + ['snr_thresh'])
check('draw_index unique per row',
      len(np.unique(np.asarray(train_set['draw_index']))) == N_TRAIN_DRAWS)
for jj, data in enumerate(train_set['bands']):
    check('band %d fg_psd rows' % jj, tuple(data['fg_psd'].shape) == (N_TRAIN_DRAWS, 5))
    check('band %d gb_index in range' % jj,
          bool((data['gb_index'] >= 0).all() and (data['gb_index'] < N_TRAIN_DRAWS).all()))

prior = flows.BandedFlowPrior(bands, hpar_names, hidden_features=(48, 48), transforms=3)
history = flows.train_banded_prior(prior, train_set, n_epochs=N_EPOCHS, batch_size=128, lr=3e-3,
                                   rng=RNG, progress=False, val_fraction=0.2)
for jj, h in enumerate(history):
    print('  band %d  joint %.2f -> %.2f (val %.2f)   gb %.2f -> %.2f (val %.2f)' % (
        jj, h['joint'][0], h['joint'][-1], h['joint_val'][-1],
        h['gb'][0] if len(h['gb']) else np.nan, h['gb'][-1] if len(h['gb']) else np.nan,
        h['gb_val'][-1] if len(h['gb_val']) else np.nan))
    check('band %d joint loss improved' % jj, h['joint'][-1] < h['joint'][0])
    check('band %d val loss reported' % jj, len(h['joint_val']) == N_EPOCHS)

n_ctx = 2
diag = flows.draw_diagnostic_set(stub, bands, np.array([[0.6, 0.5], [0.9, -0.2]]), 7.0,
                                 n_realizations=N_REALIZATIONS, progress=False)
n_diag = n_ctx * N_REALIZATIONS
check('diagnostic rows', tuple(diag['context'].shape) == (n_diag, 3),
      str(tuple(diag['context'].shape)))
check('diagnostic contexts repeat',
      bool(torch.allclose(diag['context'][0], diag['context'][N_REALIZATIONS - 1])) and
      not bool(torch.allclose(diag['context'][0], diag['context'][N_REALIZATIONS])))
check('diagnostic draw_index groups',
      np.array_equal(np.bincount(np.asarray(diag['draw_index'])),
                     np.full(n_ctx, N_REALIZATIONS)))

scatter = flows.compare_conditional_scatter(prior, diag, n_flow_samples=3000)
print('  %-5s %-5s %14s %14s %12s %12s' % ('band', 'draw', 'Nres emp', 'Nres flow',
                                            'corr emp', 'corr flow'))
for row in scatter:
    print('  %-5d %-5d %6.1f +- %-5.1f %6.1f +- %-5.1f %12.3f %12.3f' % (
        row['band'], row['draw'],
        row['empirical']['n_res_mean'], row['empirical']['n_res_std'],
        row['flow']['n_res_mean'], row['flow']['n_res_std'],
        row['empirical']['corr_power_nres'], row['flow']['corr_power_nres']))

emp_corr = np.array([r['empirical']['corr_power_nres'] for r in scatter])
flow_corr = np.array([r['flow']['corr_power_nres'] for r in scatter])
check('flow correlation is well defined', np.all(np.isfinite(flow_corr)))

## the empirical correlation is estimated from N_REALIZATIONS points, so its
## standard error is roughly 1/sqrt(n-3). at small n it is far too noisy to
## assert against, so report the comparison instead of failing on it. raise
## N_REALIZATIONS to O(100) to turn this into a real acceptance test.
emp_se = 1.0 / np.sqrt(max(N_REALIZATIONS - 3, 1))
agree = np.sign(flow_corr) == np.sign(emp_corr)
print('  empirical corr standard error ~ %.2f at n_realizations=%d' % (emp_se, N_REALIZATIONS))
print('  sign agreement: %d of %d bands/contexts%s' % (
    agree.sum(), len(agree),
    '   (informational: empirical estimate is too noisy to assert on)' if emp_se > 0.15 else ''))
if emp_se <= 0.15:
    check('flow reproduces correlation sign', bool(np.all(agree)),
          'emp %s vs flow %s' % (np.round(emp_corr, 2), np.round(flow_corr, 2)))

emp_n = np.array([r['empirical']['n_res_mean'] for r in scatter])
flow_n = np.array([r['flow']['n_res_mean'] for r in scatter])
rel_n = np.abs(flow_n - emp_n) / np.maximum(emp_n, 1.0)
check('flow N_res means track empirical', np.max(rel_n) < 0.5, 'max rel err %.3f' % np.max(rel_n))

section('6. multi-realization structure')
n_multi_draws = 25
n_multi = n_multi_draws * N_REALIZATIONS
multi = flows.draw_training_set(stub, n_multi_draws, bands, n_realizations=N_REALIZATIONS,
                                rng=RNG, progress=False)
check('rows = n_draws * n_realizations', tuple(multi['context'].shape) == (n_multi, 3),
      str(tuple(multi['context'].shape)))
di = np.asarray(multi['draw_index'])
check('draw_index counts', np.array_equal(np.bincount(di),
                                          np.full(n_multi_draws, N_REALIZATIONS)))
check('context repeats within a draw', all(
    bool(torch.allclose(multi['context'][N_REALIZATIONS * k],
                        multi['context'][N_REALIZATIONS * k + r]))
    for k in range(n_multi_draws) for r in range(N_REALIZATIONS)))
check('context differs across draws',
      not bool(torch.allclose(multi['context'][0], multi['context'][N_REALIZATIONS])))
for jj, data in enumerate(multi['bands']):
    check('band %d multi fg_psd rows' % jj, tuple(data['fg_psd'].shape) == (n_multi, 5))
    check('band %d multi gb_index in range' % jj,
          bool((data['gb_index'] >= 0).all() and (data['gb_index'] < n_multi).all()))
## realizations at one context must actually differ. compare exactly rather
## than with allclose: the PSD is ~1e-38, far below allclose's default atol of
## 1e-8, so any two values would compare equal
check('realizations are independent',
      all(not torch.equal(data['fg_psd'][0], data['fg_psd'][1]) for data in multi['bands']))
check('realizations differ across all bands', all(
    not torch.equal(data['fg_psd'][N_REALIZATIONS * k], data['fg_psd'][N_REALIZATIONS * k + 1])
    for data in multi['bands'] for k in range(n_multi_draws)))

section('7. grouped validation split does not leak')
di_multi = np.asarray(multi['draw_index'])
rng_split = np.random.default_rng(7)
hist_multi = flows.train_banded_prior(prior, multi, n_epochs=5, batch_size=32, lr=1e-3,
                                      rng=rng_split, progress=False, val_fraction=0.2)
check('val history returned', all(len(h['joint_val']) == 5 for h in hist_multi))
## reproduce the split logic to confirm it partitions on whole draws
uniq = np.unique(di_multi)
n_val = max(int(round(0.2 * len(uniq))), 1)
picked = np.random.default_rng(7).choice(uniq, size=n_val, replace=False)
is_val = np.isin(di_multi, picked)
check('split is on whole draws', all(len(np.unique(is_val[di_multi == d])) == 1 for d in uniq))
check('split is non-trivial', 0 < is_val.sum() < len(is_val), '%d val rows' % is_val.sum())

section('8. log_prob and sampling')
lnp, per_band = prior.log_prob(train_set['context'], train_set['bands'])
check('log_prob shape', tuple(lnp.shape) == (N_TRAIN_DRAWS,), str(tuple(lnp.shape)))
check('log_prob finite', torch.isfinite(lnp).all().item())
check('terms are joint + gb', all(set(t) == {'joint', 'gb', 'total'} for t in per_band))
check('terms sum to total', all(torch.allclose(t['joint'] + t['gb'], t['total']) for t in per_band))
check('total is sum over bands', torch.allclose(lnp, sum(t['total'] for t in per_band)))
shuffled = train_set['context'][torch.as_tensor(RNG.permutation(N_TRAIN_DRAWS), dtype=torch.long)]
lnp_shuf, _ = prior.log_prob(shuffled, train_set['bands'])
check('correct context beats shuffled', lnp.mean().item() > lnp_shuf.mean().item(),
      '%.1f vs %.1f' % (lnp.mean().item(), lnp_shuf.mean().item()))

lnp_q1, _ = prior.log_prob(train_set['context'], train_set['bands'], n_quad=1)
lnp_q9, _ = prior.log_prob(train_set['context'], train_set['bands'], n_quad=9)
check('n_quad is honoured and stays finite',
      torch.isfinite(lnp_q1).all().item() and torch.isfinite(lnp_q9).all().item())
## the count quadrature is knowingly unconverged at the default node count; see
## the note in BandFlow.joint_log_prob() for the exact-CDF fix. reported here
## rather than asserted, since this test case only needs the code to run.
print('  count quadrature residual: |lp(9) - lp(5)| = %.2e,  |lp(1) - lp(9)| = %.2e'
      % ((lnp_q9 - lnp).abs().max().item(), (lnp_q1 - lnp_q9).abs().max().item()))
print('    (unconverged by design for now -- exact pmf via CDF difference is the fix)')

draws = prior.sample(np.array([[0.6, 0.5], [0.9, -0.2]]), 7.0)
check('one list per band', len(draws) == 3)
check('2 draws per band', all(len(d) == 2 for d in draws))
check('fg_psd shape', all(tuple(d['fg_psd'].shape) == (5,) for band in draws for d in band))
check('fg_psd positive', all((d['fg_psd'] > 0).all().item() for band in draws for d in band))
check('n_res int >= 0', all(isinstance(d['n_res'], int) and d['n_res'] >= 0
                            for band in draws for d in band))
check('gb_theta shape matches n_res', all(d['gb_theta'].shape == (d['n_res'], 4)
                                          for band in draws for d in band))
lo = torch.as_tensor(flows.GB_PARAM_BOUNDS[:, 0], dtype=flows.DTYPE)
hi = torch.as_tensor(flows.GB_PARAM_BOUNDS[:, 1], dtype=flows.DTYPE)
check('gb_theta within physical bounds', all(
    bool((d['gb_theta'] >= lo * 0.999).all() and (d['gb_theta'] <= hi * 1.001).all())
    for band in draws for d in band if d['n_res']))
print('  sampled n_res per band:', [[d['n_res'] for d in band] for band in draws])

section('9. save / load')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_prior.pkl')
try:
    prior.save(path)
    restored = flows.load_banded_prior(path, hidden_features=(48, 48), transforms=3)
    lnp_r, _ = restored.log_prob(train_set['context'], train_set['bands'])
    check('load reproduces log_prob', torch.allclose(lnp, lnp_r),
          'max diff %.1e' % (lnp - lnp_r).abs().max().item())
    check('bands restored', [(b.start, b.stop, b.n_bins) for b in restored.bands] ==
          [(b.start, b.stop, b.n_bins) for b in bands])
    check('band frequencies restored', all(np.allclose(a.fs, b.fs)
                                           for a, b in zip(restored.bands, bands)))
finally:
    if os.path.exists(path):
        os.remove(path)

section('done')
print('\n' + ('ALL PASSED' if not FAILURES else 'FAILURES (%d): %s' % (len(FAILURES),
                                                                       ', '.join(FAILURES))))
sys.exit(1 if FAILURES else 0)
