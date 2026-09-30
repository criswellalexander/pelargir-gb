"""
Worker for tests/test_backend_routes.py; each call runs in its own process, since the backend
binds once per process.

    python backend_route_worker.py draw  out.npz          (PELARGIR_BACKEND=numpy: the shared inputs)
    python backend_route_worker.py route in.npz out.npz   (PELARGIR_BACKEND=cupy or jax)

"route" repeats run_pelargir's simulated-data step and PopModel's likelihood evaluation on the
shared numpy inputs, each the way the active backend does it.
"""
import os
import sys

import numpy as np

FIDUCIAL = np.array([0.6, 0.15, 3.31, 0.75, 0.33, 0.5])  ## Table 1, arXiv:2604.03390
NAMES = ['m_mu', 'm_sigma', 'rh_disk', 'r_bulge', 'q_bd', 'a_alpha']
NSIM, NREAL, NPAR = 200000, 3, 2
FMIN, FMAX, FBIN = 1e-4, 1e-3, 2e-5


def draw(out):
    '''One numpy galaxy, its thresholded data (serial_array_sort), and fixed likelihood inputs.'''
    from pelargir.inference import GalacticBinaryPrior
    from pelargir.models import PopModel
    from pelargir.utils import get_amp_freq

    rng = np.random.default_rng(150914)
    fbins = np.arange(FMIN - FBIN/2, FMAX + FBIN/2, FBIN)
    prior = GalacticBinaryPrior(rng)
    prior.condition({k: np.array([v]) for k, v in zip(NAMES, FIDUCIAL)})
    gbs = prior.sample_conditional(NSIM).squeeze()                  ## (4, NSIM)
    A, f = get_amp_freq(gbs)
    pm = PopModel(NSIM, rng, fbins=fbins, Nreal=1)
    Nres, fg, res_idx = pm.thresher.serial_array_sort(np.array([f, A]), fbins, snr_thresh=pm.thresh_val,
                                                      get_indices=True)
    data_fg = pm.reweight_foreground(fg)[1:]
    ## model realizations around the data, and hyperparameters near the fiducial point
    psd = data_fg[:, None, None]*10**(0.1*rng.standard_normal((data_fg.size, NREAL, NPAR)))
    nres = rng.poisson(float(Nres), (NREAL, NPAR))
    thetas = FIDUCIAL*(1 + 0.03*rng.standard_normal((NPAR, 6)))
    np.savez(out, fbins=fbins, gbs=gbs, data_fg=data_fg, data_nres=int(Nres), gb_thetas=gbs[:, res_idx].T,
             psd=psd, nres=nres, thetas=thetas)


def route(inp, out):
    from pelargir import backend
    xp = backend.xp
    from pelargir.models import PopModel
    from pelargir.utils import get_amp_freq, to_numpy
    from pelargir.jax_population import _host

    d = np.load(inp)
    fbins = xp.asarray(d['fbins'])
    rng = xp.random.default_rng(1)

    ## run_pelargir's simulated-data step
    gbs = xp.asarray(d['gbs'])
    A, f = get_amp_freq(gbs)
    sim = PopModel(NSIM, rng, fbins=fbins, Nreal=1)
    sort = sim.thresher.jax_array_sort if backend.BACKEND == 'jax' else sim.thresher.serial_array_sort
    Nres, fg, res_idx = sort(xp.array([f, A]), sim.fbins, snr_thresh=sim.thresh_val, get_indices=True)
    data_fg = sim.reweight_foreground(fg)[1:]

    ## the likelihood terms of PopModel.ln_prob on fixed run_model outputs
    pm = PopModel(NSIM, rng, fbins=fbins, Nreal=NREAL, res_scatter=False, res_dynamic_scatter=False)
    pm.construct_likelihood({'fg': xp.asarray(d['data_fg']), 'fg_sigma': xp.asarray(0.1), 'Nres': int(d['data_nres']),
                             'gb_thetas': xp.asarray(d['gb_thetas'])}, hp_beta=0.05, hp_alpha=5)
    psd, nres, thetas = xp.asarray(d['psd']), xp.asarray(d['nres']), d['thetas']
    if backend.BACKEND == 'jax':
        pm.last_thetas = thetas
        pm._jax_ln_like(psd, nres)
        terms = [_host(t) for t in pm.last_ln_terms]
    else:
        pm.gbprior.condition({k: xp.asarray(thetas[:, i]) for i, k in enumerate(NAMES)})
        terms = [to_numpy(pm.fg_ln_prob(psd)), to_numpy(pm.N_res_ln_prob(nres)),
                 to_numpy(pm.res_astro_ln_prob(pm.gbprior, psd, pm.approx_lisa_psd, pm.thresh_val))]
    np.savez(out, nres=int(_host(Nres)), fg=to_numpy(data_fg), res_idx=np.array([int(i) for i in res_idx], dtype=np.int64),
             ln_fg=terms[0], ln_nres=terms[1], ln_res=terms[2], xp=xp.__name__)


if __name__ == '__main__':
    if sys.argv[1] == 'draw':
        draw(sys.argv[2])
    else:
        route(sys.argv[2], sys.argv[3])
