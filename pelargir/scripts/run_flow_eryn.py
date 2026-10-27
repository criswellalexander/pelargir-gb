"""
Parallel-tempered MCMC (Eryn) on the flow toy-model posterior: the eight context parameters ('pop')
and log10 S_gw in every modelled bin ('spectrum'), given the dataset's fixed resolved-GB state
(flow_eryn.run_eryn). Joint Eryn moves by default; with --gibbs, HMC-within-Gibbs: numpyro NUTS/HMC on 'spectrum' at fixed 'pop' is one more weighted move.

Writes to outdir:
- samples.npz (flow_eryn.ErynResult)
- summary.json (per-parameter mean, std, 90% interval and whether it holds the truth, autocorrelation times and ESS, move and swap acceptance, the final ladder, timings, and (with --gibbs) the HMC block's per-temperature acceptance, step sizes, steps per transition and divergences)
- corner.png, trace.png, spectrum.png
- with --compare: another run's samples.npz (NUTS or Eryn), compare.png, and each parameter's mean difference in that run's posterior standard deviations.

--time_only K instead times K steps (after an untimed warm-up run; with --gibbs after K adapting burn-in steps) and projects the run.



Usage
-----
    python -m pelargir.scripts.run_flow_eryn emulator_dir data.npz outdir [--tempering flow_prior|all]
           [--ntemps 8] [--nwalkers 16] [--nsteps 2000] [--burn B] [--init map|prior|truth] [--n_quad 8]
           [--Tmax inf] [--no_adapt] [--weights joint,pop,spectrum,prior[,hmc]] [--seed 0]
           [--gibbs] [--hmc_kernel nuts|hmc] [--hmc_adapt laplace|numpyro] [--max_tree_depth 10]
           [--trajectory_length 1.5708] [--target_accept 0.8] [--n_transitions 1]
           [--time_only K] [--compare other/samples.npz]
"""
import os
import json
import math
import time
import argparse


def main(args):
    '''

    Run Eryn (or time it) and write the results.

    Arguments
    ------------------------------
    args : argparse.Namespace
    	Parsed arguments (see cli).

    Returns
    ------------------------------
    None

    '''
    from pelargir import backend
    backend.set_backend('jax')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from pelargir import flow_data as fd
    from pelargir import flow_inference as fi
    from pelargir import flow_eryn as fe
    from pelargir import flow_plots as fp
    from pelargir.utils import set_style
    set_style()
    os.makedirs(args.outdir, exist_ok=True)

    em = fd.load_emulator(args.emulator)
    data = fi.ToyData.load(args.data)
    post = fi.FlowPosterior(em, data, width=args.width, n_quad=args.n_quad)
    weights = None
    if args.weights is not None:
        weights = dict(zip(('joint', 'pop', 'spectrum', 'prior', 'hmc'), (float(w) for w in args.weights.split(','))))
    hmc = None
    if args.gibbs:
        hmc = dict(kernel=args.hmc_kernel, adapt=args.hmc_adapt, max_tree_depth=args.max_tree_depth,
                   trajectory_length=args.trajectory_length, target_accept=args.target_accept,
                   n_transitions=args.n_transitions)
    ## the HMC block's step size adapts during burn-in
    burn = args.burn if args.burn is not None else (500 if args.gibbs else 0)
    kw = dict(ntemps=args.ntemps, nwalkers=args.nwalkers, tempering=args.tempering, init=args.init, Tmax=args.Tmax,
              adaptive=not args.no_adapt, move_weights=weights, seed=args.seed, progress=not args.time_only, hmc=hmc)

    if args.time_only:
        ## an untimed warm-up compiles everything; then time a run's sampling phase alone (excludes the start point)
        t0 = time.time()
        fe.run_eryn(post, 2, **kw)
        t_first = time.time() - t0
        ## with the HMC block, K adapting burn-in steps first so the timed steps use a tuned step size
        res = fe.run_eryn(post, args.time_only, burn=args.time_only if args.gibbs else 0, **kw)
        per_step = res.meta['seconds']['run']/args.time_only
        print("{} temps x {} walkers ({}{}): {:.2f} s per step; start point (MAP) {:.0f} s; first run incl. compiles "
              "{:.0f} s. Projected {} steps: ~{:.0f} min".format(args.ntemps, args.nwalkers, args.tempering,
                                                                 ', HMC-within-Gibbs' if args.gibbs else '', per_step,
                                                                 res.meta['seconds']['init'], t_first, burn + args.nsteps,
                                                                 (burn + args.nsteps)*per_step/60), flush=True)
        if args.gibbs:
            h = res.meta['hmc']['sampling']
            print("HMC block: {} calls; per temperature accept {}, leapfrog steps per transition {}, median step size "
                  "{}".format(h.get('calls', 0), np.round(h.get('accept_prob', []), 2).tolist(),
                              np.round(h.get('num_steps', []), 1).tolist(),
                              np.round(h.get('step_size_median', []), 3).tolist()), flush=True)
        return

    res = fe.run_eryn(post, args.nsteps, burn=burn, **kw)
    res.save(os.path.join(args.outdir, 'samples.npz'))
    S, W, _ = res.theta.shape
    flat_th, flat_y = res.theta.reshape(S*W, 8), res.y.reshape(S*W, -1)
    names, truth = fd.CONTEXT_NAMES, data.truth_theta
    tau = np.asarray(res.meta['autocorr_time']['pop'])
    params = {}
    for j, n in enumerate(names):
        lo, hi = np.percentile(flat_th[:, j], [5, 95])
        params[n] = dict(truth=float(truth[j]), mean=float(flat_th[:, j].mean()), std=float(flat_th[:, j].std()),
                         q05=float(lo), q95=float(hi), truth_in_90=bool(lo <= truth[j] <= hi),
                         autocorr_time=float(tau[j]), ess=float(S*W/tau[j]))
    y_true = np.log10(data.truth_S)
    lo, hi = np.percentile(flat_y, [5, 95], axis=0)
    summary = dict(parameters=params, spectrum=dict(truth_in_90_fraction=float(np.mean((lo <= y_true) & (y_true <= hi))),
                                                     max_autocorr_time=float(np.max(res.meta['autocorr_time']['spectrum']))),
                   move_acceptance_T1=res.meta['move_acceptance_T1'], swap_acceptance=res.meta['swap_acceptance'],
                   final_betas=res.betas[-1].tolist(), seconds=res.meta['seconds'], settings=res.meta['settings'])
    if args.gibbs:
        summary['hmc'] = res.meta['hmc']

    fp.plot_corner(os.path.join(args.outdir, 'corner.png'), flat_th, truth, names)
    ## T = 1 ln_post per walker, as for HMC
    ln_post = (res.log_like[:, 0] + res.log_prior[:, 0]).T
    fp.plot_trace(os.path.join(args.outdir, 'trace.png'), np.swapaxes(res.theta, 0, 1), ln_post, truth, names)
    fp.plot_spectrum(os.path.join(args.outdir, 'spectrum.png'), data.fs, 10**flat_y, data.truth_S, data.data_psd,
                     data.noise_psd)
    label = 'Eryn + HMC' if args.gibbs else 'Eryn'
    if args.compare:
        other = np.load(args.compare)
        h = other['theta'].reshape(-1, 8)
        ## the other run's sampler, from its meta (Eryn results carry move acceptance, Gibbs ones the HMC block)
        meta = json.loads(str(other['meta']))
        other_label = ('NUTS' if 'move_acceptance_T1' not in meta else 'Eryn + HMC' if meta.get('hmc') else 'Eryn')
        fp.plot_corner_compare(os.path.join(args.outdir, 'compare.png'), flat_th, h, (label, other_label), truth, names)
        summary['vs_compare'] = dict(path=args.compare, sampler=other_label, parameters={
            n: dict(mean_diff_in_std=float((flat_th[:, j].mean() - h[:, j].mean())/h[:, j].std()),
                    std_ratio=float(flat_th[:, j].std()/h[:, j].std())) for j, n in enumerate(names)})
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as fh_:
        json.dump(summary, fh_, indent=1)
    for n, p in params.items():
        extra = summary.get('vs_compare', {}).get('parameters', {}).get(n)
        print("{:>18}: truth {:.4g}, mean {:.4g} +- {:.3g}, 90% [{:.4g}, {:.4g}] {}, tau {:.0f}, ESS {:.0f}{}".format(
            n, p['truth'], p['mean'], p['std'], p['q05'], p['q95'], 'ok' if p['truth_in_90'] else 'MISS',
            p['autocorr_time'], p['ess'], '' if extra is None else ", vs {} {:+.2f} sd, std ratio {:.2f}".format(
                summary['vs_compare']['sampler'], extra['mean_diff_in_std'], extra['std_ratio'])))
    print("T=1 move acceptance:", {k: round(v, 3) for k, v in summary['move_acceptance_T1'].items()},
          "; swap acceptance:", np.round(summary['swap_acceptance'], 3).tolist() if summary['swap_acceptance'] else None)
    if args.gibbs:
        for phase in ('burn', 'sampling'):
            h = res.meta['hmc'][phase] or {}
            print("HMC block ({}): {} calls; per temperature accept {}, leapfrog steps per transition {}, divergences "
                  "{}, median step size {}".format(phase, h.get('calls', 0), np.round(h.get('accept_prob', []), 2).tolist(),
                                                   np.round(h.get('num_steps', []), 1).tolist(), h.get('divergences'),
                                                   np.round(h.get('step_size_median', []), 3).tolist()))
    print("burn {:.0f} s, run {:.0f} s; wrote {}".format(res.meta['seconds']['burn'], res.meta['seconds']['run'],
                                                        args.outdir), flush=True)


def cli(argv=None):
    '''

    Parse the command line and run main.

    Arguments
    ------------------------------
    argv : list of str or None
    	Arguments; None reads sys.argv. (Default: None)

    Returns
    ------------------------------
    None

    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('emulator', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--tempering', choices=['flow_prior', 'all'], default='flow_prior')
    parser.add_argument('--ntemps', type=int, default=8)
    parser.add_argument('--nwalkers', type=int, default=16)
    parser.add_argument('--nsteps', type=int, default=2000)
    parser.add_argument('--burn', type=int, default=None, help="Unstored burn-in steps (default 0; 500 with --gibbs).")
    parser.add_argument('--init', choices=['map', 'prior', 'truth'], default='map')
    parser.add_argument('--n_quad', type=int, default=8)
    parser.add_argument('--width', type=float, default=0.1, help="Logistic width (SNR) of the resolved-GB term.")
    parser.add_argument('--Tmax', type=float, default=float('inf'))
    parser.add_argument('--no_adapt', action='store_true', help="Fixed temperature ladder.")
    parser.add_argument('--weights', type=str, default=None,
                        help="Move weights: joint, pop, spectrum (the Poisson-move analogue), prior draws[, hmc]. "
                             "Default 0.3,0.25,0.3,0.15 (flow_eryn.DEFAULT_WEIGHTS), or with --gibbs "
                             "0.1,0.4,0,0.1,0.4 (flow_gibbs.GIBBS_WEIGHTS).")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_only', type=int, default=0, help="Time this many steps and project the run, then stop.")
    parser.add_argument('--gibbs', action='store_true',
                        help="HMC-within-Gibbs: add numpyro NUTS/HMC on the spectrum at fixed population parameters.")
    parser.add_argument('--hmc_kernel', choices=['nuts', 'hmc'], default='nuts')
    parser.add_argument('--hmc_adapt', choices=['laplace', 'numpyro'], default='laplace',
                        help="laplace: per-temperature Laplace mass, step size adapted; numpyro: step size and dense "
                             "mass adapted. Both only during burn-in.")
    parser.add_argument('--max_tree_depth', type=int, default=10)
    parser.add_argument('--trajectory_length', type=float, default=math.pi/2,
                        help="Fixed-length HMC trajectory (mass-whitened units). Default pi/2, a quarter orbit.")
    parser.add_argument('--target_accept', type=float, default=0.8)
    parser.add_argument('--n_transitions', type=int, default=1, help="numpyro transitions per HMC move.")
    parser.add_argument('--compare', type=str, default=None,
                        help="Another run's samples.npz (NUTS or Eryn) to compare against.")
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
