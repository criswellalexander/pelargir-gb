"""
Joint parallel-tempered MCMC (Eryn) on the flow toy-model posterior: the eight context parameters
('pop') and log10 S_gw in every modelled bin ('spectrum'), given the dataset's fixed resolved-GB state
(flow_eryn.run_eryn).

Writes to outdir: samples.npz (flow_eryn.ErynResult); summary.json (per-parameter mean, std, 90%
interval and whether it holds the truth, autocorrelation times and ESS, move and swap acceptance, the
final ladder, timings); corner.png, trace.png, spectrum.png, and with --compare an HMC run's
samples.npz, compare_hmc.png and each parameter's mean difference in posterior standard deviations.
--time_only K instead times K steps (after an untimed warm-up run) and projects the run.

Usage
-----
    python -m pelargir.scripts.run_flow_eryn emulator_dir data.npz outdir [--tempering flow_prior|all]
           [--ntemps 8] [--nwalkers 16] [--nsteps 2000] [--burn 0] [--init map|prior|truth] [--n_quad 8]
           [--Tmax inf] [--no_adapt] [--weights 0.3,0.25,0.3,0.15] [--seed 0] [--time_only K]
           [--compare hmc/samples.npz]
"""
import os
import json
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
    weights = dict(zip(('joint', 'pop', 'spectrum', 'prior'), (float(w) for w in args.weights.split(','))))
    kw = dict(ntemps=args.ntemps, nwalkers=args.nwalkers, tempering=args.tempering, init=args.init, Tmax=args.Tmax,
              adaptive=not args.no_adapt, move_weights=weights, seed=args.seed, progress=not args.time_only)

    if args.time_only:
        ## an untimed warm-up compiles everything; then time a run's sampling phase alone (excludes the start point)
        t0 = time.time()
        fe.run_eryn(post, 2, **kw)
        t_first = time.time() - t0
        res = fe.run_eryn(post, args.time_only, **kw)
        per_step = res.meta['seconds']['run']/args.time_only
        print("{} temps x {} walkers ({}): {:.2f} s per step; start point (MAP) {:.0f} s; first run incl. compiles "
              "{:.0f} s. Projected {} steps: ~{:.0f} min".format(args.ntemps, args.nwalkers, args.tempering, per_step,
                                                                 res.meta['seconds']['init'], t_first,
                                                                 args.burn + args.nsteps,
                                                                 (args.burn + args.nsteps)*per_step/60), flush=True)
        return

    res = fe.run_eryn(post, args.nsteps, burn=args.burn, **kw)
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

    fp.plot_corner(os.path.join(args.outdir, 'corner.png'), flat_th, truth, names)
    ## T = 1 ln_post per walker, as for HMC
    ln_post = (res.log_like[:, 0] + res.log_prior[:, 0]).T
    fp.plot_trace(os.path.join(args.outdir, 'trace.png'), np.swapaxes(res.theta, 0, 1), ln_post, truth, names)
    fp.plot_spectrum(os.path.join(args.outdir, 'spectrum.png'), data.fs, 10**flat_y, data.truth_S, data.data_psd,
                     data.noise_psd)
    if args.compare:
        from pelargir import flow_hmc as fh
        hmc = fh.HMCResult.load(args.compare)
        h = hmc.theta.reshape(-1, 8)
        fp.plot_corner_compare(os.path.join(args.outdir, 'compare_hmc.png'), flat_th, h, ('Eryn', 'NUTS'), truth, names)
        summary['vs_hmc'] = {n: dict(mean_diff_in_std=float((flat_th[:, j].mean() - h[:, j].mean())/h[:, j].std()),
                                     std_ratio=float(flat_th[:, j].std()/h[:, j].std())) for j, n in enumerate(names)}
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as fh_:
        json.dump(summary, fh_, indent=1)
    for n, p in params.items():
        extra = summary.get('vs_hmc', {}).get(n)
        print("{:>18}: truth {:.4g}, mean {:.4g} +- {:.3g}, 90% [{:.4g}, {:.4g}] {}, tau {:.0f}, ESS {:.0f}{}".format(
            n, p['truth'], p['mean'], p['std'], p['q05'], p['q95'], 'ok' if p['truth_in_90'] else 'MISS',
            p['autocorr_time'], p['ess'], '' if extra is None else ", vs NUTS {:+.2f} sd, std ratio {:.2f}".format(
                extra['mean_diff_in_std'], extra['std_ratio'])))
    print("T=1 move acceptance:", {k: round(v, 3) for k, v in summary['move_acceptance_T1'].items()},
          "; swap acceptance:", np.round(summary['swap_acceptance'], 3).tolist() if summary['swap_acceptance'] else None)
    print("run {:.0f} s; wrote {}".format(res.meta['seconds']['run'], args.outdir), flush=True)


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
    parser.add_argument('--burn', type=int, default=0)
    parser.add_argument('--init', choices=['map', 'prior', 'truth'], default='map')
    parser.add_argument('--n_quad', type=int, default=8)
    parser.add_argument('--width', type=float, default=0.1, help="Logistic width (SNR) of the resolved-GB term.")
    parser.add_argument('--Tmax', type=float, default=float('inf'))
    parser.add_argument('--no_adapt', action='store_true', help="Fixed temperature ladder.")
    parser.add_argument('--weights', type=str, default='0.3,0.25,0.3,0.15',
                        help="Move weights: joint, pop, spectrum (the Poisson-move analogue), prior draws.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_only', type=int, default=0, help="Time this many steps and project the run, then stop.")
    parser.add_argument('--compare', type=str, default=None, help="An HMC run's samples.npz to compare against.")
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
