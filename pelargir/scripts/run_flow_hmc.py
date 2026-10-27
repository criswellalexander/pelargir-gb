"""
Joint NUTS (numpyro) on the flow toy-model posterior: the eight context parameters and log10 S_gw in
every modelled bin, given the dataset's fixed resolved-GB state (flow_hmc.run_nuts).

Writes to outdir: samples.npz (flow_hmc.HMCResult); summary.json (per-parameter R-hat, ESS, 90%
intervals and whether they contain the truth, divergences, tree depths, step sizes, timings, and the
quadrature error of the chosen n_quad on posterior draws); corner.png, trace.png and spectrum.png.
--time_only K instead times K warmup iterations (against 2, to remove compile time) and projects the
full run's cost.

Usage
-----
    python -m pelargir.scripts.run_flow_hmc emulator_dir data.npz outdir [--n_quad 8] [--width 0.1]
           [--chains 4] [--warmup 500] [--samples 500] [--init map|prior|truth] [--diag]
           [--target_accept 0.8] [--max_tree_depth 10] [--seed 0] [--time_only K]
"""
import os
import json
import argparse


def main(args):
    '''

    Run NUTS (or time its warmup) and write the results.

    Arguments
    ------------------------------
    args : argparse.Namespace
    	Parsed arguments (see cli).

    Returns
    ------------------------------
    None

    '''
    ## JAX's default allocator: unlike the forward-model scripts, NUTS runs one compiled program, and under
    ## the 'platform' allocator numpyro's initialization ran out of GPU memory (8-byte allocations failing
    ## with a 0.7 GiB peak) on an 8 GB RTX 4070
    from pelargir import backend
    backend.set_backend('jax')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from pelargir import flow_data as fd
    from pelargir import flow_inference as fi
    from pelargir import flow_hmc as fh
    from pelargir.flow_plots import plot_corner, plot_trace, plot_spectrum
    from pelargir.utils import set_style
    set_style()
    os.makedirs(args.outdir, exist_ok=True)

    em = fd.load_emulator(args.emulator)
    data = fi.ToyData.load(args.data)
    post = fi.FlowPosterior(em, data, width=args.width, n_quad=args.n_quad)
    nuts = dict(num_chains=args.chains, init=args.init, seed=args.seed, dense_mass=not args.diag,
                target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth)

    if args.time_only:
        ## two warmup lengths, so the sampler's compile time cancels
        t_short, _ = fh.time_warmup(post, 2, **nuts)
        t_long, steps = fh.time_warmup(post, args.time_only, **nuts)
        per_iter = (t_long - t_short)/(args.time_only - 2)
        n_iter = args.warmup + args.samples
        print("{} warmup iterations x {} chains: {:.2f} s per iteration (all chains), {:.1f} gradients per chain per "
              "iteration (max {}), {:.1f} ms per gradient batch; compile ~{:.0f} s. Projected {} + {}: ~{:.0f} min "
              "(early warmup takes more steps than sampling, so this is pessimistic)".format(
                  args.time_only, args.chains, per_iter, steps.mean(), steps.max(),
                  1e3*per_iter/max(steps.mean(), 1), t_short, args.warmup, args.samples, n_iter*per_iter/60),
              flush=True)
        return

    res = fh.run_nuts(post, num_warmup=args.warmup, num_samples=args.samples, progress=True, **nuts)
    res.save(os.path.join(args.outdir, 'samples.npz'))
    C, D, _ = res.theta.shape
    flat_th, flat_y = res.theta.reshape(C*D, 8), res.y.reshape(C*D, -1)
    ln_post = np.concatenate([np.asarray(post.ln_post(flat_th[i:i+256], flat_y[i:i+256]))
                              for i in range(0, C*D, 256)]).reshape(C, D)

    ## per-parameter diagnostics, and whether each 90% interval holds the truth
    names, truth = fd.CONTEXT_NAMES, data.truth_theta
    summ = res.meta['summary']
    params = {}
    for j, n in enumerate(names):
        s = summ[n]
        lo, hi = np.percentile(res.theta[:, :, j], [5, 95])
        params[n] = dict(truth=float(truth[j]), mean=s['mean'], std=s['std'], r_hat=s['r_hat'], n_eff=s['n_eff'],
                         q05=float(lo), q95=float(hi), truth_in_90=bool(lo <= truth[j] <= hi))
    y_true = np.log10(data.truth_S)
    lo, hi = np.percentile(res.y, [5, 95], axis=(0, 1))
    spectrum = dict(max_r_hat=float(np.max(summ['y']['r_hat'])), min_n_eff=float(np.min(summ['y']['n_eff'])),
                    truth_in_90_fraction=float(np.mean((lo <= y_true) & (y_true <= hi))))
    ## the chosen n_quad's error on 64 posterior draws, against 512 nodes
    pick = np.random.default_rng(args.seed).choice(C*D, size=min(64, C*D), replace=False)
    quad = fi.quadrature_errors(post, flat_th[pick], flat_y[pick], n_quads=(args.n_quad,), ref=512, n_grad=2)
    ex = res.extra
    summary = dict(parameters=params, spectrum=spectrum, divergences=int(ex['diverging'].sum()),
                   tree_depth_mean=float(np.log2(ex['num_steps'] + 1).mean()),
                   num_steps_max=int(ex['num_steps'].max()), step_size=np.unique(ex['adapt_state.step_size']).tolist(),
                   accept_prob_mean=float(ex['accept_prob'].mean()), seconds=res.meta['seconds'],
                   gradients_sampling=res.meta['gradients_sampling'], quadrature_on_posterior=quad[args.n_quad],
                   settings=res.meta['settings'])
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as fh_:
        json.dump(summary, fh_, indent=1)

    plot_corner(os.path.join(args.outdir, 'corner.png'), flat_th, truth, names)
    plot_trace(os.path.join(args.outdir, 'trace.png'), res.theta, ln_post, truth, names)
    plot_spectrum(os.path.join(args.outdir, 'spectrum.png'), data.fs, 10**flat_y, data.truth_S, data.data_psd,
                  data.noise_psd)
    for n, p in params.items():
        print("{:>18}: truth {:.4g}, mean {:.4g} +- {:.3g}, 90% [{:.4g}, {:.4g}] {}, R-hat {:.3f}, ESS {:.0f}".format(
            n, p['truth'], p['mean'], p['std'], p['q05'], p['q95'], 'ok' if p['truth_in_90'] else 'MISS', p['r_hat'],
            p['n_eff']))
    print("spectrum: max R-hat {:.3f}, min ESS {:.0f}, truth inside 90% in {:.0%} of bins".format(
        spectrum['max_r_hat'], spectrum['min_n_eff'], spectrum['truth_in_90_fraction']))
    print("divergences {}, mean tree depth {:.1f}, accept {:.2f}, run {:.0f} s; n_quad {} error on posterior draws: "
          "{:.1e} nats".format(summary['divergences'], summary['tree_depth_mean'], summary['accept_prob_mean'],
                               res.meta['seconds']['run'], args.n_quad, quad[args.n_quad]['value']), flush=True)
    print("wrote", args.outdir)


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
    parser.add_argument('--n_quad', type=int, default=8,
                        help="Gauss-Legendre nodes of the flow's count dequantization. Default 8 (within 2e-4 nats of "
                             "512 nodes near the truth for the production flow; check_flow_posterior.py section 7).")
    parser.add_argument('--width', type=float, default=0.1, help="Logistic width (SNR) of the resolved-GB term.")
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--init', choices=['map', 'prior', 'truth'], default='map')
    parser.add_argument('--diag', action='store_true', help="Diagonal mass matrix (default dense).")
    parser.add_argument('--target_accept', type=float, default=0.8)
    parser.add_argument('--max_tree_depth', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_only', type=int, default=0,
                        help="Time this many warmup iterations and project the full run's cost, then stop.")
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
