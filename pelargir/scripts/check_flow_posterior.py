"""
Checks the flow toy-model log posterior (flow_posterior.FlowPosterior) before any sampler runs on it:

1. each term at the truth;
2. 1-D slices of ln_post through the truth for each context parameter (the others at the truth);
3. the gradient at the truth against central differences;
4. typicality: the percentile of the truth's flow log density among flow samples at the true context;
5. the smoothing bias of the resolved-GB term, ln_res(w) - ln_res(step), at the truth;
6. timing of ln_post (batches of 1 and 64) and of its gradient;
7. quadrature: the largest change of the log posterior and its gradient with the number of
   Gauss-Legendre nodes, against 512 nodes, at the truth and 64 points around it, and the gradient's
   cost per node count; recommends the smallest n within 0.01 nats. (HMC uses the exact gradient of
   the n-node density, so it samples that density exactly; only its distance from the converged
   density matters. The gradient difference is reported for information.)

Writes report.json, slices.png and quadrature.png to outdir.

Usage
-----
    python -m pelargir.scripts.check_flow_posterior emulator_dir data.npz outdir [--width 0.1] [--n_flow 2000]
"""
import os
import json
import time
import logging
import argparse

from pelargir.log_utils import add_verbosity_args, setup_logging, timed, progress_bar

logger = logging.getLogger('pelargir.scripts.check_flow_posterior')


def _timed(fn, repeats=20):
    '''

    Median wall time of fn() after one warm-up call, blocking on its JAX outputs.

    Arguments
    ------------------------------
    fn : callable
    	Function returning JAX arrays (or a pytree of them).
    repeats : int
    	Timed calls. (Default: 20)

    Returns
    ------------------------------
    seconds : float
    	Median seconds per call.

    '''
    import jax
    import numpy as np
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main(args):
    '''

    Run the checks and write the report.

    Arguments
    ------------------------------
    args : argparse.Namespace
    	Parsed arguments (see cli).

    Returns
    ------------------------------
    None

    '''
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    setup_logging(args.verbose, args.quiet)
    from pelargir import backend
    backend.set_backend('jax')
    jax = backend.import_jax()
    import jax.numpy as jnp
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pelargir.flows import data as fd
    from pelargir.inference import flow_posterior as fi
    from pelargir.utils import set_style
    set_style()
    os.makedirs(args.outdir, exist_ok=True)

    with timed(logger, f"loading the emulator from {args.emulator}"):
        em = fd.load_emulator(args.emulator)
    data = fi.ToyData.load(args.data)
    post = fi.FlowPosterior(em, data, width=args.width, n_quad=args.n_quad)
    th0, y0 = post.truth
    report = dict(emulator=os.path.abspath(args.emulator), data=os.path.abspath(args.data), width=args.width,
                  n_quad=args.n_quad, n_res=int(data.theta_res.shape[0]), N_bands=data.N_bands(em.bands).tolist())

    ## 1. terms at the truth
    with timed(logger, '1. terms at the truth'):
        terms = {k: float(v[0]) for k, v in post.terms(th0, y0).items()}
        terms['ln_post'] = float(post.ln_post(th0, y0)[0])
        report['terms_at_truth'] = terms
        print("terms at the truth:", terms, flush=True)

    ## 2. slices through the truth, each over the prior's 1-99% range (all points in one batch)
    names = fd.CONTEXT_NAMES
    with timed(logger, f"2. slices through the truth ({len(names)} parameters x {args.n_grid} points)"):
        slices = {}
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
        for j, (name, ax) in enumerate(progress_bar(zip(names, axes.ravel()), total=len(names), desc='slices',
                                                     unit='parameter', leave=False)):
            lo, hi = fd.HYPERPRIOR[name].ppf([0.01, 0.99])
            grid = np.linspace(lo, hi, args.n_grid)
            th = np.repeat(np.asarray(th0), args.n_grid, axis=0)
            th[:, j] = grid
            lp = np.asarray(post.ln_post(th, jnp.repeat(y0, args.n_grid, axis=0)))
            best = grid[np.nanargmax(lp)]
            step = grid[1] - grid[0]
            slices[name] = dict(truth=float(th0[0, j]), argmax=float(best), grid_step=float(step),
                                offset_in_steps=float((best - th0[0, j])/step))
            ax.plot(grid, lp - np.nanmax(lp), color='navy')
            ax.axvline(th0[0, j], color='goldenrod', ls='--', label='truth')
            ax.set_xlabel(name.replace('_', ' '))
            ax.set_ylim(max(np.nanmin(lp - np.nanmax(lp)), -200), 5)
        axes[0, 0].set_ylabel(r'$\ln p - \max$')
        axes[0, 0].legend(fontsize=9)
        fig.suptitle('conditional slices of ln_post through the truth (y at the true spectrum)', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'slices.png'), dpi=120)
        plt.close(fig)
        report['slices'] = slices
        for k, v in slices.items():
            print("slice {}: truth {:.4g}, argmax {:.4g} ({:+.1f} grid steps)".format(k, v['truth'], v['argmax'],
                                                                                    v['offset_in_steps']), flush=True)

    ## 3. gradient at the truth against central differences (all perturbed points in one batch)
    with timed(logger, '3. gradient at the truth against central differences'):
        val, (g_th, g_y) = post.value_and_grad(th0[0], y0[0])
        x0 = np.concatenate([np.asarray(th0[0]), np.asarray(y0[0])])
        ## small steps: the flow's splines have kinks, so wider differences mix in curvature across knots
        h = np.concatenate([1e-6*np.maximum(np.abs(np.asarray(th0[0])), 1e-2), np.full(y0.shape[1], 1e-6)])
        X = np.concatenate([x0 + np.diag(h), x0 - np.diag(h)])
        lp = np.asarray(post.ln_post(X[:, :8], X[:, 8:]))
        fd_grad = (lp[:x0.size] - lp[x0.size:])/(2*h)
        ad_grad = np.concatenate([np.asarray(g_th), np.asarray(g_y)])
        err = np.abs(ad_grad - fd_grad)
        rel = err/np.maximum(np.abs(fd_grad), 1e-6)
        ## error relative to the largest gradient component, robust to components near zero
        scaled = lambda s: float(err[s].max()/np.abs(ad_grad[s]).max())
        report['gradient'] = dict(value=float(val), max_rel_err_theta=float(rel[:8].max()),
                                  max_rel_err_y=float(rel[8:].max()), max_scaled_err_theta=scaled(slice(0, 8)),
                                  max_scaled_err_y=scaled(slice(8, None)), grad_theta=ad_grad[:8].tolist())
        print("gradient: max rel. error {:.2e} (theta), {:.2e} (y); max error / max |grad| {:.2e} (theta), {:.2e} (y)".format(
            rel[:8].max(), rel[8:].max(), scaled(slice(0, 8)), scaled(slice(8, None))), flush=True)

    ## 4. typicality of the dataset under the flow at the true context
    with timed(logger, f"4. typicality of the dataset ({args.n_flow} flow samples)"):
        ctx = np.repeat(np.asarray(th0), args.n_flow, axis=0)
        N_s, S_s = em.sample(ctx)
        N_obs = data.N_bands(em.bands)
        band_pct = []
        for j, b in enumerate(progress_bar(em.bands, desc='typicality', unit='band', leave=False)):
            lp_s = em.band_log_prob(j, ctx, N_s[:, j], S_s[:, b.slice])
            lp_t = em.band_log_prob(j, ctx[:1], N_obs[j:j+1], data.truth_S[None, b.slice])[0]
            band_pct.append(float(100*np.mean(lp_s < lp_t)))
        lp_s = em.log_prob(ctx, N_s, S_s)
        lp_t = float(em.log_prob(ctx[:1], N_obs, data.truth_S)[0])
        report['typicality'] = dict(percentile=float(100*np.mean(lp_s < lp_t)), band_percentiles=band_pct,
                                    truth_log_prob=lp_t, sample_log_prob_mean=float(lp_s.mean()),
                                    sample_log_prob_std=float(lp_s.std()))
        print("typicality: the truth's flow log density is at percentile {:.1f} of {} flow samples; per band {}".format(
            report['typicality']['percentile'], args.n_flow, ['{:.0f}'.format(p) for p in band_pct]), flush=True)

    ## 5. smoothing bias of the resolved-GB term
    with timed(logger, '5. smoothing bias of the resolved-GB term'):
        step = float(post.terms(th0, y0, width=None)['res'][0])
        bias = {str(w): float(post.terms(th0, y0, width=w)['res'][0]) - step for w in (0.01, 0.03, 0.1, 0.3, 1.0)}
        report['smoothing'] = dict(ln_res_step=step, bias=bias)
        print("ln_res(step) = {:.3f}; ln_res(w) - ln_res(step):".format(step), bias, flush=True)

    ## 6. timing
    with timed(logger, '6. timing'):
        rng = np.random.default_rng(0)
        th64 = np.repeat(np.asarray(th0), 64, axis=0)*(1 + 1e-3*rng.standard_normal((64, 8)))
        y64 = np.repeat(np.asarray(y0), 64, axis=0)
        report['timing_s'] = dict(ln_post_M1=_timed(lambda: post.ln_post(th0, y0)),
                                  ln_post_M64=_timed(lambda: post.ln_post(th64, y64)),
                                  value_and_grad=_timed(lambda: post.value_and_grad(th0[0], y0[0])))
        print("timing [ms]:", {k: round(1e3*v, 2) for k, v in report['timing_s'].items()}, flush=True)

    ## 7. quadrature nodes: the truth and 64 points around it (theta by 2% of each prior's 1-99% range, y by 0.05 dex)
    if not args.skip_quad:
        with timed(logger, '7. quadrature nodes (65 points, 4-64 nodes against 512)'):
            lo, hi = np.array([fd.HYPERPRIOR[n].ppf([0.01, 0.99]) for n in names]).T
            th_q = np.asarray(th0) + 0.02*(hi - lo)*rng.standard_normal((65, 8))
            th_q[0] = np.asarray(th0[0])
            th_q = np.clip(th_q, lo, hi)
            y_q = np.asarray(y0) + 0.05*rng.standard_normal((65, y0.shape[1]))
            y_q[0] = np.asarray(y0[0])
            n_quads = (4, 8, 16, 32, 64)
            quad = fi.quadrature_errors(post, th_q, y_q, n_quads=n_quads, ref=512, n_grad=4, progress=True)
            for n in progress_bar(n_quads, desc='gradient cost', unit='n_quad', leave=False):
                p_n = fi.FlowPosterior(em, data, width=args.width, n_quad=n)
                quad[n]['grad_ms'] = 1e3*_timed(lambda: p_n.value_and_grad(th0[0], y0[0]), repeats=10)
            ok = [n for n in n_quads if quad[n]['value'] < 0.01]
            report['quadrature'] = dict(points=65, reference=512, errors={str(n): v for n, v in quad.items()},
                                        recommended=min(ok) if ok else None)
            print("quadrature (vs 512 nodes):", {n: {k: '{:.2e}'.format(v) for k, v in e.items()} for n, e in quad.items()},
                  "; recommended n_quad:", report['quadrature']['recommended'], flush=True)
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            ax.loglog(n_quads, [quad[n]['value'] for n in n_quads], 'o-', label=r'max $|\Delta \ln p|$ [nats]')
            ax.loglog(n_quads, [quad[n]['grad'] for n in n_quads], 's-', label=r'max $|\Delta \nabla \ln p| / \max|\nabla \ln p|$')
            ax.axhline(0.01, color='0.5', ls=':', lw=1)
            ax2 = ax.twinx()
            ax2.plot(n_quads, [quad[n]['grad_ms'] for n in n_quads], 'x--', color='firebrick')
            ax2.set_ylabel('gradient cost [ms]', color='firebrick')
            ax.set_xlabel('Gauss-Legendre nodes')
            ax.legend(fontsize=9)
            ax.set_title('quadrature error vs 512 nodes (65 points)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, 'quadrature.png'), dpi=120)
            plt.close(fig)

    with open(os.path.join(args.outdir, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
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
    parser.add_argument('--width', type=float, default=0.1, help="Logistic width (SNR) of the resolved-GB term.")
    parser.add_argument('--n_quad', type=int, default=32)
    parser.add_argument('--n_flow', type=int, default=2000, help="Flow samples for the typicality check.")
    parser.add_argument('--n_grid', type=int, default=101, help="Points per parameter slice.")
    parser.add_argument('--skip_quad', action='store_true', help="Skip the quadrature-node check (section 7).")
    add_verbosity_args(parser)
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
