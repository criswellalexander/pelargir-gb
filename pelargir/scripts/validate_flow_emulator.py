"""
Validates a trained flow emulator (either flow base) against the simulator at held-out contexts: n_sim
simulator realizations and n_flow flow samples per context, compared per band (N_res moments and
two-sample KS) and per bin (in the flows' variable x = log10(S_gw + C): mean offset in units of the
simulator scatter, and scatter ratio; and the probability that S_gw = 0), and over the full band
(total N_res moments and KS). Also reports the mean log_prob of simulator realizations and of flow
samples (equal in expectation if the flow is right), the Gauss-Legendre convergence of log_prob in
n_quad (rows with S_gw > 0) and the scrambled-Sobol convergence for rows with zero bins.

Figures per validation point i: point_{i}.png (N_res and x per band; at most 12 bands, evenly spaced),
point_{i}_spectrum.png (full-band S_gw + C against f, log-log, simulator and flow medians with 68%
and 95% bands, over the LISA noise and C; below, P(S_gw = 0) per bin, simulator and flow) and
point_{i}_nres.png (total N_res histograms).

Usage (with pelargir installed; or run this file with python)
-----
    pelargir-validate-flows emulator_dir outdir [--n_sim 200] [--n_flow 2000] [--device cuda] [--weights best|final]
"""
import os
import json
import logging
import argparse

from pelargir.log_utils import add_verbosity_args, setup_logging, timed, progress_bar

logger = logging.getLogger('pelargir.scripts.validate_flow_emulator')

FIDUCIAL = [0.6, 0.15, 3.31, 0.75, 0.33, 0.5]  ## Table 1, arXiv:2604.03390
## (Lambda, rho_thresh, lambda_tot)
POINTS = [(FIDUCIAL, rho, lam) for lam in (1e6, 1e7) for rho in (6.0, 8.0, 10.0)] + \
         [([0.3, 0.25, 2.0, 1.5, 0.8, 1.2], 8.0, 3e6), ([1.0, 0.1, 8.0, 0.2, 0.1, -0.3], 8.0, 3e6)]


def _tex_sci(x):
    '''x as TeX scientific notation, e.g. 3\\times10^{6}.'''
    m, e = '{:.1e}'.format(x).split('e')
    m = m.rstrip('0').rstrip('.')
    return r'{}10^{{{}}}'.format('' if m == '1' else m + r'\times', int(e))


def point_title(i, L, rho, lam):
    '''Two-line TeX title naming validation point i.'''
    return (r'point {}: $\rho_{{\rm thresh}} = {}$, $\lambda_{{\rm tot}} = {}$'.format(i, rho, _tex_sci(lam))
            + '\n' + r'$\Lambda = [{}]$'.format(', '.join('{:g}'.format(v) for v in L)))


def plot_spectrum(path, fs, S_sim, S_flow, noise, floor, title):
    '''

    Full-band S_gw + C against f (log-log; C the flows' floor, so zero-foreground draws sit at C): median, 68%
    and 95% bands of the simulator and the flow, over the noise; below, P(S_gw = 0) per bin for both.

    Arguments
    ------------------------------
    path : str
    	Output file.
    fs : numpy.ndarray
    	(Nf',) bin frequencies [Hz].
    S_sim, S_flow : numpy.ndarray
    	(n, Nf') simulator and flow S_gw draws (NaN outside the bands).
    noise : numpy.ndarray
    	(Nf',) LISA noise PSD.
    floor : float
    	C [1/Hz] (0 for a pre-Stage-3 emulator).
    title : str
    	Figure title.

    Returns
    ------------------------------
    None

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ## the noise sits beneath the GW spectra
    ax.loglog(fs, noise, ls='--', color='0.5', lw=1.2, label='LISA noise', zorder=1)
    if floor > 0:
        ax.axhline(floor, color='0.3', ls=':', lw=1, label=r'$C$ ($S_{\rm GW} = 0$)', zorder=1)
    for k, (S, label, color) in enumerate(((S_sim, 'sim', 'C0'), (S_flow, 'flow', 'C1'))):
        q = np.nanpercentile(S + floor, [2.5, 16, 50, 84, 97.5], axis=0)
        ax.fill_between(fs, q[0], q[4], color=color, alpha=0.15, lw=0, zorder=2 + k)
        ax.fill_between(fs, q[1], q[3], color=color, alpha=0.3, lw=0, zorder=2 + k)
        ax.plot(fs, q[2], color=color, label=label + ' (median, 68%, 95%)', zorder=4 + k)
        ax1.step(fs, np.nanmean(np.where(np.isnan(S), np.nan, S == 0), axis=0), where='mid', color=color, label=label)
    ax.set_ylabel(r'$S_{\rm GW} + C$ [1/Hz]')
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    ax1.set_xlabel('f [Hz]')
    ax1.set_ylabel(r'$P(S_{\rm GW} = 0)$')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_total_nres(path, N_sim, N_flow, title):
    '''Density histograms of the total N_res over all bands, simulator and flow.'''
    import numpy as np
    import matplotlib.pyplot as plt
    lo, hi = np.percentile(np.concatenate([N_sim, N_flow]), [0.5, 99.5])
    bins = np.linspace(lo, hi + 1, 40)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.hist(N_sim, bins=bins, density=True, alpha=0.5, color='C0', label='sim')
    ax.hist(N_flow, bins=bins, density=True, alpha=0.5, color='C1', label='flow')
    ax.set_xlabel(r'total $N_{\rm res}$')
    ax.set_ylabel('density')
    ax.legend(fontsize=10)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    ## JAX's default allocator fragments across the many padded sizes and runs out of memory on 8 GB
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    setup_logging(args.verbose, args.quiet)
    from pelargir import backend
    backend.set_backend('jax')
    jax = backend.import_jax()
    import numpy as np
    import scipy.stats as ss
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pelargir.flows import data as flow_data
    from pelargir.utils import lisa_noise_psd, set_style
    set_style()

    os.makedirs(args.outdir, exist_ok=True)
    with timed(logger, f"loading the emulator from {args.emulator} ({args.weights} weights)"):
        em = flow_data.load_emulator(args.emulator, device=args.device, weights=args.weights)
    contexts = np.array([list(L) + [rho, np.log10(lam)] for L, rho, lam in POINTS])
    nres_f, psd, _ = flow_data.simulate(jax.random.key(args.seed), contexts, args.n_sim, em.fbins,
                                        progress=not args.quiet)
    noise = lisa_noise_psd(em.fs, cpu=True)
    ## C of the flows' variable x = log10(S_gw + C)
    floor = float(em.floor)
    logger.info("emulator: %d bands on %d bins up to %.3g Hz, floor C = %.4g", len(em.bands), em.fs.size, em.fs[-1], floor)

    rows = []
    for i, (L, rho, lam) in enumerate(progress_bar(POINTS, desc='validation points', unit='point')):
        c_sim = np.repeat(contexts[i:i+1], args.n_sim, axis=0)
        N_sim = np.stack([nres_f[i][:, b.slice].sum(axis=1) for b in em.bands], axis=1)
        S_sim = psd[i]
        N_flow, S_flow = em.sample(np.repeat(contexts[i:i+1], args.n_flow, axis=0))
        lp_sim = em.log_prob(c_sim, N_sim, S_sim, n_quad=args.n_quad)
        lp_flow = em.log_prob(c_sim[:args.n_sim], N_flow[:args.n_sim], S_flow[:args.n_sim], n_quad=args.n_quad)
        err, n_ok = flow_data.quadrature_convergence(em, c_sim[:50], N_sim[:50], S_sim[:50], n_quads=(4, 8, 16, 32, 64, 512),
                                                     progress=not args.quiet)
        ## rows with zero bins: the joint scrambled-Sobol integral, against 4096 points
        qerr, k_ok = flow_data.qmc_convergence(em, c_sim[:20], N_sim[:20], S_sim[:20], progress=not args.quiet)
        ## probability of a zero-foreground bin, simulator and flow
        z_sim, z_flow = np.mean(S_sim == 0, axis=0), np.mean(S_flow == 0, axis=0)
        for j, b in enumerate(em.bands):
            ls, lf = flow_data.log10_shifted(S_sim[:, b.slice], floor), flow_data.log10_shifted(S_flow[:, b.slice], floor)
            ## bins the simulator never moves (always S_gw = 0) have no scatter to compare against
            sd = ls.std(0)
            ok = sd > 0
            rows.append(dict(point=i, rho=rho, lam=lam, band=j, f_lo=float(b.fs[0]), f_hi=float(b.fs[-1]),
                             N_sim_mean=float(N_sim[:, j].mean()), N_flow_mean=float(N_flow[:, j].mean()),
                             N_sim_std=float(N_sim[:, j].std()), N_flow_std=float(N_flow[:, j].std()),
                             N_ks=float(ss.ks_2samp(N_sim[:, j], N_flow[:, j]).statistic),
                             x_max_mean_offset_sigma=float(np.max(np.abs(lf.mean(0) - ls.mean(0))[ok]/sd[ok])) if ok.any() else float('nan'),
                             x_scatter_ratio_min=float(np.min(lf.std(0)[ok]/sd[ok])) if ok.any() else float('nan'),
                             x_scatter_ratio_max=float(np.max(lf.std(0)[ok]/sd[ok])) if ok.any() else float('nan'),
                             zero_frac_sim_mean=float(z_sim[b.slice].mean()), zero_frac_flow_mean=float(z_flow[b.slice].mean()),
                             quad_err_n16=float(err[j, 2]), quad_err_n32=float(err[j, 3]), quad_err_n64=float(err[j, 4]),
                             qmc_err_k256=float(qerr[j, 1])))
        ## bands are independent given the context, so a row's band draws sum to a draw of the total
        Nt_sim, Nt_flow = N_sim.sum(axis=1), N_flow.sum(axis=1)
        rows[-1].update(lp_sim_mean=float(lp_sim.mean()), lp_flow_mean=float(lp_flow.mean()),
                        lp_diff_se=float(np.sqrt(lp_sim.var()/lp_sim.size + lp_flow.var()/lp_flow.size)),
                        n_quad_converged=n_ok, N_total_sim_mean=float(Nt_sim.mean()),
                        N_total_sim_std=float(Nt_sim.std()), N_total_flow_mean=float(Nt_flow.mean()),
                        N_total_flow_std=float(Nt_flow.std()),
                        N_total_ks=float(ss.ks_2samp(Nt_sim, Nt_flow).statistic),
                        zero_frac_max_abs_diff=float(np.max(np.abs(z_flow - z_sim))), n_qmc_converged=k_ok)
        logger.info("point %d (rho=%g, lambda=%.0e): <log p> sim %.2f, flow %.2f (se %.2f); n_quad ok %s, n_qmc ok %s; "
                    "max |P(S=0) flow - sim| %.3f", i, rho, lam, lp_sim.mean(), lp_flow.mean(), rows[-1]['lp_diff_se'],
                    n_ok, k_ok, rows[-1]['zero_frac_max_abs_diff'])

        ## at most 12 bands, evenly spaced, so a full-band emulator still gives a readable figure
        shown = np.unique(np.linspace(0, len(em.bands) - 1, min(12, len(em.bands))).round().astype(int))
        fig, axes = plt.subplots(2, len(shown), figsize=(2.2*len(shown), 4.4), squeeze=False)
        for col, j in enumerate(shown):
            b = em.bands[j]
            lo, hi = np.percentile(np.concatenate([N_sim[:, j], N_flow[:, j]]), [0.5, 99.5])
            bins = np.linspace(lo, hi + 1, 30)
            axes[0, col].hist(N_sim[:, j], bins=bins, density=True, alpha=0.5, label='sim')
            axes[0, col].hist(N_flow[:, j], bins=bins, density=True, alpha=0.5, label='flow')
            axes[0, col].set_title("{:.2f}-{:.2f} mHz".format(1e3*b.fs[0], 1e3*b.fs[-1]), fontsize=8)
            for S, lab in ((S_sim, 'sim'), (S_flow, 'flow')):
                q = np.percentile(flow_data.log10_shifted(S[:, b.slice], floor), [16, 50, 84], axis=0)
                axes[1, col].plot(1e3*b.fs, q[1], label=lab)
                axes[1, col].fill_between(1e3*b.fs, q[0], q[2], alpha=0.3)
            if floor > 0:
                axes[1, col].axhline(np.log10(floor), color='0.3', ls=':', lw=0.8)
        axes[0, 0].set_ylabel(r'$N_{\rm res}$'); axes[1, 0].set_ylabel(r'$x = \log_{10}(S_{\rm GW} + C)$')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(point_title(i, L, rho, lam).replace('\n', ', '), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'point_{}.png'.format(i)), dpi=110)
        plt.close(fig)

        title = point_title(i, L, rho, lam)
        plot_spectrum(os.path.join(args.outdir, 'point_{}_spectrum.png'.format(i)), em.fs, S_sim, S_flow, noise, floor, title)
        r = rows[-1]
        plot_total_nres(os.path.join(args.outdir, 'point_{}_nres.png'.format(i)), Nt_sim, Nt_flow,
                        title + "\n" + r"sim ${:.0f} \pm {:.0f}$, flow ${:.0f} \pm {:.0f}$, KS ${:.3f}$".format(
                            r['N_total_sim_mean'], r['N_total_sim_std'], r['N_total_flow_mean'],
                            r['N_total_flow_std'], r['N_total_ks']))

    with open(os.path.join(args.outdir, 'validation.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    logger.info("wrote validation.json and the per-point plots to %s", args.outdir)
    keys = ['point', 'band', 'N_sim_mean', 'N_flow_mean', 'N_sim_std', 'N_flow_std', 'N_ks',
            'x_max_mean_offset_sigma', 'x_scatter_ratio_min', 'x_scatter_ratio_max', 'zero_frac_sim_mean',
            'zero_frac_flow_mean']
    print(" ".join(keys))
    for r in rows:
        print(" ".join("{:.3g}".format(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys))


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('emulator', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--n_sim', type=int, default=200)
    parser.add_argument('--n_flow', type=int, default=2000)
    parser.add_argument('--n_quad', type=int, default=32)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weights', choices=['best', 'final'], default='best',
                        help="Best-validation (default) or last-epoch weights.")
    add_verbosity_args(parser)
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
