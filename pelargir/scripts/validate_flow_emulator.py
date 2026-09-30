"""
Validates a trained flow emulator (either flow base) against the simulator at held-out contexts: n_sim
simulator realizations and n_flow flow samples per context, compared per band (N_res moments and
two-sample KS) and per bin (log10 S_gw mean offset in units of the simulator scatter, and scatter
ratio). Also reports the mean log_prob of simulator realizations and of flow samples (equal in
expectation if the flow is right), and the Gauss-Legendre convergence of log_prob in n_quad.

Usage (with pelargir installed; or run this file with python)
-----
    pelargir-validate-flows emulator_dir outdir [--n_sim 200] [--n_flow 2000] [--device cuda]
"""
import os
import json
import argparse

FIDUCIAL = [0.6, 0.15, 3.31, 0.75, 0.33, 0.5]  ## Table 1, arXiv:2604.03390
## (Lambda, rho_thresh, lambda_tot)
POINTS = [(FIDUCIAL, rho, lam) for lam in (1e6, 1e7) for rho in (6.0, 8.0, 10.0)] + \
         [([0.3, 0.25, 2.0, 1.5, 0.8, 1.2], 8.0, 3e6), ([1.0, 0.1, 8.0, 0.2, 0.1, -0.3], 8.0, 3e6)]


def main(args):
    ## JAX's default allocator fragments across the many padded sizes and runs out of memory on 8 GB
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    from pelargir import backend
    backend.set_backend('jax')
    jax = backend.import_jax()
    import numpy as np
    import scipy.stats as ss
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pelargir import flow_data

    os.makedirs(args.outdir, exist_ok=True)
    em = flow_data.load_emulator(args.emulator, device=args.device)
    contexts = np.array([list(L) + [rho, np.log10(lam)] for L, rho, lam in POINTS])
    nres_f, psd, _ = flow_data.simulate(jax.random.key(args.seed), contexts, args.n_sim, em.fbins)

    rows = []
    for i, (L, rho, lam) in enumerate(POINTS):
        c_sim = np.repeat(contexts[i:i+1], args.n_sim, axis=0)
        N_sim = np.stack([nres_f[i][:, b.slice].sum(axis=1) for b in em.bands], axis=1)
        S_sim = psd[i]
        N_flow, S_flow = em.sample(np.repeat(contexts[i:i+1], args.n_flow, axis=0))
        lp_sim = em.log_prob(c_sim, N_sim, S_sim, n_quad=args.n_quad)
        lp_flow = em.log_prob(c_sim[:args.n_sim], N_flow[:args.n_sim], S_flow[:args.n_sim], n_quad=args.n_quad)
        err, n_ok = flow_data.quadrature_convergence(em, c_sim[:50], N_sim[:50], S_sim[:50], n_quads=(4, 8, 16, 32, 64, 512))
        for j, b in enumerate(em.bands):
            ls, lf = np.log10(S_sim[:, b.slice]), np.log10(S_flow[:, b.slice])
            rows.append(dict(point=i, rho=rho, lam=lam, band=j, f_lo=float(b.fs[0]), f_hi=float(b.fs[-1]),
                             N_sim_mean=float(N_sim[:, j].mean()), N_flow_mean=float(N_flow[:, j].mean()),
                             N_sim_std=float(N_sim[:, j].std()), N_flow_std=float(N_flow[:, j].std()),
                             N_ks=float(ss.ks_2samp(N_sim[:, j], N_flow[:, j]).statistic),
                             logS_max_mean_offset_sigma=float(np.max(np.abs(lf.mean(0) - ls.mean(0))/ls.std(0))),
                             logS_scatter_ratio_min=float(np.min(lf.std(0)/ls.std(0))),
                             logS_scatter_ratio_max=float(np.max(lf.std(0)/ls.std(0))),
                             quad_err_n16=float(err[j, 2]), quad_err_n32=float(err[j, 3]), quad_err_n64=float(err[j, 4])))
        rows[-1].update(lp_sim_mean=float(lp_sim.mean()), lp_flow_mean=float(lp_flow.mean()),
                        lp_diff_se=float(np.sqrt(lp_sim.var()/lp_sim.size + lp_flow.var()/lp_flow.size)),
                        n_quad_converged=n_ok)
        print("point {} (rho={}, lambda={:.0e}): <log p> sim {:.2f}, flow {:.2f} (se {:.2f}); n_quad ok {}".format(
            i, rho, lam, lp_sim.mean(), lp_flow.mean(), rows[-1]['lp_diff_se'], n_ok), flush=True)

        fig, axes = plt.subplots(2, len(em.bands), figsize=(2.2*len(em.bands), 4.4))
        for j, b in enumerate(em.bands):
            lo, hi = np.percentile(np.concatenate([N_sim[:, j], N_flow[:, j]]), [0.5, 99.5])
            bins = np.linspace(lo, hi + 1, 30)
            axes[0, j].hist(N_sim[:, j], bins=bins, density=True, alpha=0.5, label='sim')
            axes[0, j].hist(N_flow[:, j], bins=bins, density=True, alpha=0.5, label='flow')
            axes[0, j].set_title("{:.2f}-{:.2f} mHz".format(1e3*b.fs[0], 1e3*b.fs[-1]), fontsize=8)
            for S, lab in ((S_sim, 'sim'), (S_flow, 'flow')):
                q = np.percentile(np.log10(S[:, b.slice]), [16, 50, 84], axis=0)
                axes[1, j].plot(1e3*b.fs, q[1], label=lab)
                axes[1, j].fill_between(1e3*b.fs, q[0], q[2], alpha=0.3)
        axes[0, 0].set_ylabel('N_res'); axes[1, 0].set_ylabel('log10 S_gw')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("point {}: rho={}, lambda_tot={:.0e}".format(i, rho, lam), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'point_{}.png'.format(i)), dpi=110)
        plt.close(fig)

    with open(os.path.join(args.outdir, 'validation.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    keys = ['point', 'band', 'N_sim_mean', 'N_flow_mean', 'N_sim_std', 'N_flow_std', 'N_ks',
            'logS_max_mean_offset_sigma', 'logS_scatter_ratio_min', 'logS_scatter_ratio_max']
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
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
