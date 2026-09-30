"""
Compares the zuko (flows.py) and flax (flax_flows.py) flow bases on the same training set, band
by band: training speed, evaluation speed and fidelity. Needs torch, zuko, flax, distrax, optax.

For each configuration, zuko (float32) and flax (float32 and float64) are trained with identical
data, validation split, epochs, batch size and learning rate:

    small : zuko NSF 3 transforms x 2x60 hidden   | flax coupling 4 layers x 2x50  | lr 1e-3, batch 64, 8 epochs
    atlas : zuko NSF 4 transforms x 2x128 hidden  | flax coupling 4 layers x 2x128 | lr 1e-4, batch 256, 30 epochs

Reported: parameters per band; training wall time (total; band 0, which includes compilation;
mean of the other bands) and steady-state time per step; full-emulator log_prob time (n_quad = 32)
at batch sizes 1, 375 and 1e4 and sampling time at 1e4; held-out NLL in physical units; and, at
validate_flow_emulator's 8 points (simulated once and cached), N_res KS, log S mean offset (in
simulator sigma) and scatter ratio per band, the mean log p gap between simulator draws and flow
samples, and quadrature convergence.

Usage
-----
    python benchmark_flow_bases.py train.npz heldout.npz outdir [--configs small atlas] [--bases zuko flax32 flax64]
"""
import os
import sys
import json
import time
import argparse

CONFIGS = {
    'small': dict(zuko=dict(transforms=3, hidden_size=60), flax=dict(flow_num_layers=4, hidden_size=50),
                  train=dict(lr=1e-3, batch_size=64, n_epochs=8)),
    'atlas': dict(zuko=dict(transforms=4, hidden_size=128), flax=dict(flow_num_layers=4, hidden_size=128),
                  train=dict(lr=1e-4, batch_size=256, n_epochs=30)),
}
BASES = {'zuko': ('zuko', None), 'flax32': ('flax', 'float32'), 'flax64': ('flax', 'float64')}


def main(args):
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    from pelargir import backend
    backend.set_backend('jax')
    jax = backend.import_jax()
    import numpy as np
    import scipy.stats as ss
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pelargir import flow_data as fd
    from pelargir import flows
    from pelargir import flax_flows
    from pelargir.scripts.validate_flow_emulator import POINTS

    os.makedirs(args.outdir, exist_ok=True)
    ts = fd.TrainingSet.load(args.train)
    ho = fd.TrainingSet.load(args.heldout)
    bands = fd.make_bands(ts.fs)
    N_ho = np.stack([fd.band_view(ho, b)[1] for b in bands], axis=1).astype(np.float64)

    ## simulator draws at the validation points, cached
    sim_path = os.path.join(args.outdir, 'validation_sims.npz')
    contexts = np.array([list(L) + [rho, np.log10(lam)] for L, rho, lam in POINTS])
    if os.path.exists(sim_path):
        d = np.load(sim_path)
        nres_f, psd = d['nres_f'], d['psd']
    else:
        t0 = time.time()
        nres_f, psd, _ = fd.simulate(jax.random.key(args.seed), contexts, args.n_sim, ts.fbins)
        np.savez(sim_path, nres_f=nres_f, psd=psd)
        print("simulated validation points in {:.0f} s".format(time.time() - t0), flush=True)

    def sync(base):
        if base == 'zuko' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(fn, base, repeats):
        fn(); sync(base)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); sync(base); times.append(time.perf_counter() - t0)
        return float(np.median(times))

    results = []
    for cname in args.configs:
        cfg = CONFIGS[cname]
        for bname in args.bases:
            base, dtype = BASES[bname]
            label = '{}/{}'.format(cname, bname)
            print("=== {}".format(label), flush=True)
            row = dict(config=cname, base=bname, **cfg['train'])
            ## training, band by band
            if base == 'zuko':
                em = flows.BandedFlowEmulator(bands, ts.fbins, **cfg['zuko']).to(args.device)
                trainer = flows.train_band_flow
                row['params_per_band'] = [int(sum(p.numel() for p in f.flow.parameters())) for f in em.flows]
            else:
                em = flax_flows.BandedFlowEmulator(bands, ts.fbins, seed=args.seed, dtype=dtype, **cfg['flax'])
                trainer = flax_flows.train_band_flow
                row['params_per_band'] = em.n_params()
            band_s, hist = [], []
            for f in em.flows:
                c, N, S = fd.band_view(ts, f.band)
                t0 = time.perf_counter()
                hist.append(trainer(f, c, N, S, seed=args.seed, progress=False, **cfg['train']))
                sync(base)
                band_s.append(time.perf_counter() - t0)
            if base == 'zuko':
                em.eval()
            n_trn = ts.context.shape[0] - int(round(0.1*ts.context.shape[0]))
            steps = cfg['train']['n_epochs']*(n_trn//cfg['train']['batch_size'] if base == 'flax'
                                              else -(-n_trn//cfg['train']['batch_size']))
            row.update(train_total_s=float(sum(band_s)), train_band0_s=band_s[0],
                       train_band_steady_s=float(np.mean(band_s[1:])), steps_per_band=int(steps),
                       step_ms=1e3*float(np.mean(band_s[1:]))/steps,
                       final_train_loss=[h['train'][-1] for h in hist], final_val_loss=[h['val'][-1] for h in hist],
                       val_curves=[h['val'] for h in hist])
            em.save(os.path.join(args.outdir, cname + '_' + bname))
            print("  trained in {:.1f} s ({:.2f} ms/step steady)".format(row['train_total_s'], row['step_ms']), flush=True)

            ## held-out NLL in physical units
            lp_bands = np.stack([em.band_log_prob(j, ho.context, N_ho[:, j], ho.psd[:, b.slice], n_quad=32)
                                 for j, b in enumerate(bands)], axis=1)
            row['heldout_nll_per_band'] = (-lp_bands.mean(axis=0)).tolist()
            row['heldout_nll_total'] = float(-lp_bands.sum(axis=1).mean())

            ## evaluation speed
            for M in (1, 375, 10000):
                idx = np.arange(M) % ho.context.shape[0]
                row['log_prob_ms_M{}'.format(M)] = 1e3*timed(
                    lambda: em.log_prob(ho.context[idx], N_ho[idx], ho.psd[idx], n_quad=32), base, args.repeats)
            idx = np.arange(10000) % ho.context.shape[0]
            row['sample_ms_M10000'] = 1e3*timed(lambda: em.sample(ho.context[idx]), base, args.repeats)

            ## fidelity against the simulator
            ks, off, ratio, gaps = [], [], [], []
            for i in range(len(POINTS)):
                c_sim = np.repeat(contexts[i:i+1], args.n_sim, axis=0)
                N_sim = np.stack([nres_f[i][:, b.slice].sum(axis=1) for b in bands], axis=1)
                N_fl, S_fl = em.sample(np.repeat(contexts[i:i+1], args.n_flow, axis=0))
                lp_sim = em.log_prob(c_sim, N_sim, psd[i], n_quad=32)
                lp_fl = em.log_prob(c_sim, N_fl[:args.n_sim], S_fl[:args.n_sim], n_quad=32)
                gaps.append(float(lp_sim.mean() - lp_fl.mean()))
                for j, b in enumerate(bands):
                    ls, lf = np.log10(psd[i][:, b.slice]), np.log10(S_fl[:, b.slice])
                    ks.append(float(ss.ks_2samp(N_sim[:, j], N_fl[:, j]).statistic))
                    off.append(float(np.max(np.abs(lf.mean(0) - ls.mean(0))/ls.std(0))))
                    ratio.append(float(np.median(lf.std(0)/ls.std(0))))
            row.update(ks_mean=float(np.mean(ks)), ks_max=float(np.max(ks)), logS_offset_sigma_median=float(np.median(off)),
                       logS_offset_sigma_max=float(np.max(off)), scatter_ratio_median=float(np.median(ratio)),
                       scatter_ratio_range=[float(np.min(ratio)), float(np.max(ratio))],
                       lp_gap_per_point=gaps, lp_gap_mean=float(np.mean(gaps)))
            err, n_ok = fd.quadrature_convergence(em, ho.context[:50], N_ho[:50], ho.psd[:50],
                                                  n_quads=(4, 8, 16, 32, 64, 512))
            row.update(quad_err_max_by_n={n: float(e) for n, e in zip((4, 8, 16, 32, 64), err.max(axis=0)[:-1])},
                       n_quad_converged=n_ok)
            print("  held-out NLL {:.2f}; KS mean {:.3f}; scatter ratio {:.2f}; log p gap {:.2f}; log_prob(375) {:.1f} ms".format(
                row['heldout_nll_total'], row['ks_mean'], row['scatter_ratio_median'], row['lp_gap_mean'],
                row['log_prob_ms_M375']), flush=True)
            results.append(row)
            with open(os.path.join(args.outdir, 'results.json'), 'w') as fh:
                json.dump(results, fh, indent=1)

    ## summary table and plots
    keys = ['config', 'base', 'params_per_band', 'train_total_s', 'step_ms', 'heldout_nll_total', 'ks_mean',
            'scatter_ratio_median', 'logS_offset_sigma_median', 'lp_gap_mean', 'log_prob_ms_M1', 'log_prob_ms_M375',
            'log_prob_ms_M10000', 'sample_ms_M10000', 'n_quad_converged']
    with open(os.path.join(args.outdir, 'summary.csv'), 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in results:
            fh.write(','.join(str(r[k][0] if k == 'params_per_band' else r[k]) for k in keys) + '\n')
    for cname in args.configs:
        rows = [r for r in results if r['config'] == cname]
        fig, axes = plt.subplots(1, len(bands), figsize=(2.2*len(bands), 2.4))
        for r in rows:
            for j, ax in enumerate(axes):
                ax.plot(r['val_curves'][j], label=r['base'])
        for j, (ax, b) in enumerate(zip(axes, bands)):
            ax.set_title("{:.2f}-{:.2f} mHz".format(1e3*b.fs[0], 1e3*b.fs[-1]), fontsize=8)
            ax.set_xlabel('epoch')
        axes[0].set_ylabel('validation NLL'); axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, 'val_curves_{}.png'.format(cname)), dpi=110)
        plt.close(fig)
    print("wrote", args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train')
    parser.add_argument('heldout')
    parser.add_argument('outdir')
    parser.add_argument('--configs', nargs='+', default=['small', 'atlas'], choices=list(CONFIGS))
    parser.add_argument('--bases', nargs='+', default=list(BASES), choices=list(BASES))
    parser.add_argument('--n_sim', type=int, default=200)
    parser.add_argument('--n_flow', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    main(parser.parse_args())
