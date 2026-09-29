"""
Benchmarks the JAX thresholder (SNR_Threshold.jax_array_sort) against a reference sort
(serial_array_sort or block_array_sort) on identical galaxy draws, and checks parity.

Each population size Ntot runs in its own subprocess, so GPU memory held by one size
(JAX's allocator and cupy's pool both cache) cannot starve the next.

Reported per (Ntot, pre-filter cut, batch size): JAX compile time, steady-state time per
call and per galaxy, JAX device memory per compiled batch (from XLA's memory analysis:
temporaries + arguments + outputs), the survivor capacity and largest survivor count of the
per-bin pre-filter, reference time, and parity (N_res exact, max foreground rel. error).

Usage
-----
    python benchmark_jax_thresholder.py [--Ntot N1 N2 ...] [--n_galaxies G] [--batch_sizes B1 B2 ...]
                                        [--prefilter_snrs none 1]
    python benchmark_jax_thresholder.py --preset h200 --outfile h200.csv    (N = 1e6 x 375 galaxies)
    python benchmark_jax_thresholder.py --forward [...]    (JAX sampling + thresholding vs the backend's
                                                            draw + reference sort; see forward_worker)
"""
import os
import sys
import json
import time
import argparse
import subprocess

PRESETS = {'h200': dict(Ntot=[1e6], n_galaxies=375, batch_sizes=[375, 125, 25])}
FIDUCIAL = dict(m_mu=0.6, m_sigma=0.15, rh_disk=3.31, r_bulge=0.75, q_bd=0.33, a_alpha=0.5)  ## Table 1, arXiv:2604.03390
COLUMNS = ["Ntot", "n_galaxies", "prefilter_snr", "batch_size", "compile_s", "call_s", "per_galaxy_s", "jax_mem_GB",
           "capacity", "n_surv_max", "reference", "reference_s", "Nres_match", "fg_max_rel_err", "error"]


def worker(cfg):
    from pelargir import backend
    backend.set_backend(cfg['backend'])
    xp = backend.xp
    import numpy as np
    import legwork as lw
    import astropy.units as u
    from pelargir.inference import GalacticBinaryPrior
    from pelargir.thresholding import SNR_Threshold
    from pelargir.utils import get_amp_freq, lisa_noise_psd, to_numpy

    Ntot, G = int(cfg['Ntot']), int(cfg['n_galaxies'])
    rows = []

    def sync():
        if backend.GPU:
            xp.cuda.Device().synchronize()

    def free_cupy():
        if backend.GPU:
            xp.get_default_memory_pool().free_all_blocks()

    fbins = xp.arange(cfg['fmin'] - cfg['fbin']/2, cfg['fmax'] + cfg['fbin']/2, cfg['fbin'])
    rx = xp.asarray(lw.psd.approximate_response_function(to_numpy(fbins)*u.Hz, 19.09*u.mHz).value)
    th = SNR_Threshold(fbins, xp.asarray(lisa_noise_psd(fbins)), rx, block_after=cfg['block_after'])

    try:
        gbprior = GalacticBinaryPrior(xp.random.default_rng(cfg['seed']))
        gbprior.condition({k: xp.full(G, v) for k, v in FIDUCIAL.items()})
        draw = gbprior.sample_conditional(Ntot)          ## (4, Ntot, 1, G)
        A, f = get_amp_freq(draw)
        del draw
        obs = xp.array([f, A])                           ## (2, Ntot, 1, G)
        del A, f
        free_cupy()
    except Exception as err:
        return [dict(Ntot=Ntot, n_galaxies=G, error="draw: " + str(err).splitlines()[0][:120])]

    ref = None
    ref_s = float('nan')
    if cfg['reference'] != 'none':
        sort = th.serial_array_sort if cfg['reference'] == 'serial' else th.block_array_sort
        try:
            sync(); t0 = time.time()
            r_Nres, r_fg = sort(obs, fbins, snr_thresh=cfg['snr_thresh'])
            sync(); ref_s = time.time() - t0
            ref = (to_numpy(r_Nres), to_numpy(r_fg))
            del r_Nres, r_fg
        except Exception as err:
            ref_s = float('nan')
            rows_err = "reference: " + str(err).splitlines()[0][:120]
            rows.append(dict(Ntot=Ntot, n_galaxies=G, reference=cfg['reference'], error=rows_err))
        free_cupy()

    import jax
    from pelargir import jax_thresholding
    edges = fbins + 0.5*th.delf
    sd = jax.ShapeDtypeStruct
    f64 = jax.numpy.float64
    nf = int(fbins.shape[0])
    for cut in cfg['prefilter_snrs']:
        cut = None if cut == 'none' else float(cut)
        for B in cfg['batch_sizes']:
            B = int(min(B, G))
            row = dict(Ntot=Ntot, n_galaxies=G, prefilter_snr=str(cut), batch_size=B, reference=cfg['reference'],
                       reference_s=ref_s)
            kw = dict(snr_thresh=cfg['snr_thresh'], batch_size=B, prefilter_snr=cut)
            try:
                th._jax_capacity.clear()
                sync(); t0 = time.time()
                th.jax_array_sort(obs, fbins, **kw)
                sync(); row['compile_s'] = time.time() - t0
                times = []
                for _ in range(cfg['repeats']):
                    sync(); t0 = time.time()
                    j_Nres, j_fg = th.jax_array_sort(obs, fbins, **kw)
                    sync(); times.append(time.time() - t0)
                row['call_s'] = min(times)
                row['per_galaxy_s'] = row['call_s']/G

                capacity = th._jax_capacity.get((Ntot, B))
                if cut is not None:
                    row['capacity'] = capacity
                    consts = [jax_thresholding._to_jax(c) for c in (edges, th.noisePSD, th.LISA_rx)]
                    obs_g = obs.reshape(2, Ntot, G)
                    row['n_surv_max'] = max(int(jax.numpy.max(jax_thresholding._count_survivors(
                        jax_thresholding._to_jax(xp.ascontiguousarray(xp.moveaxis(obs_g[:, :, g0:g0+B], -1, 0))),
                        *consts, float(th.duration), cut))) for g0 in range(0, G, B))
                mem = jax_thresholding._threshold_batch.lower(
                    sd((B, 2, Ntot), f64), sd((nf,), f64), sd((nf,), f64), sd((nf,), f64),
                    sd((), f64), sd((), f64), sd((), f64), sd((), f64),
                    capacity=capacity).compile().memory_analysis()
                row['jax_mem_GB'] = (mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes)/1e9

                if ref is not None:
                    j_Nres, j_fg = to_numpy(j_Nres), to_numpy(j_fg)
                    row['Nres_match'] = bool(np.array_equal(j_Nres, ref[0]))
                    nz = ref[1] > 0
                    row['fg_max_rel_err'] = float(np.max(np.abs(j_fg[nz] - ref[1][nz])/ref[1][nz])) if nz.any() else 0.0
                del j_Nres, j_fg
            except Exception as err:
                row['error'] = str(err).splitlines()[0][:120]
            rows.append(row)
    return rows


def forward_worker(cfg):
    '''--forward: the JAX forward model (sampling + thresholding in one jit) against the
    backend's draw (GalacticBinaryPrior) + reference sort, both timed for the whole call.
    Parity is checked on galaxy 0: serial_array_sort on the forward model's materialized draw.'''
    from pelargir import backend
    backend.set_backend(cfg['backend'])
    xp = backend.xp
    import numpy as np
    import legwork as lw
    import astropy.units as u
    import jax
    import jax.numpy as jnp
    from pelargir import jax_population as jp
    from pelargir import jax_thresholding
    from pelargir.inference import GalacticBinaryPrior
    from pelargir.thresholding import SNR_Threshold
    from pelargir.utils import get_amp_freq, lisa_noise_psd, to_numpy

    Ntot, G = int(cfg['Ntot']), int(cfg['n_galaxies'])
    rows = []

    def sync():
        if backend.GPU:
            xp.cuda.Device().synchronize()

    def free_cupy():
        if backend.GPU:
            xp.get_default_memory_pool().free_all_blocks()

    fbins = xp.arange(cfg['fmin'] - cfg['fbin']/2, cfg['fmax'] + cfg['fbin']/2, cfg['fbin'])
    rx = xp.asarray(lw.psd.approximate_response_function(to_numpy(fbins)*u.Hz, 19.09*u.mHz).value)
    th = SNR_Threshold(fbins, xp.asarray(lisa_noise_psd(fbins)), rx, block_after=cfg['block_after'])
    consts = (fbins + 0.5*th.delf, th.noisePSD, th.LISA_rx, th.duration, th.duration_eff)
    gbprior = GalacticBinaryPrior(xp.random.default_rng(cfg['seed']))
    bounds = jp.prior_bounds(gbprior)
    theta = np.array([FIDUCIAL[k] for k in gbprior.pop_params])
    thetas = np.tile(theta, (G, 1))
    Ns = np.full((1, G), Ntot)
    key = jax.random.key(cfg['seed'])

    n_pad = jp.pad_bucket(Ntot)
    for cut in cfg['prefilter_snrs']:
        cut = None if cut == 'none' else float(cut)
        for B in cfg['batch_sizes']:
            B = int(min(B, G))
            row = dict(Ntot=Ntot, n_galaxies=G, prefilter_snr=str(cut), batch_size=B, reference=cfg['reference'])
            cache = {}
            kw = dict(snr_thresh=cfg['snr_thresh'], batch_size=B, prefilter_snr=cut, capacity_cache=cache,
                      out_module=xp)
            try:
                sync(); t0 = time.time()
                jp.jax_forward_model(key, thetas, Ns, bounds, *consts, **kw)
                sync(); row['compile_s'] = time.time() - t0
                times = []
                for _ in range(cfg['repeats']):
                    sync(); t0 = time.time()
                    j_N, j_fg = jp.jax_forward_model(key, thetas, Ns, bounds, *consts, **kw)
                    sync(); times.append(time.time() - t0)
                row['call_s'] = min(times)
                row['per_galaxy_s'] = row['call_s']/G

                capacity = cache.get((n_pad, B))
                row['capacity'] = capacity
                c_j = [jax_thresholding._to_jax(np.ascontiguousarray(jp._host(c_), dtype=np.float64)) for c_ in consts[:3]]
                mem = jp._forward_batch.lower(
                    jp.galaxy_keys(key, G)[:B], jnp.asarray(thetas[:B]), jnp.asarray(Ns.ravel()[:B]), *c_j,
                    float(th.duration), float(th.duration_eff), jnp.full(B, float(cfg['snr_thresh'])),
                    jnp.full(B, 0.0 if cut is None else cut),
                    bounds, n_pad, capacity=capacity).compile().memory_analysis()
                row['jax_mem_GB'] = (mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes)/1e9

                row['_galaxy0'] = (int(to_numpy(j_N).reshape(-1)[0]), to_numpy(j_fg).reshape(len(fbins), -1)[:, 0])
                del j_N, j_fg
            except Exception as err:
                row['error'] = str(err).splitlines()[0][:120]
            rows.append(row)

    ## the reference and parity draws run after the JAX timings, so their GPU memory can't starve the kernel
    import gc
    gc.collect()
    free_cupy()
    ref_s = float('nan')
    if cfg['reference'] != 'none':
        sort = th.serial_array_sort if cfg['reference'] == 'serial' else th.block_array_sort
        try:
            sync(); t0 = time.time()
            gbprior.condition({k: xp.full(G, v) for k, v in FIDUCIAL.items()})
            draw = gbprior.sample_conditional(Ntot)
            A, f = get_amp_freq(draw)
            del draw
            obs = xp.array([f, A])
            del A, f
            sort(obs, fbins, snr_thresh=cfg['snr_thresh'])
            sync(); ref_s = time.time() - t0
            del obs
        except Exception as err:
            rows.append(dict(Ntot=Ntot, n_galaxies=G, reference=cfg['reference'],
                             error="reference: " + str(err).splitlines()[0][:120]))
        free_cupy()

    parity = None
    try:
        A, f = get_amp_freq(jp.sample_galaxy_draw(key, 0, theta, Ntot, bounds, out_module=xp))
        p_N, p_fg = th.serial_array_sort(xp.array([f, A]), fbins, snr_thresh=cfg['snr_thresh'])
        parity = (int(p_N), to_numpy(p_fg))
        del A, f, p_N, p_fg
    except Exception as err:
        rows.append(dict(Ntot=Ntot, n_galaxies=G, error="parity reference: " + str(err).splitlines()[0][:120]))
    free_cupy()

    for row in rows:
        galaxy0 = row.pop('_galaxy0', None)
        if 'batch_size' in row:
            row['reference_s'] = ref_s
        if galaxy0 is not None and parity is not None:
            row['Nres_match'] = galaxy0[0] == parity[0]
            nz = parity[1] > 0
            row['fg_max_rel_err'] = float(np.max(np.abs(galaxy0[1][nz] - parity[1][nz])/parity[1][nz])) if nz.any() else 0.0
    return rows


def print_table(rows):
    fmt = "{:>9} {:>6} {:>9} {:>6} {:>9} {:>9} {:>11} {:>9} {:>9} {:>9} {:>10} {:>9} {:>11}  {}"
    print(fmt.format("Ntot", "G", "cut", "B", "compile_s", "call_s", "per_gal_ms", "jax_GB", "capacity", "n_surv",
                     "ref_s", "Nres_eq", "fg_rel_err", "error"))
    num = lambda v, spec: "-" if v is None or v != v else format(v, spec)
    for r in rows:
        print(fmt.format(num(r.get('Ntot'), '.0e'), r.get('n_galaxies', '-'), r.get('prefilter_snr', '-'), r.get('batch_size', '-'),
                         num(r.get('compile_s'), '.2f'), num(r.get('call_s'), '.3f'),
                         num(None if r.get('per_galaxy_s') is None else 1e3*r['per_galaxy_s'], '.2f'),
                         num(r.get('jax_mem_GB'), '.3f'), r.get('capacity') or '-', r.get('n_surv_max') or '-',
                         num(r.get('reference_s'), '.2f'),
                         str(r.get('Nres_match', '-')), num(r.get('fg_max_rel_err'), '.1e'), r.get('error', '')))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--_worker':
        cfg = json.loads(sys.argv[2])
        rows = forward_worker(cfg) if cfg.get('forward') else worker(cfg)
        print("ROWS:" + json.dumps(rows))
        sys.exit(0)

    parser = argparse.ArgumentParser(prog='benchmark_jax_thresholder',
                                     description='Time and check the JAX thresholder against a reference sort.')
    parser.add_argument('--preset', type=str, choices=list(PRESETS), default=None,
                        help="Named configuration; 'h200' is Ntot = 1e6 with 375 galaxies (batch sizes 375, 125, 25).")
    parser.add_argument('--Ntot', type=float, nargs='+', default=[1e6, 1e7, 5e7], help='Binaries per galaxy.')
    parser.add_argument('--n_galaxies', type=int, default=1, help='Galaxies per call (Nreal*Nparallel).')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None,
                        help='Galaxies per jitted call. Default: 1 and n_galaxies.')
    parser.add_argument('--backend', type=str, choices=['numpy', 'cupy', 'jax'], default='jax')
    parser.add_argument('--reference', type=str, choices=['serial', 'block', 'none'], default='serial',
                        help="Reference sort for timing/parity. block_array_sort's padding needs far more memory.")
    parser.add_argument('--repeats', type=int, default=3, help='Timed calls per configuration (minimum reported).')
    parser.add_argument('--seed', type=int, default=150914)
    parser.add_argument('--snr_thresh', type=float, default=7.0)
    parser.add_argument('--prefilter_snrs', type=str, nargs='+', default=['none', '1'],
                        help="Per-bin pre-filter SNR cuts to run ('none' = unfiltered).")
    parser.add_argument('--fmin', type=float, default=1e-4)
    parser.add_argument('--fmax', type=float, default=5e-3)
    parser.add_argument('--fbin', type=float, default=2e-5)
    parser.add_argument('--block_after', type=int, default=4)
    parser.add_argument('--forward', action='store_true',
                        help='Benchmark the JAX forward model (sampling + thresholding, jax_population.py) against '
                             "the backend's draw + reference sort, instead of the thresholder alone.")
    parser.add_argument('--outfile', type=str, default=None, help='Optional CSV output path.')
    args = parser.parse_args()

    if args.preset is not None:
        for key, val in PRESETS[args.preset].items():
            setattr(args, key, val)
    batch_sizes = args.batch_sizes if args.batch_sizes is not None else sorted({1, args.n_galaxies})

    rows = []
    for Ntot in args.Ntot:
        cfg = dict(Ntot=Ntot, n_galaxies=args.n_galaxies, batch_sizes=batch_sizes, backend=args.backend,
                   reference=args.reference, repeats=args.repeats, seed=args.seed, snr_thresh=args.snr_thresh,
                   prefilter_snrs=args.prefilter_snrs, forward=args.forward,
                   fmin=args.fmin, fmax=args.fmax, fbin=args.fbin, block_after=args.block_after)
        print("Running Ntot = {:.0e} ...".format(Ntot), flush=True)
        res = subprocess.run([sys.executable, os.path.abspath(__file__), '--_worker', json.dumps(cfg)],
                             capture_output=True, text=True)
        out = [line for line in res.stdout.splitlines() if line.startswith("ROWS:")]
        if out:
            rows.extend(json.loads(out[-1][5:]))
        else:
            err = (res.stderr.strip().splitlines() or ["worker exited with code {}".format(res.returncode)])[-1]
            rows.append(dict(Ntot=int(Ntot), n_galaxies=args.n_galaxies, error=err[:120]))

    print()
    if args.forward:
        print("Forward model (sampling + thresholding); ref_s includes the backend's draw.")
    print_table(rows)
    if args.outfile:
        import csv
        with open(args.outfile, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, '') for c in COLUMNS})
        print("\nWrote {}".format(args.outfile))
