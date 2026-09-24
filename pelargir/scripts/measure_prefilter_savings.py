"""
Measures naive-SNR pre-filter survivor fraction and the projected block_array_sort-
style dense-array memory footprint at realistic Ntot, to check against H200 GPU
memory (141GB). Avoids running the full per-bin sort at large Ntot -- prefiltering
is O(Ntot) and stays tractable well past ~5e7.

Usage
-----
    python measure_prefilter_savings.py [--Ntot N1 N2 ...] [options]

Example
-------
    python measure_prefilter_savings.py --Ntot 1e6 1e7 5e7 --time_full_sort
"""
import os
import sys
import time
import argparse

H200_MEMORY_BYTES = 141e9  # H200 GPU memory, for context in the printed report
BYTES_PER_ELEMENT = 8      # float64, the default dtype used throughout pelargir


def measure_one(Ntot, pop_theta, gbprior, thresher, fbins, block_after, snr_thresh, xp):
    Ntot = int(Ntot)

    gbprior.condition(pop_theta)
    galaxy_draw = gbprior.sample_conditional(Ntot)
    amp_draws, fgw_draws = get_amp_freq(galaxy_draw)

    amps, f_idx = thresher.coarsegrain_bin(xp.array([fgw_draws, amp_draws]), fbins)
    survive_mask = thresher.naive_snr_survives(amps, snr_thresh=snr_thresh)

    Nf = len(fbins)
    ## replicate block_array_sort's dense-array sizing (post-block_after bins only)
    before_counts = xp.array([int(xp.sum(f_idx == ii)) for ii in range(block_after, Nf)])
    after_counts = xp.array([int(xp.sum((f_idx == ii) & survive_mask)) for ii in range(block_after, Nf)])

    survivor_fraction = float(xp.mean(survive_mask))
    max_before = int(xp.max(before_counts))
    max_after = int(xp.max(after_counts))
    nbins_dense = Nf - block_after

    mem_before = max_before * nbins_dense * BYTES_PER_ELEMENT
    mem_after = max_after * nbins_dense * BYTES_PER_ELEMENT

    return {
        "Ntot": Ntot,
        "survivor_fraction": survivor_fraction,
        "max_counts_before": max_before,
        "max_counts_after": max_after,
        "mem_before_GB": mem_before / 1e9,
        "mem_after_GB": mem_after / 1e9,
        "fits_h200_before": mem_before < H200_MEMORY_BYTES,
        "fits_h200_after": mem_after < H200_MEMORY_BYTES,
    }


def print_report(rows):
    header = "{:>12} | {:>10} | {:>14} | {:>14} | {:>12} | {:>12} | {:>10} | {:>10}".format(
        "Ntot", "surv.frac", "maxcnt(before)", "maxcnt(after)", "mem before", "mem after",
        "fit(bef)", "fit(aft)")
    print(header)
    print("-" * len(header))
    for row in rows:
        print("{:>12.3g} | {:>10.4f} | {:>14d} | {:>14d} | {:>10.2f}GB | {:>10.2f}GB | {:>10} | {:>10}".format(
            row["Ntot"], row["survivor_fraction"], row["max_counts_before"], row["max_counts_after"],
            row["mem_before_GB"], row["mem_after_GB"],
            "yes" if row["fits_h200_before"] else "NO", "yes" if row["fits_h200_after"] else "NO"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='measure_prefilter_savings',
        description='Measure naive-SNR pre-filter survivor fraction and projected '
                     'dense-array memory footprint at realistic Ntot.')

    parser.add_argument('--Ntot', type=float, nargs='+', default=[1e6, 1e7, 5e7],
                         help='Population sizes to measure. Default sweeps up to the '
                              'upper end of the realistic ~5e6-5e7 range.')
    parser.add_argument('--pelargirpath', type=str,
                         default='/home/awc/Documents/LISA/projects/lisa_population_inference/pelargir-gb/pelargir/',
                         help='Directory containing the pelargir package.')
    parser.add_argument('--gpu', action='store_true', help='Run on GPU (cupy) if available.')
    parser.add_argument('--seed', type=int, default=150914, help='RNG seed for the population draw.')
    parser.add_argument('--fmin', type=float, default=1e-4)
    parser.add_argument('--fmax', type=float, default=5e-3)
    parser.add_argument('--fbin', type=float, default=2e-5)
    parser.add_argument('--snr_thresh', type=float, default=7.0)
    parser.add_argument('--block_after', type=int, default=4)
    parser.add_argument('--outfile', type=str, default=None, help='Optional path to write results as CSV.')
    parser.add_argument('--time_full_sort', action='store_true',
                         help='At the smallest given Ntot only, additionally time/cross-check '
                              'the full block_array_sort with and without the pre-filter, as a '
                              'sanity re-confirmation of exact-match at tractable scale.')
    parser.add_argument('--random_hyperprior', action='store_true',
                         help='Draw random population hyperparameters instead of using the '
                              'fixed fiducial values (arXiv:2604.03390 toy model).')

    args = parser.parse_args()

    os.environ['PELARGIR_GPU'] = '1' if args.gpu else '0'
    sys.path.insert(1, args.pelargirpath)

    from inference import GalacticBinaryPrior, PopulationHyperPrior
    from utils import get_amp_freq
    from thresholding import SNR_Threshold
    from utils import lisa_noise_psd
    import legwork as lw
    import astropy.units as u
    if args.gpu:
        import cupy as xp
    else:
        import numpy as xp

    fbins = xp.arange(args.fmin - args.fbin / 2, args.fmax + args.fbin / 2, args.fbin)
    noisePSD = xp.asarray(lisa_noise_psd(fbins))
    fbins_cpu = xp.asnumpy(fbins) if args.gpu else fbins
    lisa_rx = xp.asarray(lw.psd.approximate_response_function(fbins_cpu * u.Hz, 19.09 * u.mHz).value)
    thresher = SNR_Threshold(fbins, noisePSD, lisa_rx, block_after=args.block_after)

    if args.random_hyperprior:
        _hyperprior_for_sample = PopulationHyperPrior(xp.random.default_rng(args.seed))
        _fixed_pop_theta = _hyperprior_for_sample.sample(1)
        print("Using a random hyperprior draw for population hyperparameters.")

        def make_pop_theta():
            return {k: xp.array(v) for k, v in _fixed_pop_theta.items()}
    else:
        ## fiducial values from arXiv:2604.03390's toy model (Table 1)
        print("Using fixed fiducial population hyperparameters (arXiv:2604.03390 toy model).")

        def make_pop_theta():
            return {'m_mu': xp.array([0.6]), 'm_sigma': xp.array([0.15]),
                    'rh_disk': xp.array([3.31]), 'r_bulge': xp.array([0.75]),
                    'q_bd': xp.array([0.33]), 'a_alpha': xp.array([0.5])}

    print("Frequency resolution {:.2e} Hz, {} bins ({} beyond block_after={}).".format(
        args.fbin, fbins.size, fbins.size - args.block_after, args.block_after))
    print("H200 GPU memory: {:.0f} GB\n".format(H200_MEMORY_BYTES / 1e9))

    rows = []
    for Ntot in args.Ntot:
        ## fresh GalacticBinaryPrior + pop_theta per Ntot: re-using either across
        ## sample_conditional calls hits a pre-existing state-reuse bug in
        ## distributions.py (see follow-up task), unrelated to the pre-filter.
        gbprior = GalacticBinaryPrior(xp.random.default_rng(args.seed))
        pop_theta = make_pop_theta()
        t0 = time.time()
        row = measure_one(Ntot, pop_theta, gbprior, thresher, fbins, args.block_after,
                           args.snr_thresh, xp)
        row["prefilter_seconds"] = time.time() - t0
        rows.append(row)
        print("Ntot={:.3g}: prefilter took {:.2f}s".format(Ntot, row["prefilter_seconds"]))

    print()
    print_report(rows)

    if args.outfile:
        import csv
        with open(args.outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("\nWrote results to {}".format(args.outfile))

    if args.time_full_sort:
        from models import PopModel

        Ntot_small = int(min(args.Ntot))
        print("\nCross-checking full sort at Ntot={:.3g} (prefilter on vs. off)...".format(Ntot_small))

        pm_off = PopModel(Ntot_small, xp.random.default_rng(args.seed), fbins=fbins,
                           Nreal=1, block_after=args.block_after, threshold_val=args.snr_thresh,
                           use_naive_prefilter=False)
        pm_on = PopModel(Ntot_small, xp.random.default_rng(args.seed), fbins=fbins,
                          Nreal=1, block_after=args.block_after, threshold_val=args.snr_thresh,
                          use_naive_prefilter=True)

        t0 = time.time()
        fs_off, fg_off, Nres_off = pm_off.run_model()
        t_off = time.time() - t0

        t0 = time.time()
        fs_on, fg_on, Nres_on = pm_on.run_model()
        t_on = time.time() - t0

        print("  without prefilter: {:.3f}s, N_res={}".format(t_off, int(Nres_off)))
        print("  with prefilter:    {:.3f}s, N_res={}".format(t_on, int(Nres_on)))
        print("  N_res match: {}".format(bool(Nres_off == Nres_on)))
        print("  foreground_psd allclose: {}".format(bool(xp.allclose(fg_off, fg_on))))
