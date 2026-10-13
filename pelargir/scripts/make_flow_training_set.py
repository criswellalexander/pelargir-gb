"""
Generates a flow_data.TrainingSet from the JAX forward model: n_draws hyperprior draws (flow_data.HYPERPRIOR,
including rho_thresh and lambda_tot) x n_real realizations each, with per-bin N_res and S_gw on the
model grid below fmax. Runs on the jax backend; batches are sized to the GPU's free memory unless
--max_binaries_per_batch is given. Set JAX_COMPILATION_CACHE_DIR to reuse compiled kernels across runs.

Usage (with pelargir installed; or run this file with python)
-----
    pelargir-make-flow-set out.npz [--n_draws 3200] [--n_real 5] [--fmin 1e-4] [--fmax 1e-3]
                                     [--fbin 2e-5] [--seed 1] [--max_binaries_per_batch N] [--chunk_dir chunks/]
                                     [--n_devices N]
"""
import os
import logging
import argparse


def main(args):
    ## JAX's default allocator fragments across the many batch shapes and runs out of memory on 8 GB
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    ## INFO messages say whether the run is new or resuming
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    logging.getLogger('pelargir').setLevel(logging.INFO)
    from pelargir import backend
    backend.set_backend('jax')
    from pelargir import flow_data

    fbins = flow_data.model_fbins(args.fmin, args.fmax, args.fbin)
    print("{} modelled bins on [{:.3g}, {:.3g}] Hz; {} draws x {} realizations".format(
        fbins.size - 1, fbins[1], fbins[-1], args.n_draws, args.n_real), flush=True)
    ts = flow_data.draw_training_set(args.n_draws, args.n_real, fbins, args.seed, prefilter_snr=args.prefilter_snr,
                                     max_binaries_per_batch=args.max_binaries_per_batch, chunk_dir=args.chunk_dir,
                                     devices=args.n_devices)
    ## written atomically: a job killed mid-save must not leave a file the next stage would trust
    tmp = args.outfile + '.tmp.npz'
    ts.save(tmp)
    os.replace(tmp, args.outfile)
    n_sim = ts.meta['simulated_galaxies']
    print("saved {} rows to {} ({:.0f} s; {} galaxies simulated this run, {} done in earlier runs, {:.1f} ms each; "
          "{} compiles, {} overflow reruns)".format(ts.context.shape[0], args.outfile, ts.meta['seconds'], n_sim,
                                                    ts.meta['resumed_galaxies'], 1e3*ts.meta['seconds']/max(n_sim, 1),
                                                    ts.meta['compiles'], ts.meta['overflow_reruns']))


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outfile', type=str)
    parser.add_argument('--n_draws', type=int, default=3200)
    parser.add_argument('--n_real', type=int, default=5)
    parser.add_argument('--fmin', type=float, default=1e-4)
    parser.add_argument('--fmax', type=float, default=1e-3)
    parser.add_argument('--fbin', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--prefilter_snr', type=float, default=1.0)
    parser.add_argument('--chunk_dir', type=str, default=None,
                        help="Save the plan and each batch here as it completes and reuse them (resumable runs).")
    parser.add_argument('--max_binaries_per_batch', type=float, default=None,
                        help="Drawn binaries (galaxies x chunks x DRAW_CHUNK) per jitted batch. Default: fill 80%% of "
                             "the GPU's free memory (galaxies are drawn in chunks, so memory doesn't grow with N).")
    parser.add_argument('--n_devices', type=int, default=None,
                        help="GPUs to spread batches over (one worker thread each). Default: every visible GPU.")
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
