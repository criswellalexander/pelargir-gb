"""
Simulates the flow toy-model dataset (flow_inference.ToyData): one galaxy from the JAX forward model at
the truth, its unperturbed total PSD as the data (as run_pelargir), and its resolved binaries as the
fixed resolved-GB state. The defaults are the stage-2 truth: Table 1's Lambda, rho_thresh = 7,
lambda_tot = 1e7, a natural-log data width of 0.1, on the flow's default grid.

Usage
-----
    python -m pelargir.scripts.make_flow_toy_data out.npz [--seed 150914] [--rho 7] [--lambda_tot 1e7]
                                                  [--sigma 0.1] [--fmin 1e-4] [--fmax 1e-3] [--fbin 2e-5]
"""
import os
import argparse


def main(args):
    '''

    Simulate the dataset and save it.

    Arguments
    ------------------------------
    args : argparse.Namespace
    	Parsed arguments (see cli).

    Returns
    ------------------------------
    None

    '''
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    from pelargir import backend
    backend.set_backend('jax')
    from pelargir import flow_data as fd
    from pelargir import flow_inference as fi

    fbins = fd.model_fbins(args.fmin, args.fmax, args.fbin)
    data = fi.simulate_toy_data(fi.FIDUCIAL, args.rho, args.lambda_tot, fbins, args.seed, args.sigma)
    data.save(args.outfile)
    print("saved {}: N_tot = {}, {} resolved binaries in {} modelled bins".format(
        args.outfile, data.meta['N_tot'], data.theta_res.shape[0], data.fs.size))


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
    parser.add_argument('outfile', type=str)
    parser.add_argument('--seed', type=int, default=150914)
    parser.add_argument('--rho', type=float, default=7.0, help="True rho_thresh.")
    parser.add_argument('--lambda_tot', type=float, default=1e7, help="True rate of N_tot ~ Poisson.")
    parser.add_argument('--sigma', type=float, default=0.1, help="Natural-log width of the data likelihood.")
    parser.add_argument('--fmin', type=float, default=1e-4)
    parser.add_argument('--fmax', type=float, default=1e-3)
    parser.add_argument('--fbin', type=float, default=2e-5)
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
