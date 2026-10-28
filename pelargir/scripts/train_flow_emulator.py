"""
Trains a per-band flow emulator on a saved TrainingSet (one flow per band) with either flow base,
zuko (flows/zuko_base.py) or flax (flows/flax_base.py), and saves it with its per-band loss curves. Each band keeps
its best-validation weights (validation holds out whole hyperprior draws); the last epoch's are
saved beside them (emulator_final.*).

Usage (with pelargir installed; or run this file with python)
-----
    pelargir-train-flows training.npz outdir [-f zuko|flax] [--bins_per_band 5 | --edges f0 f1 ...]
                                  [--n_epochs 500] [--batch_size 4096] [--lr 1e-4] [--device cuda] [--dtype float64]
                                  [--flow_layers 8] [--hidden_size 512] [--mlp_layers 2]
"""
import os
import json
import logging
import argparse

from pelargir.log_utils import add_verbosity_args, setup_logging, timed

logger = logging.getLogger('pelargir.scripts.train_flow_emulator')


def main(args):
    '''

    Train, save and plot the emulator described by the command-line arguments.

    Arguments
    ------------------------------
    args : argparse.Namespace
    	Parsed arguments (see cli).

    Returns
    ------------------------------
    None

    '''
    setup_logging(args.verbose, args.quiet)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pelargir.flows import data as flow_data

    with timed(logger, f"loading the training set {args.training_set}"):
        ts = flow_data.TrainingSet.load(args.training_set)
    bands = flow_data.make_bands(ts.fs, bins_per_band=args.bins_per_band, edges=args.edges)
    logger.info("%d rows; %d bands of %s bins; flow base %s; %d epochs, batch %d, lr %g", ts.context.shape[0],
                len(bands), [b.nf for b in bands], args.flow_base, args.n_epochs, args.batch_size, args.lr)
    train = dict(n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac, seed=args.seed)
    ## architecture options left unset keep the flow base's defaults
    layers_key = 'transforms' if args.flow_base == 'zuko' else 'flow_num_layers'
    arch = {k: v for k, v in ((layers_key, args.flow_layers), ('hidden_size', args.hidden_size),
                              ('mlp_num_layers', args.mlp_layers)) if v is not None}
    with timed(logger, f"training {len(bands)} band flows"):
        if args.flow_base == 'zuko':
            from pelargir.flows import zuko_base as flows
            em, histories = flows.train_emulator(ts, bands=bands, device=args.device, flow_kwargs=arch, **train)
        else:
            from pelargir.flows import flax_base as flax_flows
            em, histories = flax_flows.train_emulator(ts, bands=bands, flow_kwargs=dict(arch, dtype=args.dtype),
                                                      **train)
    em.save(args.outdir)
    logger.info("saved the emulator and losses.json to %s", args.outdir)
    with open(os.path.join(args.outdir, 'losses.json'), 'w') as fh:
        json.dump(histories, fh)

    fig, axes = plt.subplots(1, len(bands), figsize=(2.2*len(bands), 2.4), sharey=False)
    for ax, b, h in zip(np.atleast_1d(axes), bands, histories):
        ax.plot(h['train'], label='train')
        ax.plot(h['val'], label='val')
        ## the epoch whose weights the emulator keeps
        ax.axvline(h['best_epoch'], color='goldenrod', ls=':', lw=1, label='best')
        ax.set_title("{:.2f}-{:.2f} mHz".format(1e3*b.fs[0], 1e3*b.fs[-1]), fontsize=8)
        ax.set_xlabel('epoch')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'losses.png'), dpi=120)
    for b, h in zip(bands, histories):
        print("band {}: best val {:.4f} at epoch {} of {}".format(b.index, h['best_val'], h['best_epoch'],
                                                                 len(h['val'])))
    print("saved emulator to", args.outdir)


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
    parser.add_argument('training_set', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('-f', '--flow-base', dest='flow_base', choices=['zuko', 'flax'], default='zuko',
                        help="Flow implementation: zuko (torch) or flax (JAX). Default zuko.")
    parser.add_argument('--bins_per_band', type=int, default=5)
    parser.add_argument('--edges', type=float, nargs='+', default=None, help="Band edges in Hz (overrides bins_per_band).")
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_frac', type=float, default=0.1, help="Fraction of hyperprior draws held out.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--flow_layers', type=int, default=None,
                        help="Flow layers (zuko transforms, flax coupling layers). Default: the base's (8).")
    parser.add_argument('--hidden_size', type=int, default=None,
                        help="Width of each conditioner hidden layer. Default: the base's (512).")
    parser.add_argument('--mlp_layers', type=int, default=None,
                        help="Hidden layers per conditioner. Default: the base's (2).")
    parser.add_argument('--device', type=str, default='cuda', help="torch device (zuko only; flax uses JAX's device).")
    parser.add_argument('--dtype', choices=['float64', 'float32'], default='float64', help="flax only. Default float64.")
    add_verbosity_args(parser)
    main(parser.parse_args(argv))


if __name__ == '__main__':
    cli()
