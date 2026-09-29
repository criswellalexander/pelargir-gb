"""
Trains a flows.BandedFlowEmulator on a saved TrainingSet (one flow per band) and saves it with its
per-band loss curves.

Usage
-----
    python train_flow_emulator.py training.npz outdir [--bins_per_band 5 | --edges f0 f1 ...]
                                  [--n_epochs 8] [--batch_size 64] [--lr 1e-3] [--device cuda]
"""
import os
import sys
import json
import argparse


def main(args):
    sys.path.insert(1, args.pelargirpath)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import flows

    ts = flows.TrainingSet.load(args.training_set)
    bands = flows.make_bands(ts.fs, bins_per_band=args.bins_per_band, edges=args.edges)
    print("{} rows; {} bands of {} bins".format(ts.context.shape[0], len(bands), [b.nf for b in bands]), flush=True)
    em, histories = flows.train_emulator(ts, bands=bands, device=args.device, n_epochs=args.n_epochs,
                                         batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac, seed=args.seed)
    em.save(args.outdir)
    with open(os.path.join(args.outdir, 'losses.json'), 'w') as fh:
        json.dump(histories, fh)

    fig, axes = plt.subplots(1, len(bands), figsize=(2.2*len(bands), 2.4), sharey=False)
    for ax, b, h in zip(np.atleast_1d(axes), bands, histories):
        ax.plot(h['train'], label='train')
        ax.plot(h['val'], label='val')
        ax.set_title("{:.2f}-{:.2f} mHz".format(1e3*b.fs[0], 1e3*b.fs[-1]), fontsize=8)
        ax.set_xlabel('epoch')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'losses.png'), dpi=120)
    print("saved emulator to", args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('training_set', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--pelargirpath', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    parser.add_argument('--bins_per_band', type=int, default=5)
    parser.add_argument('--edges', type=float, nargs='+', default=None, help="Band edges in Hz (overrides bins_per_band).")
    parser.add_argument('--n_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    main(parser.parse_args())
