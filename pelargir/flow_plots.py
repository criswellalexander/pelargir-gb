"""
Plots shared by the flow toy-model samplers (scripts/run_flow_hmc.py, scripts/run_flow_eryn.py): corner
plots of the context parameters (one sampler, or two overlaid), per-chain traces, and the S_gw posterior
against the truth, data and noise.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_corner(path, theta, truth, names):
    '''

    Corner plot of the context samples with the truth marked.

    Arguments
    ------------------------------
    path : str
    	Output file.
    theta : numpy.ndarray
    	(n, 8) samples (chains flattened).
    truth : numpy.ndarray
    	(8,) true context.
    names : list of str
    	Parameter names.

    Returns
    ------------------------------
    None

    '''
    import corner
    fig = corner.corner(theta, labels=[n.replace('_', ' ') for n in names], color='mediumorchid', truths=truth,
                        truth_color='teal', quantiles=[0.05, 0.5, 0.95], show_titles=True, title_fmt='.3g', title_kwargs=dict(fontsize=9))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_trace(path, theta, ln_post, truth, names):
    '''

    Per-chain traces of the context parameters and of ln_post.

    Arguments
    ------------------------------
    path : str
    	Output file.
    theta : numpy.ndarray
    	(chains, draws, 8) samples.
    ln_post : numpy.ndarray
    	(chains, draws) log posterior of each draw.
    truth : numpy.ndarray
    	(8,) true context.
    names : list of str
    	Parameter names.

    Returns
    ------------------------------
    None

    '''
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()[:8]):
        ax.plot(theta[:, :, j].T, lw=0.6, alpha=0.8)
        ax.axhline(truth[j], color='teal', ls='--', lw=1)
        ax.set_ylabel(names[j].replace('_', ' '), fontsize=10)
    axes.ravel()[8].plot(ln_post.T, lw=0.6, alpha=0.8)
    axes.ravel()[8].set_ylabel('ln post', fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('draw')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_spectrum(path, fs, S, truth_S, data_psd, noise):
    '''

    S_gw posterior quantiles against the truth, the data and the noise, and the posterior relative
    to the truth.

    Arguments
    ------------------------------
    path : str
    	Output file.
    fs : numpy.ndarray
    	(Nf',) bin frequencies [Hz].
    S : numpy.ndarray
    	(n, Nf') S_gw samples [1/Hz].
    truth_S, data_psd, noise : numpy.ndarray
    	(Nf',) true S_gw, observed total PSD and noise PSD [1/Hz].

    Returns
    ------------------------------
    None

    '''
    q = np.percentile(S, [2.5, 16, 50, 84, 97.5], axis=0)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax.loglog(fs, noise, ls='--', color='0.5', label='Noise')
    ax.loglog(fs, data_psd, color='slategrey', lw=1, alpha=0.7, label='Data (total)')
    ax.fill_between(fs, q[0], q[4], color='C0', alpha=0.2, lw=0)
    ax.fill_between(fs, q[1], q[3], color='C0', alpha=0.4, lw=0)
    ax.loglog(fs, q[2], color='C0', label=r'Posterior $S_{\rm GW}$ (median, 68%, 95%)')
    ax.loglog(fs, truth_S, color='teal', lw=1.1, label=r'True $S_{\rm GW}$')
    ax.set_ylabel(r'PSD [1/Hz]')
    ax.legend(fontsize=9)
    r = np.log10(q/truth_S)
    ax2.fill_between(fs, r[0], r[4], color='C0', alpha=0.2, lw=0)
    ax2.fill_between(fs, r[1], r[3], color='C0', alpha=0.4, lw=0)
    ax2.plot(fs, r[2], color='C0')
    ax2.axhline(0, color='teal', lw=1)
    ax2.set_xlabel('f [Hz]')
    ax2.set_ylabel(r'$\log_{10}(S/S_{\rm true})$')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_corner_compare(path, theta_a, theta_b, labels, truth, names):
    '''

    Two samplers' context posteriors overlaid on one corner plot, with the truth marked.

    Arguments
    ------------------------------
    path : str
    	Output file.
    theta_a, theta_b : numpy.ndarray
    	(n_a, 8) and (n_b, 8) samples (chains flattened).
    labels : tuple of str
    	Legend labels of the two samplers.
    truth : numpy.ndarray
    	(8,) true context.
    names : list of str
    	Parameter names.

    Returns
    ------------------------------
    None

    '''
    import corner
    from matplotlib.lines import Line2D
    ## common ranges, so both sets of contours share the axes
    lo = np.minimum(np.percentile(theta_a, 0.5, axis=0), np.percentile(theta_b, 0.5, axis=0))
    hi = np.maximum(np.percentile(theta_a, 99.5, axis=0), np.percentile(theta_b, 99.5, axis=0))
    kw = dict(range=list(zip(lo, hi)), plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
              hist_kwargs=dict(density=True))
    fig = corner.corner(theta_a, labels=[n.replace('_', ' ') for n in names], color='C0', **kw)
    corner.corner(theta_b, fig=fig, color='C1', truths=truth, truth_color='teal', **kw)
    fig.legend(handles=[Line2D([], [], color='C0', label=labels[0]), Line2D([], [], color='C1', label=labels[1])],
               loc='upper right', fontsize=14)
    fig.savefig(path, dpi=110)
    plt.close(fig)
