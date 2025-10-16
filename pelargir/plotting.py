#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:15:04 2025

@author: Alexander W. Criswell

Plotting methods.
"""

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import LineCollection
import matplotlib.cm
import corner
import sys

## TODO -- fix this once we've packaged things up
prop_path = '/home/awc/Documents/LISA/projects/lisa_population_inference/pelargir-gb/pelargir/'
sys.path.insert(1, prop_path)
from utils import lisa_noise_psd, to_numpy

def savefig_png_pdf(filepath,extensions=['.png','.pdf'],**savefig_kwargs):
    """
    Utility function to save a figure with multple extensions.

    Parameters
    ----------
    ilepath : str
        '/path/to/file/save/location/filename.'
    extensions : list of str, optional
        Filetype extensions to save as, given as a list of strings. The default is ['.png','.pdf'].
    **savefig_kwargs : kwargs
        Keyword arguments for matplotlib.pyplot.savefig.
    
    Returns
    -------
    None.

    """
    
    for ext in extensions:
        ## catch filetype extensions without leading '.'
        if ext[0] != '.':
            ext = '.'+ext
        ## save
        plt.savefig(filepath+ext,**savefig_kwargs)
    
    return

def savefig_to_path(filename,saveto=None):
    """
    Utility function to save a figure of name [filename] to path [saveto] as both png and pdf.

    Parameters
    ----------
    filename : str
        Desired filename, sans extensions.
    saveto : str
        '/path/to/file/save/location/'. The default is None (save to current directory).

    Returns
    -------
    None.

    """
    
    if saveto is not None:
        fig_path_base = (saveto + '/{}'.format(filename)).replace('//','/')
    else:
        fig_path_base = filename
    savefig_png_pdf(fig_path_base, dpi=200)
    
    return
    

def plot_corners(samples,parameters=None,Nbins=20,figsize=(10,10),
                 subset=None,truths=None,priors=None,
                 save=False,saveto=None,show=True,
                 **corner_kwargs):
    """
    Creates a corner plot of 1D and 2D marginal posterior samples.

    Parameters
    ----------
    samples : array
        Samples to plot.
    parameters : list of str, optional
        List of parameter names. The default is None.
    Nbins : int, optional
        Number of bins to use in the histograms. The default is 20.
    figsize : tuple of float, optional
        Matplotlib figure size. The default is (10,10).
    subset : NotImplemented, optional
        Not yet implemented. For future use, to allow for plotting of only a subset of parameters. The default is None.
    truths : dict or list of float, optional
        True (simulated) values for each parameter. Can be provided as a list of floats (in order of samples/labels) 
        or a dict of {parameter_name : val for parameter_name in labels}.The default is None.
    priors : dict, optional
        NOT IMPLEMENTED YET Eryn pior dictionary. The default is None.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    **corner_kwargs : kwargs
        Keyword arguments to pass to corner.corner.

    Raises
    ------
    NotImplementedError
        Need to work on co-plotting of prior distributions.

    Returns
    -------
    None.

    """
    
    
    default_ckwargs = {'plot_datapoints':False,
                       'plot_density':True,
                       'density':True,
                       'fill_contours':True,
                       'smooth':0.75,
                       'show_titles':False,
                       'color':'teal',
                       }
    
    corner_kwargs = default_ckwargs | corner_kwargs
    
    plt.rcParams.update({'axes.labelsize':16})
    
    fig = plt.figure(figsize=figsize)
    corner.corner(samples, bins=Nbins, fig=fig, labels=parameters, **corner_kwargs)#, labelpad=0.1)
    
    ## add prior distributions if desired (WIP)
    if priors is not None:
        raise NotImplementedError("This still needs work.")
        ndim = samples.shape[-1]
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for ii in range(ndim):
            ax = axes[ii,ii]
            ## probably replace this with a pdf with clever scaling
            prior_samps = priors[ii]['dist'].rvs(4200)
            ax.hist(prior_samps,color='mediumorchid',bins=Nbins,alpha=0.3)
            ax.axis('auto')
            for ax2 in axes[ii:,ii]:
                ax2.sharex(ax)
            ax3 = axes[ndim-1,ii]
            ticks = [np.min(prior_samps),(np.max(prior_samps) + np.min(prior_samps))/2,np.max(prior_samps)]
            labels = ["{:0.2f}".format(tick) for tick in ticks]
            ax3.set_xticks(ticks,labels)
    
    ## add truevals
    if truths is not None:
        if type(truths) is dict:
            if parameters is None:
                raise TypeError("True values (truths) can only be provided as a dictionary if \
                                 you have also provided parameter names via the parameters argument.")
            truths = [truths[parameter_name] for parameter_name in parameters]
        corner.overplot_lines(fig, truths, ls='--', c='k', alpha=0.7)
    
    ## save
    if save:
        savefig_to_path('population_corners',saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return


def plot_current_spectra(current_state,datadict,popmodel,cmap='cool',
                         show=True,save=False,saveto=None,return_spectra=False):
    """
    Plots the foreground spectra of the current state.

    Parameters
    ----------
    current_state : array
        Current state of the sampler.
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc..
    popmodel : pelargir.models.PopModel
        The Pelargir population model object.
    cmap : str, optional
        Name of a matplotlib colormap, for use in the log likelihood colorbar. The default is 'cool'.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    return_spectra : bool, optional
        Whether to return the computed spectra and auxilliary information as a dictionary.
    Returns
    -------
    None if return_spectra==False (default)
    
    else spec_dict : dict
        Computed spectra and auxilliary information as a dictionary.
        

    """
    
    ## get the current state
    current_state = current_state.squeeze()
    nwalkers = current_state.shape[0]
    
    ## run the popmodel for each
    current_likes = []
    current_astro = []
    for i in range(current_state.shape[0]):
        like_i, astro_i = popmodel.fg_N_ln_prob(current_state[i,:],return_spec=True)
        current_likes.append(like_i)
        current_astro.append(astro_i)
    
    ## get data spectra
    fs = to_numpy(datadict['fs'])
    sim_noise_psd = to_numpy(lisa_noise_psd(datadict['fs']))
    sim_spec = to_numpy(datadict['fg']) + sim_noise_psd
    sigma = to_numpy(datadict['fg_sigma'])
    
    ## plot
    plt.figure(figsize=(7,4))
    spec_draws = [np.column_stack([current_astro[i][0], sim_noise_psd[1:]+current_astro[i][1]]) for i in range(nwalkers)]
    line_collection = LineCollection(spec_draws, array=current_likes, cmap=cmap,alpha=0.75,label='Current Draws')
    plt.gca().add_collection(line_collection)
    plt.colorbar(line_collection,label='Log Likelihood')
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='noise')

    plt.fill_between(fs,sim_spec-2*sigma,sim_spec+2*sigma,
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty')
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal')
    plt.legend()
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    # plt.title('2-sigma log-normal uncertainty')
    # plt.ylim(1e-40,1e-36)
    # plt.xlim(5e-4,3e-3)
    plt.tight_layout()
    
    ## save
    if save:
        savefig_to_path('current_spectra',saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    if return_spectra:
        return {'fs':fs,
                'spectra':[current_astro[i][1] for i in range(len(current_astro))],
                'N_res':[current_astro[i][2] for i in range(len(current_astro))],
                'loglike':current_likes,
                'noise':sim_noise_psd,
                'data_spec':to_numpy(datadict['fg'])}
    else:
        return

def plot_model_chains(ensemble,names=None,model_name='model_0',
                show=True,save=False,saveto=None):
    """
    Makes the chain plots (parameter values as a function of sampler iteration).

    Parameters
    ----------
    ensemble : eryn.ensemble.EnsembleSampler object
        The Eryn sampler object.
    names : list of str, optional
        Parameter names. The default is None.
    model_name : str, optional
        Name by which Eryn refers to the desired model (branch). The default is 'model_0'.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    
    Returns
    -------
    None.

    """
    
    ## get dimension info
    ndim = ensemble.ndims[model_name]
    nwalkers = ensemble.nwalkers
    
    ## plot
    fig, ax = plt.subplots(ndim, 1, sharex=True)
    fig.set_size_inches(10, 8)
    for i in range(ndim):
        for walk in range(nwalkers):
            ax[i].plot(ensemble.get_chain()[model_name][:, 0, walk, :, i], color='k', alpha=0.1)
        if names is not None:
            ax[i].set_ylabel(names[i],fontsize=12)
    ax[i].set_xlabel("Step",fontsize=12)
    
    ## save
    if save:
        savefig_to_path('parameter_chains',saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_model_loglikes(ensemble,names=None,ylim=None,
                        show=True,save=False,saveto=None):
    """
    Makes the log likelihood evolution plot (log likelihood values as a function of sampler iteration).

    Parameters
    ----------
    ensemble : eryn.ensemble.EnsembleSampler object
        The Eryn sampler object.
    names : list of str, optional
        Parameter names. The default is None.
    ylim : tuple of float, optional
        Matplotlib y axis limits, provided as a tuple. The default is None.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    
    Returns
    -------
    None.

    """
    
    ## get dim info
    nwalkers = ensemble.nwalkers
    
    ## grab log likelihood
    loglike = ensemble.get_log_like().reshape(ensemble.get_log_like().shape[0],nwalkers)
    
    ## make figure
    plt.figure(figsize=(10,3))
    for i in range(nwalkers):
        plt.plot(loglike[:,i])
    
    ## aesthetics
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Step")
    plt.ylabel("Log Likelihood")
    
    ## save
    if save:
        savefig_to_path('log_likelihoods',saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_distance_recovery(gamma_samples,prior_min=[2.5,2.5],prior_max=[5.5,5.5],
                           show=True,save=False,saveto=None):
    """
    

    Parameters
    ----------
    gamma_samples : array
        Samples of gamma a and b parameters. Must be of shape (N_samples,2).
    prior_min : list of float, optional
        Prior minimum for gamma parameters, given as [a_min,b_min]. The default is [2.5,2.5].
    prior_max : list of float, optional
        Prior maximum for gamma parameters, given as [a_max,b_max].. The default is [5.5,5.5].
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ## force 2D samples of [gamma_a,gamma_b]
    if gamma_samples.shape[1] !=2:
        raise ValueError("gamma_samples must be chains of gamma_a and gamma_b (i.e., of shape (N_samples,2)")
    
    ## make a grid to compare against
    xs = np.linspace(0.5,50,101)
    a_grid, scale_grid = np.meshgrid(np.linspace(prior_min[0],prior_max[0],40),np.linspace(prior_min[1],prior_max[1],40))
    gamma_grid = st.gamma.pdf(xs.reshape(-1,1),
                          a=a_grid.flatten().reshape(-1,1).T,
                          scale=scale_grid.flatten().reshape(-1,1).T)
    
    ## looking at the distance recovery
    plt.figure()
    lower = np.min(gamma_grid,axis=1)
    upper = np.max(gamma_grid,axis=1)
    plt.fill_between(xs,lower,upper,
                     alpha=0.1,color='teal',label='prior')
    for i in range(gamma_samples.shape[0]):
        if i == 0:
            plt.plot(xs,st.gamma.pdf(xs,a=gamma_samples[i,0],scale=gamma_samples[i,1]),
                     lw=0.1,c='slategrey',alpha=0.1,label='Samples')
        else:
            plt.plot(xs,st.gamma.pdf(xs,a=gamma_samples[i,0],scale=gamma_samples[i,1]),
                     lw=0.1,c='slategrey',alpha=0.1,label='__nolabel__')
    plt.plot(xs,st.gamma.pdf(xs,a=4,scale=4),lw=2,c='magenta',label='Simulation')
    plt.legend()
    plt.xlabel("$d_L$ [kpc]")
    plt.ylabel("$p(d_L)$")
    
    ## save
    if save:
        savefig_to_path('log_likelihoods',saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return
    
    
    
    
    
    