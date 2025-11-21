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
# from matplotlib.ticker import AutoLocator
# from matplotlib.pyplot import cycler
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
# import matplotlib.cm
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
    filepath : str
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
        plt.savefig(filepath+ext,bbox_inches='tight',**savefig_kwargs)
    
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
                 save=False,saveto=None,savename='population_corners',show=True,
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
    savename : str, optional
        If save, override the default filename with savename.
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
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return


def plot_spectra_flexible(current_state,datadict,popmodel,eryn_supplemental=None,eryn_loglikes=None,eryn_nwalkers=None,
                         eryn_model_name='model_0',iteration=-1,
                         cmap='cool',show=True,save=False,saveto=None,savename='spectra',return_spectra=False,
                         xlim=None,ylim=None):
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
    eryn_supplemental : eryn.state.BranchSupplemental or eryn.backend.SupplementalBackend, optional
        If desired, the current Eryn Branch Supplemental object (or a SupplementalBackend), which contains the spectra and N_res computed within
        Eryn at runtime. If applicable, only the cold chain will be plotted. The default is None (generate fresh spectra
        from the current_state with new population draws). You must also provide eryn_loglikes.
    eryn_loglikes : array, optional
        Precomputed log likelihood of points in the current state. Necessary (and only used) if eryn_supplemental is provided.
    eryn_nwalkers : int, optional
        Number of walkers. Only needed for plot_spectra_from_ensemble.
    eryn_model_name : str, optional
        Name of the eryn model, for use as a key to eryn_supplemental. The default is 'model_0'.
    iteration : float, optional
        Which iteration of the chain to plot. Default is -1 (most recent). Only needed if passing full chains
        from the ensemble.
    cmap : str, optional
        Name of a matplotlib colormap, for use in the log likelihood colorbar. The default is 'cool'.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    savename : str, optional
        If save, override the default filename with savename.
    return_spectra : bool, optional
        Whether to return the computed spectra and auxilliary information as a dictionary.
    xlim, ylim : tuple, optional
        x and y axis limits. Default is None (matplotlib auto-limits).
    Returns
    -------
    None if return_spectra==False (default)
    
    else spec_dict : dict
        Computed spectra and auxilliary information as a dictionary.
        

    """
    
    ## get data spectra
    fs = to_numpy(datadict['fs'])
    sim_noise_psd = to_numpy(lisa_noise_psd(datadict['fs']))
    sim_spec = to_numpy(datadict['fg']) + sim_noise_psd
    sigma = to_numpy(datadict['fg_sigma'])
    
    if current_state is not None:
        ## get the current state
        current_state = current_state.squeeze()
        nwalkers = current_state.shape[0]
    else:
        nwalkers = eryn_nwalkers
    
    if eryn_supplemental is None:
        ## run the popmodel for each
        current_likes = []
        current_astro = []
        for i in range(current_state.shape[0]):
            like_i, astro_i = popmodel.fg_N_ln_prob(current_state[i,:],return_spec=True)
            current_likes.append(like_i)
            current_astro.append(astro_i)
        spec_draws = [np.column_stack([current_astro[i][0], sim_noise_psd[1:]+current_astro[i][1]]) for i in range(nwalkers)]
    else:
        ## get the drawn spectra
        if eryn_loglikes is None:
            raise RuntimeError("If eryn_supplemental is provided, you must also provide the current log likelihood via eryn_loglikes.")
        ## allow for a branch supplmental object to be passed sans dict or the backend to be passed directly
        if type(eryn_supplemental) is dict:
            try:
                ## SupplementalBackend.get_chain_supplemental()
                branch_spectra = eryn_supplemental[eryn_model_name]['spectra']
            except:
                ## dict-wrapped BranchSupplemental
                branch_spectra = eryn_supplemental[eryn_model_name][0]['spectra']
                ## handle initial draw, pre-sampler
                if branch_spectra.ndim == 3 and branch_spectra.shape[1]==1:
                    branch_spectra = branch_spectra[np.newaxis,np.newaxis,...]
        else:
            try:
                # raw SupplementalBackend
                branch_spectra = eryn_supplemental.get_chain_supplemental()[eryn_model_name]['spectra']
            except:
                ## raw BranchSupplemental
                branch_spectra = eryn_supplemental[0]['spectra']
                ## handle initial draw, pre-sampler
                if branch_spectra.ndim == 3 and branch_spectra.shape[1]==1:
                    branch_spectra = branch_spectra[np.newaxis,np.newaxis,...]
        
        
        ## TODO -- fix this
        ## handle different configurations of the array depending on steps, temps, etc.
        supp_ndim_eff = branch_spectra.squeeze().ndim
        if supp_ndim_eff >= 4:
            ## take cold chain
            temps_inds = 0
            nreal = branch_spectra.shape[-2]
        elif supp_ndim_eff == 3 or supp_ndim_eff == 2:
            temps_inds = ...
            nreal = branch_spectra.shape[-2]
        else:
            raise IndexError("Provided branch supplemental is of effective (squeezed) dimension {}; this is unexpected.\
                              Branch supplemental should have shape (ntemps,nwalkers,nfreqs) or (nwalkers,nfreqs).".format(supp_ndim_eff))
        
        spec_draws = []
        for i in range(nwalkers):
            for j in range(nreal):
                spec_draws.append(np.column_stack([fs, sim_noise_psd+branch_spectra[iteration,i,temps_inds,:,j,:].squeeze()]))

        if eryn_loglikes.ndim > 2:
            current_likes = eryn_loglikes[iteration,temps_inds,:].squeeze().repeat(nreal)
        else:
            current_likes = eryn_loglikes[iteration,:].squeeze().repeat(nreal)
    
    ## plot
    plt.figure(figsize=(7,4))
    
    line_collection = LineCollection(spec_draws, array=current_likes, cmap=cmap,alpha=0.75,label='Current Draws')
    plt.gca().add_collection(line_collection)
    plt.colorbar(line_collection,label='Log Likelihood')
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='Instrumental Noise')

    plt.fill_between(fs,10**(np.log10(sim_spec)-2*sigma),10**(np.log10(sim_spec)+2*sigma),
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty')
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal')
    plt.legend()
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    if iteration == -1:
        iteration = branch_spectra.shape[0]
    plt.title("Spectrum Draws for Iteration {}".format(iteration))
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    # plt.title('2-sigma log-normal uncertainty')
    # plt.ylim(1e-40,1e-36)
    # plt.xlim(5e-4,3e-3)
    # plt.tight_layout()
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
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

def plot_spectra(ensemble,datadict,chain_kwargs={},**kwargs):
    """
    Wrapper function for plot_current_spectra which only needs the Eryn ensemble object and
    the data dictionary. Any additional kwargs are passed to plot_current_spectra.

    Parameters
    ----------
    ensemble : eryn.ensemble.EnsembleSampler
        The instantiated Eryn ensemble. Must have branch supplemental activated and use the
        SupplementalBackend backend.
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc.. At minimum, must
        have keys 'fs', 'fg', and 'fg_sigma' for the data frequencies, foreground PSD, and 
        PSD uncertainty, respectively.
    chain_kwargs : dict
        Keyword arguments to pass to ensemble.get_chain_supplemental and get_log_like.
        (See those functions' documentation.)
    **kwargs : kwargs
        Other keyword arguments to pass to plot_current_spectra().

    Returns
    -------
    out : dict, optional
        If return_spectra=True is passed as a kwarg, this will be a dictionary with the plotted
        arrays. Otherwise returns None.

    """
    
    out = plot_spectra_flexible(None,datadict,None,
                                eryn_supplemental=ensemble.get_chain_supplemental(**chain_kwargs),
                                eryn_loglikes=ensemble.get_log_like(**chain_kwargs),
                                eryn_nwalkers=ensemble.nwalkers,
                                **kwargs)
    return out

def plot_spectra_chains(ensemble,datadict,eryn_model_name='model_0',
                        show=True,save=False,saveto=None,savename='spectral_chains',
                         xlim=None,ylim=None,**kwargs):
    """
    Plots the foreground spectra of the current state.

    Parameters
    ----------
    ensemble : eryn.ensemble.EnsembleSampler
        The instantiated Eryn ensemble. Must have branch supplemental activated and use the
        SupplementalBackend backend.
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc..
    eryn_model_name : str, optional
        Name of the eryn model, for use as a key to eryn_supplemental. The default is 'model_0'.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    savename : str, optional
        If save, override the default filename with savename.
    xlim, ylim : tuple, optional
        x and y axis limits. Default is None (matplotlib auto-limits).
    **kwargs : keyword arguments
        Keyword arguments to pass to ensemble.get_chain_supplemental()
    
    Returns
    -------
    None

    """
    
    ## get data spectra
    fs = to_numpy(datadict['fs'])
    sim_noise_psd = to_numpy(lisa_noise_psd(datadict['fs']))
    sim_spec = to_numpy(datadict['fg']) + sim_noise_psd
    sigma = to_numpy(datadict['fg_sigma'])
        
    Nf = len(fs)
    spec_chain = ensemble.get_chain_supplemental(**kwargs)['model_0']['spectra']
    # import pdb; pdb.set_trace()
    ## plot
    plt.figure(figsize=(7,4))
    
    spec_chain_color = 'mediumorchid'
    spec_chain_lw = 1
    spec_chain_alpha = 0.01
    
    ## set dims for iteration and plotting
    ## because reshape breaks things for some reason
    ## this will break for nwalkers,ntemps>1 but I'll fix it later
    Ni, Nj = np.argwhere(np.array(spec_chain.squeeze().shape) != Nf).flatten()
    for i in range(spec_chain.squeeze().shape[Ni]):
        for j in range(spec_chain.squeeze().shape[Nj]):
            plt.loglog(datadict['fs'].get(),sim_noise_psd+spec_chain.squeeze()[i,:,j],
                       alpha=spec_chain_alpha,c=spec_chain_color,
                       linewidth=spec_chain_lw,label='__nolabel__')
    # plt.loglog(fs,sim_noise_psd[:,None]+spec_chain,alpha=spec_chain_alpha,c=spec_chain_color,linewidth=spec_chain_lw,label='__nolabel__')
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='Instrumenal Noise')

    plt.fill_between(fs,10**(np.log10(sim_spec)-2*sigma),10**(np.log10(sim_spec)+2*sigma),
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty',zorder=-10)
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal')
    
    # adding custom legend entry
    handles, labels = plt.gca().get_legend_handles_labels()
    spec_line_handle = Line2D([0], [0], label='Spectral Posterior Draws', color=spec_chain_color, alpha=1, linewidth=spec_chain_lw)
    handles.extend([spec_line_handle])
    
    plt.legend(handles=handles)
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    plt.title("Foreground Spectrum Posterior")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    # plt.title('2-sigma log-normal uncertainty')
    # plt.ylim(1e-40,1e-36)
    # plt.xlim(5e-4,3e-3)
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return


def plot_Nres_hist(ensemble,datadict,eryn_model_name='model_0',showtrue=True,
                   xlim=None,bins=None,show=True,save=False,saveto=None,savename='Nres_histogram',
                   **kwargs):
    """
    
    Parameters
    ----------
    ensemble : eryn.ensemble.EnsembleSampler
        The instantiated Eryn ensemble. Must have branch supplemental activated and use the
        SupplementalBackend backend.
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc.. At minimum, must
        have keys 'fs', 'fg', and 'fg_sigma' for the data frequencies, foreground PSD, and 
        PSD uncertainty, respectively.
    eryn_model_name : str, optional
        Name of the eryn model, for use as a key to eryn_supplemental. The default is 'model_0'.
    show : bool, optional
        Whether to show the plot at runtime. The default is True.
    save : bool, optional
        Whether to save the created figures to disk. The default is False.
    saveto : str, optional
        If save, the desired output directory. The default is None (saves in current directory).
    savename : str, optional
        If save, override the default filename with savename.
    bins : array, optional
        Histogram bins. The default is 'auto' (plt.hist() auto bins).
    xlim : tuple, optional
        x axis limits. Default is None (matplotlib auto-limits).
    
    **kwargs : kwargs
        Keyword arguments to pass to ensemble.get_chain_supplemental().

    Returns
    -------
    None.

    """
    
    plt.figure()
    Nres_samps = ensemble.get_chain_supplemental(**kwargs)['model_0']['Nres'].flatten()
    plt.hist(Nres_samps,alpha=0.8,bins=bins,label='Samples')
    if showtrue:
        plt.axvline(to_numpy(datadict['Nres']),ls='--',color='cyan',label='Simulated')
    if xlim is not None:
        plt.xlim(*xlim)
    plt.title(r"Posterior Distribution for Number of Resolved GBs ($N_{\rm res}$)")
    plt.legend()
    plt.xlabel(r"$N_{\rm res}$")
    plt.ylabel("Count")
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_model_chains(ensemble,names=None,model_name='model_0',
                show=True,save=False,saveto=None,savename='chains',**kwargs):
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
    savename : str, optional
        If save, override the default filename with savename.
    **kwargs : kwargs
        Keyword arguments to pass to ensemble.get_chain().
    
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
            ax[i].plot(ensemble.get_chain(**kwargs)[model_name][..., walk, :, i], color='k', alpha=0.1)
        if names is not None:
            ax[i].set_ylabel(names[i])
    ax[i].set_xlabel("Step")
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_model_loglikes(ensemble,names=None,ylim=None,
                        show=True,save=False,saveto=None,savename='loglikes',
                        **kwargs):
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
    savename : str, optional
        If save, override the default filename with savename.
    
    Returns
    -------
    None.

    """
    
    ## get dim info
    nwalkers = ensemble.nwalkers
    
    ## grab log likelihood
    loglike = ensemble.get_log_like(**kwargs)
    
    ## make figure
    plt.figure(figsize=(10,4))
    for i in range(nwalkers):
        plt.plot(loglike[:,i])
    
    ## aesthetics
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Step")
    plt.ylabel("Log Likelihood")
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_distance_recovery(gamma_samples,prior_min=[2.5,2.5],prior_max=[5.5,5.5],
                           show=True,save=False,saveto=None,savename='dist_recovery'):
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
    savename : str, optional
        If save, override the default filename with savename.

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
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return
    
    
    
    
    
    