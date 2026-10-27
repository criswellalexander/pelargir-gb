#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:15:04 2025

@author: Alexander W. Criswell

Plotting methods, including those shared by the flow toy-model samplers (plot_corner, plot_trace,
plot_spectrum, plot_corner_compare).
"""

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import corner

from .utils import lisa_noise_psd, to_numpy

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
                       'bins':Nbins
                       }
    
    corner_kwargs = default_ckwargs | corner_kwargs
    
    plt.rcParams.update({'axes.labelsize':16})
    
    fig = plt.figure(figsize=figsize)
    corner.corner(samples, fig=fig, labels=parameters, **corner_kwargs)#, labelpad=0.1)
    
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
    popmodel : pelargir.models.population.PopModel
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
        spec_draws = [np.column_stack([current_astro[i][0], sim_noise_psd+current_astro[i][1]]) for i in range(nwalkers)]
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
    plt.legend(loc='upper right')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    if iteration == -1:
        iteration = branch_spectra.shape[0]
    plt.title("Spectrum Draws for Iteration {}".format(iteration))
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        ## dynamic ylims
        ylim_0 = ylim[0]
        ylim_1 = ylim[1]
        if ylim_0 is None:
            ylim_0 = 0.99*sim_noise_psd.min()
        if ylim_1 is None:
            ylim_1 = 10*sim_spec.max()
        ylim = (ylim_0,ylim_1)
        plt.ylim(*ylim)
    
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
                         xlim=None,ylim=None,rasterize_chains=True,**kwargs):
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
    rasterize_chains : bool, optional
        Whether to rasterize the individual spectral chains. Default is True.
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
    
    ## set dims for iteration and plotting
    ## because reshape breaks things for some reason
    ## formatting
    spec_chain_color = 'mediumorchid'
    spec_chain_lw = 1
    spec_chain_alpha = 0.01
    
    ## this will only plot the first leaf for nleaves>1 but I'll fix it later
    Nidx = np.argwhere(np.array(spec_chain.shape) != Nf).flatten()
    ## steps
    for i in range(spec_chain.shape[Nidx[0]]):
        ## walkers
        for j in range(spec_chain.shape[Nidx[1]]):
            ## temps
            for k in range(spec_chain.shape[Nidx[2]]):
                ## realizations
                for l in range(spec_chain.shape[Nidx[3]]):
                    plt.loglog(to_numpy(datadict['fs']),sim_noise_psd+spec_chain[i,j,k,:,l,0],
                               alpha=spec_chain_alpha,c=spec_chain_color,
                               linewidth=spec_chain_lw,label='__nolabel__')
    
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='Instrumental Noise')

    plt.fill_between(fs,10**(np.log10(sim_spec)-2*sigma),10**(np.log10(sim_spec)+2*sigma),
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty',zorder=-10)
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal',alpha=0.75)
    
    ## set rasterization
    if rasterize_chains:
        plt.gca().set_rasterization_zorder(-8)
    
    # adding custom legend entry
    handles, labels = plt.gca().get_legend_handles_labels()
    spec_line_handle = Line2D([0], [0], label='Spectral Posterior Draws', color=spec_chain_color, alpha=1, linewidth=spec_chain_lw)
    handles.extend([spec_line_handle])
    
    plt.legend(handles=handles,loc='upper right')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    plt.title("Foreground Spectrum Posterior")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        ## dynamic ylims
        ylim_0 = ylim[0]
        ylim_1 = ylim[1]
        if ylim_0 is None:
            ylim_0 = 0.99*sim_noise_psd.min()
        if ylim_1 is None:
            ylim_1 = 10*sim_spec.max()
        ylim = (ylim_0,ylim_1)
        plt.ylim(*ylim)
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_data_spectrum(datadict,
                        show=True,save=False,saveto=None,savename='spectral_data',
                         xlim=None,ylim=None):
    """
    Plots the simulated foreground spectral data.

    Parameters
    ----------
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc..
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
    
    Returns
    -------
    None

    """
    
    ## get data spectra
    fs = to_numpy(datadict['fs'])
    sim_noise_psd = to_numpy(lisa_noise_psd(datadict['fs']))
    sim_spec = to_numpy(datadict['fg']) + sim_noise_psd
    sigma = to_numpy(datadict['fg_sigma'])
    Nres = to_numpy(datadict['Nres'])
        
    Nf = len(fs)
    ## plot
    plt.figure(figsize=(7,4))
    
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='Instrumental Noise')

    plt.fill_between(fs,10**(np.log10(sim_spec)-2*sigma),10**(np.log10(sim_spec)+2*sigma),
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty',zorder=-10)
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal',alpha=0.75)
    
    # adding custom legend entry
    handles, labels = plt.gca().get_legend_handles_labels()
    
    plt.legend(handles=handles,loc='upper right')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    plt.title(r"Simulated Foreground Spectrum ($N_{\rm res}=$"+"{})".format(Nres))
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        ## dynamic ylims
        ylim_0 = ylim[0]
        ylim_1 = ylim[1]
        if ylim_0 is None:
            ylim_0 = 0.99*sim_noise_psd.min()
        if ylim_1 is None:
            ylim_1 = 10*sim_spec.max()
        ylim = (ylim_0,ylim_1)
        plt.ylim(*ylim)
    
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return

def plot_sanity_check(datadict,modelspec,modelspec2,
                        show=True,save=False,saveto=None,savename='consistency_check',
                         xlim=None,ylim=None):
    """
    Plots the simulated foreground spectral data.

    Parameters
    ----------
    datadict : dict
        The data dictionary containing the simulated spectrum, noise, etc..
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
    
    Returns
    -------
    None

    """
    
    ## get data spectra
    fs = to_numpy(datadict['fs'])
    sim_noise_psd = to_numpy(lisa_noise_psd(datadict['fs']))
    sim_spec = to_numpy(datadict['fg']) + sim_noise_psd
    sigma = to_numpy(datadict['fg_sigma'])
    Nres = to_numpy(datadict['Nres'])
        
    Nf = len(fs)
    ## plot
    plt.figure(figsize=(7,4))
    
    plt.loglog(fs,sim_noise_psd,c='slategrey',ls='--',label='Instrumental Noise')

    plt.fill_between(fs,10**(np.log10(sim_spec)-2*sigma),10**(np.log10(sim_spec)+2*sigma),
                     color='turquoise',alpha=0.5,label=r'PSD 2$\sigma$ Uncertainty',zorder=-10)
    plt.loglog(fs,sim_spec,label='Total Simulated PSD',c='teal',alpha=0.75)
    if modelspec.ndim == 3:
        mspec = to_numpy(modelspec)[:,0,0]
    else:
        mspec = modelspec
    if modelspec2.ndim == 3:
        mspec2 = to_numpy(modelspec2)[:,0,0]
    else:
        mspec2 = modelspec2
    plt.loglog(fs,to_numpy(mspec)+sim_noise_psd,label='Model spectrum (manual)',c='mediumorchid',alpha=0.75)
    plt.loglog(fs,to_numpy(mspec2)+sim_noise_psd,label='Model spectrum (run_model)',c='navy',alpha=0.75)
    
    # adding custom legend entry
    handles, labels = plt.gca().get_legend_handles_labels()
    
    plt.legend(handles=handles,loc='upper right')
    plt.xlabel('f [Hz]')
    plt.ylabel('PSD [Hz^-1]')
    plt.title(r"Simulated Foreground Spectrum ($N_{\rm res}=$"+"{})".format(Nres))
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        ## dynamic ylims
        ylim_0 = ylim[0]
        ylim_1 = ylim[1]
        if ylim_0 is None:
            ylim_0 = 0.99*sim_noise_psd.min()
        if ylim_1 is None:
            ylim_1 = 10*sim_spec.max()
        ylim = (ylim_0,ylim_1)
        plt.ylim(*ylim)
    
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

def plot_astro_dists(ensemble,datadict,gbprior_obj,plot_true=False,model_name='model_0',
                     units=[r'$M_{\odot}$',r'$M_{\odot}$',r'kpc',r'AU'],
                        show=True,save=False,saveto=None,savename='astro_distributions',**kwargs):
    
    ## get chains
    ## get dimension info
    ndim = ensemble.ndims[model_name]
    chain = ensemble.get_chain(**kwargs)[model_name].reshape(-1,ndim)
    
    names = list(gbprior_obj.prior_dict.keys())
    
    if plot_true:
        gbkey = 'gb_thetas_true'
    else:
        gbkey = 'gb_thetas'
    
    ## if we have access, also plot the full (resolved + unresolved) distributions
    if 'gb_thetas_all' in datadict.keys():
        plot_all = True
    else:
        plot_all = False
    
    ## set max distance
    dlmax = 25.0
    
    fig, axes = plt.subplots(nrows=1,ncols=len(names),figsize=(15,3))
    for i, ax in enumerate(axes):
        if plot_all:
            samps_all_i = datadict['gb_thetas_all'][:,i]
            ## cut to max plot dist to ensure same binning
            ## since the unresolved dist in distance goes out much farther.
            if names[i] == 'd_L':
                samps_all_i = samps_all_i[samps_all_i<=dlmax]
            ax.hist(samps_all_i,density=True,bins=30,color='k',histtype='step',label='All')
        samps_i = to_numpy(datadict[gbkey][:,i])
        ## cut to max plot dist to ensure same binning
        if names[i] == 'd_L':
            samps_i = samps_i[samps_i<=dlmax]
        ax.hist(samps_i,density=True,bins=30,alpha=0.5,color='teal',label='Resolved')
        if plot_all:
            low = samps_all_i.min()
            high = samps_all_i.max()
        else:
            low = samps_i.min()
            high = samps_i.max()
        xvals = np.linspace(low,high,200)
        for jj in range(chain.shape[0]):
            gbprior_obj.condition(chain[jj,:])
            ax.plot(xvals,np.exp(to_numpy(gbprior_obj.conditional_dict[names[i]].logpdf(xvals)).squeeze()),
                    alpha=0.01,color='mediumorchid',lw=1,label='__nolabel__')
        ax.set_xlabel(r'${}$'.format(names[i])+' [{}]'.format(units[i]))
        ax.set_yticks([])
        if names[i] == 'd_L':
            ax.set_xlim(0,25)
        
    ## only make legend for rightmost plot
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='mediumorchid',lw=2,label='Posterior Conditionals'))
    ax.legend(handles=handles,loc='upper left')
    ## save
    if save:
        savefig_to_path(savename,saveto=saveto)
    
    if show:
        plt.show()
    
    plt.close()
    
    return


## ---------------------------------------------------------------------------------------------------
## Flow toy-model sampler plots (scripts/run_flow_hmc.py, scripts/run_flow_eryn.py): corner plots of the
## context parameters (one sampler, or two overlaid), per-chain traces, and the S_gw posterior against the
## truth, data and noise.
## ---------------------------------------------------------------------------------------------------

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
