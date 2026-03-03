#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:35:13 2025

@author: Alexander W. Criswell

Various utility functions
"""
import os

try:
    if ('PELARGIR_GPU' in os.environ.keys()) and int(os.environ['PELARGIR_GPU']):
        import cupy as xp
        ## check for available devices
        if xp.cuda.is_available():
            print("GPU requested and available; running Pelargir population inference on GPU.")
            os.environ['SCIPY_ARRAY_API'] = '1'
        else:
            print("GPU requested but no device is available. Defaulting to CPU.")
            import numpy as xp
    else:
        print("Running Pelargir population inference on CPU.")
        import numpy as xp
except:
    print("An error occurred in initializing GPU functionality. Defaulting to CPU.")
    import numpy as xp

import numpy as np
from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import distributions as st


msun_kg_conv = xp.array((1*u.Msun).to(u.kg).value) ## to kg
kpc_m_conv = xp.array((1*u.kpc).to(u.m).value) ## to m
au_m_conv = xp.array((1*u.AU).to(u.m).value) ## to m
G = 6.6743e-11 ## m^3 kg^-1 s^-2
c = 2.99792458e8 ## m/s

def get_mc(m_1,m_2):
    return (m_1*m_2)**(3/5) / (m_1+m_2)**(1/5)

def get_amp_freq(theta):
    """
    Utility function to map from draws on m1, m2, d_L, and a
    to GW amplitudes and frequencies

    Parameters
    ----------
    theta : array
        Array of parameter values. Leading axis should be of shape Npar.

    Returns
    -------
    amp : array
        GW strain amplitudes, assuming monochromatic binaries.
    fgw : array
        GW frequencies in Hz.

    """
    
    m_1 = theta[0,...]*msun_kg_conv ## to kg
    m_2 = theta[1,...]*msun_kg_conv ## to kg
    d_L = theta[2,...]*kpc_m_conv ## to m
    a = theta[3,...]*au_m_conv ## to m
    amp = (8/xp.sqrt(5)) * (G**2/c**4) * (m_1*m_2)/(d_L*a)
    fgw = 1/xp.pi * xp.sqrt(G*(m_1+m_2)/a**3)
    return amp, fgw

def apply_theta_lims(scattered_theta,theta_lims='default'):
    ## simulate prior bounding by wrapping anything outside the bounds to be within them
    if theta_lims == 'default':
        ## set minimum allowed distance in kpc
        d_min = 1e-3 ## no GBs closer than the closest known star
        d_max = 100 ## reasonably far past the edge of the Galaxy
        a_min = 1e-4 ## no binaries with a semimajor axis comparable to their radius
        a_max = 1e-2 ## no binaries outside of LISA's frequency range
        m_min = 0.17 ## lowest-mass observed white dwarf
        m_max = 1.44 ## no WDs with mass above the Chandrasekar limit
        theta_lims = xp.array([[m_min,m_max],[m_min,m_max],[d_min,d_max],[a_min,a_max]])

    bounded_theta = scattered_theta
    for ii in range(scattered_theta.shape[-1]):
        lower_filt_ii = bounded_theta[:,ii] <= theta_lims[ii,0]
        upper_filt_ii = bounded_theta[:,ii] >= theta_lims[ii,1]
        bounded_theta[lower_filt_ii,ii] = 1.0000001*theta_lims[ii,0]
        bounded_theta[upper_filt_ii,ii] = 0.9999999*theta_lims[ii,1]
    
    return bounded_theta

def scatter_thetas(rng,theta_true,err=xp.array([0.05,0.05,0.1,0.001]),
                   log_args_idx=[-1],bound=True,**kwargs):
    """
    Manually introduce uncertainty into a true parameter vector, assuming 1D marginal Gaussian likelihoods. 

    Parameters
    ----------
    rng : Generator
        RNG. Numpy/cupy Generator object.
    theta_true : array
        Initial true values of parameter vector.
    err : array, optional
        Standard deviations of the 1D Gaussian distributions by which to scatter theta_true. 
        The default is xp.array([0.05,0.05,0.1,0.001]) for [m_1,m_2,d_L,log10(a/1AU)] in [Msun,Msun,kpc,AU].
    log_args_idx : list, optional
        Indices of parameters to scatter in log10 space. The default is [-1] (a only).
    bound : bool, optional
        Whether to assert prior bounds. The default is True.
    kwargs : dict, optional
        Keyword arguments to pass to apply_theta_lims().
    
    Returns
    -------
    scattered_theta : array
        Scattered parameter vectors.

    """
    
    ## might be a better way to handle arrays, fine for now
    if type(log_args_idx) is not list:
        log_args_idx = list(log_args_idx)
    
    ## initialize
    scattered_theta = xp.empty_like(theta_true)
    
    ## orbital separation to log space
    theta_temp = theta_true.copy()
    for ii in log_args_idx:
        theta_temp[:,ii] = xp.log10(theta_true[:,ii])
    
    ## scatter assuming Gaussian likelihoods
    for ii in range(theta_true.shape[-1]):
        scattered_theta[:,ii] = theta_temp[:,ii] + err[ii]*st.norm(rng).rvs(theta_true.shape[0])
    
    ## orbital separation back to linear space
    for ii in log_args_idx:
        scattered_theta[:,log_args_idx] = 10**scattered_theta[:,log_args_idx]
    
    ## apply prior bounds
    if bound:
        scattered_theta = apply_theta_lims(scattered_theta,**kwargs)
    
    return scattered_theta

def to_numpy(arr):
    if xp is np:
        return arr
    elif type(arr) is np.ndarray:
        return arr
    else:
        return xp.asnumpy(arr)


def lisa_noise_psd(fs):
    """
    Simple fixed LISA noise PSD based on Robson+19

    Parameters
    ----------
    fs : array
        Frequencies at which to compute the Robson+19 approximate LISA noise PSD.

    Returns
    -------
    noise_psd : array
        LISA noise PSD at the desired frequencies.

    """
    
    L = 2.5e9
    fstar = c/(2*xp.pi*L)
    
    S_oms = (1.5e-11)**2 * (1 + (2e-3 / fs)**4)
    
    S_acc = (3e-15)**2 * (1 + (0.4e-3/fs)**2)*(1 + (fs/(8e-3))**4)
    
    noise_psd = (1/L**2) * (S_oms + 2*(1 + xp.cos(fs/fstar)**2) * S_acc/(2*xp.pi*fs)**4)
    
    return noise_psd
    
    
def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = plt.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

def set_style():
    plt.style.use('default')
    default_cycler=cycler(color=['mediumorchid','teal','navy','firebrick','goldenrod','slategrey'])
    plt.rc('axes', prop_cycle=default_cycler)
    
    plt.rcParams['font.family'] = 'STIXGeneral'  # Closely matches Computer Modern
    plt.rcParams['mathtext.fontset'] = 'stix'    # Use STIX for math
    
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    return
