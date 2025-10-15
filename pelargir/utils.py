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


msun_kg_conv = xp.array((1*u.Msun).to(u.kg).value) ## to kg
kpc_m_conv = xp.array((1*u.kpc).to(u.m).value) ## to m
au_m_conv = xp.array((1*u.AU).to(u.m).value) ## to m
G = 6.6743e-11 ## m^3 kg^-1 s^-2
c = 2.99792458e8 ## m/s

def get_mc(m_1,m_2):
    return (m_1*m_2)**(3/5) / (m_1+m_2)**(1/5)

def get_amp_freq(theta):
    m_1 = theta[0]*msun_kg_conv ## to kg
    m_2 = theta[1]*msun_kg_conv ## to kg
    d_L = theta[2]*kpc_m_conv ## to m
    a = theta[3]*au_m_conv ## to m
    amp = (8/xp.sqrt(5)) * (G**2/c**4) * (m_1*m_2)/(d_L*a)
    fgw = 1/xp.pi * xp.sqrt(G*(m_1+m_2)/a**3)
    return amp, fgw

def to_numpy(arr):
    if xp is np:
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
    
    ## TODO -- add some font size/style, etc. here
    
    return
    