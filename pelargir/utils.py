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