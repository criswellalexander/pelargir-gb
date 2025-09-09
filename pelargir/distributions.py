#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:05:07 2025

@author: Alexander W. Criswell

This essentially re-implements a slim version scipy's approach to distributions (i.e., objects with 
.rvs, .pdf, etc. methods where the shape/loc/other parameters can be set at initialization), 
but with cupy as the underlying engine.

We only implement .logpdf and .rvs as methods.

"""
import os
import numpy as np
import cupy as cp
xp = np
import scipy.special as sc

if xp == cp:
    os.environ['SCIPY_ARRAY_API'] = 1



## following scipy, define the statistical functions for the normal distribution 
## where they can be used by multiple classes
## this code is directly adapted from scipy.stats._continuous_distns.py
# Normal distribution

# loc = mu, scale = std
# Keep these implementations out of the class definition so they can be reused
# by other distributions.
_norm_pdf_C = xp.sqrt(2*xp.pi)
_norm_pdf_logC = xp.log(_norm_pdf_C)


# def _norm_pdf(x):
#     return xp.exp(-x**2/2.0) / _norm_pdf_C


# def _norm_logpdf(x):
#     return -x**2 / 2.0 - _norm_pdf_logC


# def _norm_cdf(x):
#     return sc.ndtr(x)


# def _norm_logcdf(x):
#     return sc.log_ndtr(x)


# def _norm_ppf(q):
#     return sc.ndtri(q)


# def _norm_sf(x):
#     return _norm_cdf(-x)


# def _norm_logsf(x):
#     return _norm_logcdf(-x)


# def _norm_isf(q):
#     return -_norm_ppf(q)



# class BaseDistribution():
    
#     def __init__(self,rng):
#         """
        

#         Returns
#         -------
#         None.

#         """
        

#     def rvs(self,size=1):
#         """
        

#         Parameters
#         ----------
#         size : (int or tuple of ints), optional
#             Output size for the draws. The default is 1.

#         Returns
#         -------
#         Randomly drawn values from the desired distribution.

#         """
        
#         return self._rvs(**self.dist_args,size=size)
    
#     def logpdf(self,x):
        
        
#         return self.dist.logpdf(x,**self.dist_args)



class normal:
    
    def __init__(self,rng,loc=0.0,scale=1.0):
        
        
        self.loc = loc
        self.scale = scale
        self.rng = rng
        
        
    def rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the normal distribution with mu = loc and sigma=scale.

        """
        
        return self.loc + self.scale*self.rng.standard_normal(size)
        
    def logpdf(self, x):
        """
        log PDF of the normal distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the normal logPDF.

        """
        x = (x - self.loc)/self.scale
        
        return -0.5*x**2 - _norm_pdf_logC
    

class uniform:
    
    def __init__(self,rng,loc=0.0,scale=1.0):
        
        
        self.loc = loc
        self.scale = scale
        self.rng = rng
    
    
    
    def rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the uniform distribution with lower bound loc and upper bound loc+scale.

        """
        return self.loc + self.scale*self.rng.uniform(size)
    
    def logpdf(self, x):
        """
        log PDF of the uniform distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the normal logPDF.

        """
        
        return xp.where(xp.logical_and(x>=self.loc,x<=(self.loc+self.scale)),0,-np.inf)

class truncnorm:
    
    def __init__(self,rng,loc=0,scale=1,a=-1,b=1):
        
        self.rng = rng
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
        
    def rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the truncated normal distribution with mu = loc, sigma=scale, and bounds [a_min,a_max]

        """
        
        N = 0
        draws = xp.zeros(size)
        while N < size:
            temp_arr = self.loc + self.scale*self.rng.standard_normal(size-N)
            keep = xp.logical_and(temp_arr>=self.a_min,temp_arr<=self.a_max)
            N_keep = np.sum(keep)
            draws[N:N+N_keep] = temp_arr[keep]
            N += N_keep
            
        
        return draws
        
    def logpdf(self, x):
        """
        log PDF of the truncated normal distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the truncated normal logPDF.

        """
        x = (x - self.loc)/self.scale
        
        norm_logpdf = -0.5*x**2 - _norm_pdf_logC
        
        truncnorm_logpdf = xp.where(xp.logical_and(x>=self.a_min,x<=self.a_max),norm_logpdf,-np.inf)
        
        return truncnorm_logpdf

class gamma:
    
    def __init__(self,rng,a,scale=1.0):
        
        self.a = a
        self.scale = scale
        self.rng = rng
    
    def rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the Gamma distribution with shape a and scale = 1/beta
        """
        
        return self.rng.gamma(self.a,scale=self.scale,size=size)
    
    def logpdf(self, x):
        """
        log PDF of the Gamma distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the Gamma logPDF.

        """
        
        return sc.xlogy(self.a-1.0,x) - x - sc.gammaln(self.a)

class powerlaw:
    
    def __init__(self,rng,alpha,loc=0.0,scale=1.0):
        """
        
        Power law distribution with PDF:
        
        $$f(x,\alpha) = x^{\alpha-1}$$
        
        Note that the slope is alpha - 1. See scipy.stats.powerlaw for further information.

        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        alpha : float
            power law slope + 1.
        loc : float, optional
            Lower bound of the distribution. The default is 0.0.
        scale : float, optional
            Upper bound of the distribution. The default is 1.0.

        Returns
        -------
        None.

        """
        
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.rng = rng
        
    
    def rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the power law distribution with slope alpha-1

        """
        
        return self.loc + self.scale*self.rng.power(self.alpha,size=size)
    
    def logpdf(self,x):
        """
        log PDF of the power law distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the power law logPDF.

        """
        
        return xp.log(self.alpha) + sc.xlogy(self.alpha-1,(x-self.loc)/self.scale) - self.scale
    

