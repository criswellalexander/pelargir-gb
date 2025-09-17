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
try:
    if ('PELARGIR_GPU' in os.environ.keys()) and os.environ['PELARGIR_GPU']:
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

import scipy.special as sc



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



class BaseDist:
    
    def __init__(self,cast=False):
        
        gpu_flag = ('PELARGIR_GPU' in os.environ.keys()) and os.environ['PELARGIR_GPU']
        eryn_flag = ('PELARGIR_ERYN' in os.environ.keys()) and os.environ['PELARGIR_ERYN']
        if gpu_flag and eryn_flag and cast:
            self.cast = xp.asnumpy
            self.invcast = xp.asarray
        else:
            self.cast = xp.asarray
            self.invcast = xp.asarray
        
        
    def rvs(self,size=1):
        
        return self.cast(self._rvs(size=size))
    
    def logpdf(self,x):
        
        return self.cast(self._logpdf(self.invcast(x)))
    
    def logpmf(self,x):
        
        return self.cast(self._logpmf(self.invcast(x)))

class norm(BaseDist):
    
    def __init__(self,rng,loc=0.0,scale=1.0,cast=False):
        
        super().__init__(cast=cast)
        
        self.loc = loc
        self.scale = scale
        self.rng = rng
        
        
    def _rvs(self,size=1):
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
        
        return self.loc + self.scale*self.rng.standard_normal(size=size)
        
    def _logpdf(self, x):
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
    

class uniform(BaseDist):
    
    def __init__(self,rng,loc=0.0,scale=1.0,cast=False):
        
        super().__init__(cast=cast)
        
        self.loc = loc
        self.scale = scale
        self.rng = rng
    
    
    
    def _rvs(self,size=1):
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
        return self.loc + self.scale*self.rng.uniform(size=size)
    
    def _logpdf(self, x):
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
        
        return xp.where(xp.logical_and(x>=self.loc,x<=(self.loc+self.scale)),0,-xp.inf)

class truncnorm(BaseDist):
    
    def __init__(self,rng,loc=0,scale=1,a_min=-1,a_max=1,cast=False):
        """
        

        Parameters
        ----------
        rng : TYPE
            DESCRIPTION.
        loc : TYPE, optional
            DESCRIPTION. The default is 0.
        scale : TYPE, optional
            DESCRIPTION. The default is 1.
        a_min : TYPE, optional
            Truncation minimum. Note that this is an actual value (as opposed to a number of sigmas), 
            diverging from the scipy convention. The default is -1.
        a_max : TYPE, optional
            Truncation maximum. Note that this is an actual value (as opposed to a number of sigmas), 
            diverging from the scipy convention. The default is -1.

        Returns
        -------
        None.

        """
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.loc = loc
        self.scale = scale
        self.a_min = a_min
        self.a_max = a_max
        
    def _rvs(self,size=1):
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
            temp_arr = self.loc + self.scale*self.rng.standard_normal(size=int(1.5*size))
            keep = xp.logical_and(temp_arr>=self.a_min,temp_arr<=self.a_max)
            N_keep = xp.sum(keep)
            if N_keep > (size - N):
                draws[N:] = temp_arr[keep][:size-N]
            else:
                draws[N:N+N_keep] = temp_arr[keep]
            N += N_keep
            
        
        return draws
        
    def _logpdf(self, x):
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
        
        truncnorm_logpdf = xp.where(xp.logical_and(x>=self.a_min,x<=self.a_max),norm_logpdf,-xp.inf)
        
        return truncnorm_logpdf

class gamma(BaseDist):
    
    def __init__(self,rng,a,scale=1.0,cast=False):
        
        super().__init__(cast=cast)
        
        self.a = a
        self.scale = scale
        self.rng = rng
    
    def _rvs(self,size=1):
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
    
    def _logpdf(self, x):
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

class invgamma(BaseDist):
    
    def __init__(self,rng,a,scale=1.0,cast=False):
        r"""
        Inverse Gamma distribution with PDF
        
        $$f(x, a) = \frac{x^{-a-1}}{\Gamma(a)} \exp(-\frac{1}{x})$$
        
        where $\Gamma(a)$ is the gamma function.
        
        As cupy.random does not have a method for sampling from the inverse gamma distribution directly,
        the .rvs method for the class samples 1/x from the corresponding gamma distribution and returns
        its inverse.

        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        a : float
            Shape parameter

        Returns
        -------
        None.

        """
        
        super().__init__(cast=cast)
        self.a = a
        self.rng = rng
    
    def _rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the inverse Gamma distribution with shape a and scale = 1/beta
        """
        
        return self.rng.gamma(self.a,size=size)**(-1)
    
    def _logpdf(self, x):
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

        return -(self.a+1) * xp.log(x) - sc.gammaln(self.a) - 1.0/x

class powerlaw(BaseDist):
    
    def __init__(self,rng,alpha,loc=0.0,scale=1.0,cast=False):
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
        
        super().__init__(cast=cast)
        
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.rng = rng
        
    
    def _rvs(self,size=1):
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
    
    def _logpdf(self,x):
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
    

class poisson(BaseDist):
    
    def __init__(self,rng,lam,cast=False):
        r"""
        
        Poisson distribution with PMF
        
        $$f(k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
    
        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        lam : float
            Poisson lambda rate parameter

        Returns
        -------
        None.

        """
        super().__init__(cast=cast)
        
        self.rng = rng
        self.lam = lam
    
    def _rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Poisson-distributed samples

        """
        
        return self.rng.poisson(lam=self.lam,size=size)
    
    def _logpmf(self,k):
        """
        log PMF of the Poisson distribution.

        Parameters
        ----------
        k : numpy or cupy array of ints
            Values at which to compute the log PMF.

        Returns
        -------
        (numpy or cupy array)
            Values of the Poisson log PMF

        """
        
        return sc.xlogy(k, self.lam) - sc.gammaln(k + 1) - self.lam