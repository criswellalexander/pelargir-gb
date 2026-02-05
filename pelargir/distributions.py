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
gpu = False
try:
    if ('PELARGIR_GPU' in os.environ.keys()) and int(os.environ['PELARGIR_GPU']):
        import cupy as xp
        ## check for available devices
        if xp.cuda.is_available():
            print("GPU requested and available; running Pelargir population inference on GPU.")
            os.environ['SCIPY_ARRAY_API'] = '1'
            from cupyx.scipy import special as xsc
            import cupyx
            gpu = True
        else:
            print("GPU requested but no device is available. Defaulting to CPU.")
            import numpy as xp
            import scipy.special as xsc
    else:
        print("Running Pelargir population inference on CPU.")
        import numpy as xp
        import scipy.special as xsc
except:
    print("An error occurred in initializing GPU functionality. Defaulting to CPU.")
    import numpy as xp
    import scipy.special as xsc

import scipy.special as sc
from numpy.linalg import LinAlgError
import warnings


## following scipy, define the statistical functions for the normal distribution 
## where they can be used by multiple classes
## this code is directly adapted from scipy.stats._continuous_distns.py
# Normal distribution

# loc = mu, scale = std
# Keep these implementations out of the class definition so they can be reused
# by other distributions.
_norm_pdf_C = xp.sqrt(2*xp.pi)
_norm_pdf_logC = xp.log(_norm_pdf_C)

## other code pulled from scipy
# logsumexp trick for log(p + q) with only log(p) and log(q)
def _log_sum(log_p, log_q):
    return xsc.logsumexp(xp.array([log_p, log_q]), axis=0)


# same as above, but using -exp(x) = exp(x + πi)
def _log_diff(log_p, log_q):
    return xsc.logsumexp(xp.array([log_p, log_q+xp.pi*1j]), axis=0)

def _norm_cdf(x):
    return xsc.ndtr(x)

def _norm_logcdf(x):
    return xsc.log_ndtr(x)

def _log_gauss_mass(a, b):
    """Log of Gaussian probability mass within an interval"""
    a = xp.array(a)
    b = xp.array(b)
    a, b = xp.broadcast_arrays(a, b)

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a, b):
        return _log_diff(_norm_logcdf(b), _norm_logcdf(a))

    def mass_case_right(a, b):
        return mass_case_left(-b, -a)

    def mass_case_central(a, b):
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return xsc.log1p(-_norm_cdf(a) - _norm_cdf(-b))

    # _lazyselect not working; don't care to debug it
    out = xp.full_like(a, fill_value=xp.nan, dtype=xp.complex128)
    if a[case_left].size:
        out[case_left] = mass_case_left(a[case_left], b[case_left])
    if a[case_right].size:
        out[case_right] = mass_case_right(a[case_right], b[case_right])
    if a[case_central].size:
        out[case_central] = mass_case_central(a[case_central], b[case_central])
    return xp.real(out)  # discard ~0j


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
        
        gpu_flag = ('PELARGIR_GPU' in os.environ.keys()) and int(os.environ['PELARGIR_GPU'])
        eryn_flag = ('PELARGIR_ERYN' in os.environ.keys()) and int(os.environ['PELARGIR_ERYN'])
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
        
        return -0.5*x**2 - _norm_pdf_logC - xp.log(self.scale)
    
class multivariate_normal(BaseDist):
    
    def __init__(self,rng,mean,cov,cast=False,check_valid='ignore',
                            tol=1e-08, method='cholesky'):
        '''
        Create the multivariate normal object. If vectorized, 
        leading axis MUST be the number of distributions.

        Parameters
        ----------
        rng : Generator
            Numpy/cupy Generator object.
        mean : array
            Distribution mean(s). Can be 1D or N+1D. If 1D, these are all parameter means for a single distribution;
            if ND, leading axes will be considered separate ditributions.
        cov : array
            Distribution covariance(s). Can be 2D or N+2D. If 2D, this is the covariance matrix for a single distribution;
            if ND, leading axes will be considered separate ditributions.
        cast : bool, optional
            Whether to cast results to numpy.

        Returns
        -------
        None.

        '''
        super().__init__(cast=cast)
        
        self.mean = mean
        self.cov = cov
        self.rng = rng
        
        self.method = method
        self.tol = tol
        self.check_valid = check_valid
        
        self.precalculate_pdf_terms()
        
        if gpu:
            self.precalculate_rvs_terms()
            self._rvs = self._rvs_gpu
        else:
            self._rvs = self._rvs_cpu
    
    def precalculate_rvs_terms(self):
        '''
        Perform amortized calculations to support rvs draws.

        Returns
        -------
        None.

        '''
        if (self.cov.shape[-1] != self.cov.shape[-2]):
            raise ValueError('cov must be 2 dimensional and square')
        if self.mean.shape[-1] != self.cov.shape[-1]:
            raise ValueError('mean and cov must have same length')

        

        if self.method not in {'eigh', 'svd', 'cholesky'}:
            raise ValueError(
                "method must be one of {'eigh', 'svd', 'cholesky'}")

        if self.check_valid != 'ignore':
            if self.check_valid != 'warn' and self.check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")

        if self.check_valid == 'warn':
            with cupyx.errstate(linalg='raise'):
                try:
                    self.decomp = xp.linalg.cholesky(self.cov)
                except LinAlgError:
                    with cupyx.errstate(linalg='ignore'):
                        if self.method != 'cholesky':
                            if self.method == 'eigh':
                                (s, u) = xp.linalg.eigh(self.cov)
                                psd = not xp.any(s < -self.tol)
                            if self.method == 'svd':
                                (u, s, vh) = xp.linalg.svd(self.cov)
                                psd = xp.allclose(xp.dot(vh.T * s, vh),
                                                    self.cov, rtol=self.tol, atol=self.tol)
                            self.decomp = u * xp.sqrt(xp.abs(s))
                            if not psd:
                                warnings.warn("covariance is not positive-" +
                                              "semidefinite, output may be " +
                                              "invalid.", RuntimeWarning)

                        else:
                            warnings.warn("covariance is not positive-" +
                                          "semidefinite, output *is* " +
                                          "invalid.", RuntimeWarning)
                            self.decomp = xp.linalg.cholesky(self.cov)

        else:
            with cupyx.errstate(linalg=self.check_valid):
                try:
                    if self.method == 'cholesky':
                        self.decomp = xp.linalg.cholesky(self.cov)
                    elif self.method == 'eigh':
                        (s, u) = xp.linalg.eigh(self.cov)
                        self.decomp = u * xp.sqrt(xp.abs(s))
                    elif self.method == 'svd':
                        (u, s, vh) = xp.linalg.svd(self.cov)
                        self.decomp = u * xp.sqrt(xp.abs(s))

                except LinAlgError:
                    raise LinAlgError("Matrix is not positive definite; if " +
                                      "matrix is positive-semidefinite, set" +
                                      "'check_valid' to 'warn'")

        
    
    def _rvs_gpu(self, size=None, dtype=float):
        """Returns an array of samples drawn from the multivariate normal
        distributions. 

        .. warning::
            This function calls one or more cuSOLVER routine(s) which may yield
            invalid results if input conditions are not met.
            To detect these invalid results, you can set the `linalg`
            configuration to a value that is not `ignore` in
            :func:`cupyx.errstate` or :func:`cupyx.seterr`.

        .. seealso::
            - :func:`cupy.random.multivariate_normal` for full documentation
            - :meth:`numpy.random.RandomState.multivariate_normal`
        """
        if size is None:
            shape = []
        elif isinstance(size, (int, xp.integer)):
            shape = [size]
        else:
            shape = size
        
        final_shape = list(shape[:])
        for ndx in range(self.mean.ndim):
            final_shape.append(self.mean.shape[ndx])
        
        x = self.rng.standard_normal(final_shape,
                                     dtype=dtype).reshape(-1, self.mean.shape[-1])
        x = xp.einsum('i...jk,i...k->i...j',self.decomp, x)
        x.shape = tuple(final_shape)
        x += self.mean
        return x    
    
    def _rvs_cpu(self,size=1):
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
        
        return self.rng.multivariate_normal(self.mean,self.cov,size=size)
    
    def precalculate_pdf_terms(self):
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
        # NumPy broadcasts `eigh`.
        self.vals, self.vecs = xp.linalg.eigh(self.cov)
        
        # Compute the log determinants across the second axis.
        self.logdets    = xp.sum(xp.log(self.vals), axis=1)
        
        # Invert the eigenvalues.
        self.valsinvs   = 1./self.vals
        
        # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
        self.Us         = self.vecs * xp.sqrt(self.valsinvs)[:, None]
        
        # Compute prefactor for scalar normalizers.
        dim        = len(self.vals[0])
        log2pi     = xp.log(2 * xp.pi)
        self.prefac = dim*log2pi
    
    def _logpdf(self,x):
        """Compute multivariate normal log PDF over multiple sets of parameters.
        Adapted from https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
        """
        
        ## get deviations
        devs       = x - self.mean
    
        # Use `einsum` for matrix-vector multiplications across the first dimension.
        devUs      = xp.einsum('...ni,...nij->...nj', devs, self.Us)
    
        # Compute the Mahalanobis distance by squaring each term and summing.
        mahas      = xp.sum(xp.square(devUs), axis=-1)
        
        ## normalize and return
        return -0.5 * (self.prefac + mahas + self.logdets)

class truncated_multivariate_normal(BaseDist):
    
    def __init__(self,rng,mean,cov,lims,cast=False,check_valid='warn',
                            tol=1e-08, method='cholesky'):
        '''
        Create the truncated tmultivariate normal object. If vectorized, 
        leading axis MUST be the number of distributions.

        Parameters
        ----------
        rng : Generator
            Numpy/cupy Generator object.
        mean : array
            Distribution mean(s). Can be 1D or N+1D. If 1D, these are all parameter means for a single distribution;
            if ND, leading axes will be considered separate ditributions.
        cov : array
            Distribution covariance(s). Can be 2D or N+2D. If 2D, this is the covariance matrix for a single distribution;
            if ND, leading axes will be considered separate ditributions.
        lims : array
            Distribution bounds across each axis. Must broadcast to means and have trailing axis of length 2.
        cast : bool, optional
            Whether to cast results to numpy.

        Returns
        -------
        None.

        '''
        super().__init__(cast=cast)
        
        self.mean = mean
        self.cov = cov
        self.lims = lims
        self.rng = rng
        
        self.method = method
        self.tol = tol
        self.check_valid = check_valid
        
        self.precalculate_pdf_terms()
        
        if gpu:
            self.precalculate_rvs_terms()
            self._rvs_unbound = self._rvs_gpu
        else:
            self._rvs_unbound = self._rvs_cpu
        
        self._rvs = self._rvs_bound

    def enforce_lims(self,draw):
         
        new_shape = tuple([1 for i in range(draw.ndim-1)]+[self.lims.shape[0]])
        lower = self.lims[:,0].reshape(new_shape)
        upper = self.lims[:,1].reshape(new_shape)
        
        filt = xp.prod((draw < lower)*(draw > upper),axis=-1)
        while xp.any(filt):
            new_draw = self._rvs_unbound()
            # import pdb; pdb.set_trace()
            draw[...,filt,:] = new_draw[...,filt,:]
            filt = xp.prod((draw < lower)*(draw > upper),axis=-1)
            
        return draw
    
    def _rvs_bound(self,size=None,dtype=float):
        
        return self.enforce_lims(self._rvs_unbound(size=size,dtype=dtype))
        
    
    def precalculate_rvs_terms(self):
        '''
        Perform amortized calculations to support rvs draws.

        Returns
        -------
        None.

        '''
        if (self.cov.shape[-1] != self.cov.shape[-2]):
            raise ValueError('cov must be 2 dimensional and square')
        if self.mean.shape[-1] != self.cov.shape[-1]:
            raise ValueError('mean and cov must have same length')

        

        if self.method not in {'eigh', 'svd', 'cholesky'}:
            raise ValueError(
                "method must be one of {'eigh', 'svd', 'cholesky'}")

        if self.check_valid != 'ignore':
            if self.check_valid != 'warn' and self.check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")

        if self.check_valid == 'warn':
            with cupyx.errstate(linalg='raise'):
                try:
                    self.decomp = xp.linalg.cholesky(self.cov)
                except LinAlgError:
                    with cupyx.errstate(linalg='ignore'):
                        if self.method != 'cholesky':
                            if self.method == 'eigh':
                                (s, u) = xp.linalg.eigh(self.cov)
                                psd = not xp.any(s < -self.tol)
                            if self.method == 'svd':
                                (u, s, vh) = xp.linalg.svd(self.cov)
                                psd = xp.allclose(xp.dot(vh.T * s, vh),
                                                    self.cov, rtol=self.tol, atol=self.tol)
                            self.decomp = u * xp.sqrt(xp.abs(s))
                            if not psd:
                                warnings.warn("covariance is not positive-" +
                                              "semidefinite, output may be " +
                                              "invalid.", RuntimeWarning)

                        else:
                            warnings.warn("covariance is not positive-" +
                                          "semidefinite, output *is* " +
                                          "invalid.", RuntimeWarning)
                            self.decomp = xp.linalg.cholesky(self.cov)

        else:
            with cupyx.errstate(linalg=self.check_valid):
                try:
                    if self.method == 'cholesky':
                        self.decomp = xp.linalg.cholesky(self.cov)
                    elif self.method == 'eigh':
                        (s, u) = xp.linalg.eigh(self.cov)
                        self.decomp = u * xp.sqrt(xp.abs(s))
                    elif self.method == 'svd':
                        (u, s, vh) = xp.linalg.svd(self.cov)
                        self.decomp = u * xp.sqrt(xp.abs(s))

                except LinAlgError:
                    raise LinAlgError("Matrix is not positive definite; if " +
                                      "matrix is positive-semidefinite, set" +
                                      "'check_valid' to 'warn'")

        
    
    def _rvs_gpu(self, size=None, dtype=float):
        """Returns an array of samples drawn from the multivariate normal
        distributions. 

        .. warning::
            This function calls one or more cuSOLVER routine(s) which may yield
            invalid results if input conditions are not met.
            To detect these invalid results, you can set the `linalg`
            configuration to a value that is not `ignore` in
            :func:`cupyx.errstate` or :func:`cupyx.seterr`.

        .. seealso::
            - :func:`cupy.random.multivariate_normal` for full documentation
            - :meth:`numpy.random.RandomState.multivariate_normal`
        """
        if size is None:
            shape = []
        elif isinstance(size, (int, xp.integer)):
            shape = [size]
        else:
            shape = size
        
        final_shape = list(shape[:])
        for ndx in range(self.mean.ndim):
            final_shape.append(self.mean.shape[ndx])
        
        x = self.rng.standard_normal(final_shape,
                                     dtype=dtype).reshape(-1, self.mean.shape[-1])
        x = xp.einsum('i...jk,i...k->i...j',self.decomp, x)
        x.shape = tuple(final_shape)
        x += self.mean
        return x    
    
    def _rvs_cpu(self,size=1):
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
        
        return self.rng.multivariate_normal(self.mean,self.cov,size=size)
    
    def precalculate_pdf_terms(self):
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
        # NumPy broadcasts `eigh`.
        self.vals, self.vecs = xp.linalg.eigh(self.cov)
        
        # Compute the log determinants across the second axis.
        self.logdets    = xp.sum(xp.log(self.vals), axis=1)
        
        # Invert the eigenvalues.
        self.valsinvs   = 1./self.vals
        
        # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
        self.Us         = self.vecs * xp.sqrt(self.valsinvs)[:, None]
        
        # Compute prefactor for scalar normalizers.
        dim        = len(self.vals[0])
        log2pi     = xp.log(2 * xp.pi)
        self.prefac = dim*log2pi
    
    def _logpdf(self,x):
        """Compute multivariate normal log PDF over multiple sets of parameters.
        Adapted from https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
        """
        
        ## get deviations
        devs       = x - self.mean
    
        # Use `einsum` for matrix-vector multiplications across the first dimension.
        devUs      = xp.einsum('...ni,...nij->...nj', devs, self.Us)
    
        # Compute the Mahalanobis distance by squaring each term and summing.
        mahas      = xp.sum(xp.square(devUs), axis=-1)
        
        ## normalize and return
        return -0.5 * (self.prefac + mahas + self.logdets)

class uniform(BaseDist):
    
    def __init__(self,rng,loc=0.0,scale=1.0,cast=False):
        
        super().__init__(cast=cast)
        
        self.loc = loc
        self.scale = scale
        self.rng = rng
        self.log_factor = -xp.log(self.scale)
    
    
    
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
        
        return xp.where(xp.logical_and(x>=self.loc,x<=(self.loc+self.scale)),self.log_factor,-xp.inf)

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
            diverging from the scipy convention. The default is 1.

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
        
        ## define in terms of standard deviations as well for pdf normalization
        self.scale_min = (self.a_min-self.loc)/self.scale
        self.scale_max = (self.a_max-self.loc)/self.scale
        self.normalization_fac = _log_gauss_mass(self.scale_min, self.scale_max)
        
    def _rvs(self,size=(1,)):
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
        draws = xp.zeros(size).flatten()
        while N < draws.size:
            temp_arr = self.loc + self.scale*self.rng.standard_normal(size=int(1.5*draws.size))
            keep = xp.logical_and(temp_arr>=self.a_min,temp_arr<=self.a_max)
            N_keep = xp.sum(keep)
            if N_keep > (draws.size - N):
                draws[N:] = temp_arr[keep][:draws.size-N]
            else:
                draws[N:N+N_keep] = temp_arr[keep]
            N += N_keep
        
        ## reshape to requested shape
        draws = draws.reshape(size)
        
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
        xprime = (x - self.loc)/self.scale
        
        norm_logpdf = -0.5*xprime**2 - _norm_pdf_logC - xp.log(self.scale) - self.normalization_fac
        
        truncnorm_logpdf = xp.where(xp.logical_and(x>=self.a_min,x<=self.a_max),norm_logpdf,-xp.inf)
        
        return truncnorm_logpdf

class exponential(BaseDist):
    
    def __init__(self,rng,loc=0,scale=1,a_min=1,cast=False):
        """
        

        Parameters
        ----------
        rng : TYPE
            DESCRIPTION.
        loc : TYPE, optional
            DESCRIPTION. The default is 0.
        scale : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.loc = loc
        self.scale = scale
        
    def _rvs(self,size=(1,)):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the exponntial distribution with mu = loc, sigma=scale

        """
        
        return self.loc + self.rng.exponential(scale=self.scale,size=size)
        
    def _logpdf(self, x):
        """
        log PDF of the truncated exponential distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the truncated exponential logPDF.

        """
        
        exp_logpdf = - xp.log(self.scale) - (x-self.loc)/self.scale
        
        truncexp_logpdf = xp.where(x>=self.loc,exp_logpdf,-xp.inf)
        
        return truncexp_logpdf

class truncexp(BaseDist):
    
    def __init__(self,rng,loc=0,scale=1,a_min=1,cast=False):
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
            Left-hand truncation minimum. Note that this is an actual value (as opposed to a number of sigmas), 
            diverging from the scipy convention. The default is 1.

        Returns
        -------
        None.

        """
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.loc = loc
        self.scale = scale
        self.a_min = a_min
        
        ## ensure pdf normalization
        self.normalization_fac = -self.a_min/self.scale ## probability mass from a_min to +inf
        
    def _rvs(self,size=(1,)):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the truncated exponntial distribution with mu = loc, sigma=scale, and bounds [a_min,+inf]

        """
        
        N = 0
        draws = xp.zeros(size).flatten()
        while N < draws.size:
            temp_arr = self.loc + self.rng.exponential(scale=self.scale,size=int(1.5*draws.size))
            keep = temp_arr>=self.a_min
            N_keep = xp.sum(keep)
            if N_keep > (draws.size - N):
                draws[N:] = temp_arr[keep][:draws.size-N]
            else:
                draws[N:N+N_keep] = temp_arr[keep]
            N += N_keep
        
        ## reshape to requested shape
        draws = draws.reshape(size)
        
        return draws
        
    def _logpdf(self, x):
        """
        log PDF of the truncated exponential distribution

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the truncated exponential logPDF.

        """
        
        exp_logpdf = - xp.log(self.scale) - x/self.scale - self.normalization_fac
        
        truncexp_logpdf = xp.where(x>=self.a_min,exp_logpdf,-xp.inf)
        
        return truncexp_logpdf


class gaussian_exponential_mixture(BaseDist):
    
    def __init__(self,rng,x0,disk_scale,bulge_scale,beta,bulge_cut=None,cast=False):
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.x0 = x0
        self.bulge_scale = bulge_scale
        self.disk_scale = disk_scale
        self.bulge_cut = bulge_cut
        self.beta = beta
        
        if self.bulge_cut is not None:
            self.bulge_dist = truncnorm(self.rng,
                                        loc=0,
                                        scale=self.bulge_scale,
                                        a_min=-self.bulge_cut,
                                        a_max=self.bulge_cut)
        else:
            self.bulge_dist = norm(self.rng,loc=0,scale=self.bulge_scale)
    
    
    def _rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the simplified 1D Galaxy model.
        """
        draws = xp.empty(size)
        ## draw bulge with probaility beta, disk with probability (1-beta)
        mix_bit = (self.rng.uniform(size=size) <= self.beta)
        Nbulge = int(xp.sum(mix_bit,dtype='int'))
        Ndisk = draws.size - Nbulge
        if Nbulge > 0:
            draws[mix_bit] = self.x0 + self.bulge_dist.rvs(size=Nbulge)
        if Ndisk > 0:
            inv_mix_bit = xp.invert(mix_bit)
            dir_bit = (self.rng.uniform(size=size) <= 0.5)
            Nfar = int(xp.sum(dir_bit[inv_mix_bit],dtype='int'))
            Nnear = dir_bit[inv_mix_bit].size - Nfar
            if Nfar > 0:
                draws[inv_mix_bit*dir_bit] = self.x0 + self.rng.exponential(scale=self.disk_scale,
                                                                            size=Nfar)
            if Nnear > 0:
                draws[inv_mix_bit*xp.invert(dir_bit)] = self.x0 - self.rng.exponential(scale=self.disk_scale,
                                                                                              size=Nnear)
        
        ## ensure distance measurement by taking absolute value
        draws = xp.abs(draws)
        
        return draws
    
    def _logpdf(self,x):
        """
        log PDF of the simplified 1D Galaxy model

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the Normal-exponential mixture model logPDF.

        """
        ## where x is distance from SSB, transform to Galactocentric
        x_towards = x - self.x0
        x_away = -x - self.x0
        
        ## will need a logsumexp here
        # logpdf_bulge_tw = xp.log(self.beta) - 0.5*(x_towards/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        # logpdf_bulge_aw = xp.log(self.beta) - 0.5*(x_away/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        logpdf_bulge_tw = xp.log(self.beta) + self.bulge_dist.logpdf(x_towards) #- xp.log(2)
        # logpdf_bulge_aw = xp.log(self.beta) + self.bulge_dist.logpdf(x_away) - xp.log(2)
        logpdf_disk_tw = xp.log(1-self.beta) - xp.log(self.disk_scale) - xp.abs(x_towards/self.disk_scale) - xp.log(2)
        logpdf_disk_aw = xp.log(1-self.beta) - xp.log(self.disk_scale) - xp.abs(x_away/self.disk_scale) -xp.log(2)
        # pdftowards = self.beta*() + (1-self.beta)*((1/self.disk_scale)*xp.exp(-xp.abs(x_towards/self.disk_scale)))
        
        
        return xsc.logsumexp(xp.array([logpdf_bulge_tw,logpdf_disk_tw,logpdf_disk_aw]),axis=0)


class cored_gaussian_exponential_mixture(BaseDist):
    
    def __init__(self,rng,x0,disk_scale,bulge_scale,beta,bulge_cut=None,disk_cut=None,cast=False):
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.x0 = x0
        self.bulge_scale = bulge_scale
        self.disk_scale = disk_scale
        self.bulge_cut = bulge_cut
        self.disk_cut = disk_cut
        self.beta = beta
        
        if self.bulge_cut is not None:
            self.bulge_dist = truncnorm(self.rng,
                                        loc=0,
                                        scale=self.bulge_scale,
                                        a_min=-self.bulge_cut,
                                        a_max=self.bulge_cut)
        else:
            self.bulge_dist = norm(self.rng,loc=0,scale=self.bulge_scale)
        
        if self.disk_cut is not None:
            self.disk_dist = truncexp(self.rng,
                                      loc=0,
                                      scale=self.disk_scale,
                                      a_min=self.disk_cut)
        else:
            self.disk_dist = exponential(self.rng,loc=0,scale=self.disk_scale)
    
    
    def _rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the simplified 1D Galaxy model.
        """
        draws = xp.empty(size)
        ## draw bulge with probaility beta, disk with probability (1-beta)
        mix_bit = (self.rng.uniform(size=size) <= self.beta)
        Nbulge = int(xp.sum(mix_bit,dtype='int'))
        Ndisk = draws.size - Nbulge
        if Nbulge > 0:
            draws[mix_bit] = self.x0 + self.bulge_dist.rvs(size=Nbulge)
        if Ndisk > 0:
            inv_mix_bit = xp.invert(mix_bit)
            dir_bit = (self.rng.uniform(size=size) <= 0.5)
            Nfar = int(xp.sum(dir_bit[inv_mix_bit],dtype='int'))
            Nnear = dir_bit[inv_mix_bit].size - Nfar
            if Nfar > 0:
                draws[inv_mix_bit*dir_bit] = self.x0 + self.disk_dist.rvs(size=Nfar)
            if Nnear > 0:
                draws[inv_mix_bit*xp.invert(dir_bit)] = self.x0 - self.disk_dist.rvs(size=Nnear)
        
        ## ensure distance measurement by taking absolute value
        draws = xp.abs(draws)
        
        return draws
    
    def _logpdf(self,x):
        """
        log PDF of the simplified 1D Galaxy model

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the Normal-exponential mixture model logPDF.

        """
        ## where x is distance from SSB, transform to Galactocentric
        x_towards = x - self.x0
        x_away = self.x0 - x
        
        ## will need a logsumexp here
        # logpdf_bulge_tw = xp.log(self.beta) - 0.5*(x_towards/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        # logpdf_bulge_aw = xp.log(self.beta) - 0.5*(x_away/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        logpdf_bulge_tw = xp.log(self.beta) + self.bulge_dist.logpdf(x_towards) - xp.log(2)
        logpdf_bulge_aw = xp.log(self.beta) + self.bulge_dist.logpdf(x_away) - xp.log(2)
        logpdf_disk_tw = xp.log(1-self.beta) + self.disk_dist.logpdf(x_towards) - xp.log(2)
        logpdf_disk_aw = xp.log(1-self.beta) + self.disk_dist.logpdf(x_away) - xp.log(2)
        # pdftowards = self.beta*() + (1-self.beta)*((1/self.disk_scale)*xp.exp(-xp.abs(x_towards/self.disk_scale)))
        
        
        return xsc.logsumexp(xp.array([logpdf_bulge_tw,logpdf_bulge_aw,logpdf_disk_tw,logpdf_disk_aw]),axis=0)        

class filled_cored_gaussian_exponential_mixture(BaseDist):
    
    def __init__(self,rng,x0,disk_scale,bulge_scale,beta,bulge_cut=None,cast=False):
        
        super().__init__(cast=cast)
        
        self.rng = rng
        self.x0 = x0
        self.bulge_scale = bulge_scale
        self.disk_scale = disk_scale
        self.bulge_cut = bulge_cut
        self.disk_cut = bulge_cut
        self.beta = beta
        
        if self.bulge_cut is not None:
            self.bulge_dist = truncnorm(self.rng,
                                        loc=0,
                                        scale=self.bulge_scale,
                                        a_min=-self.bulge_cut,
                                        a_max=self.bulge_cut)
        else:
            self.bulge_dist = norm(self.rng,loc=0,scale=self.bulge_scale)
        
        if self.disk_cut is not None:
            self.disk_dist = truncexp(self.rng,
                                      loc=0,
                                      scale=self.disk_scale,
                                      a_min=self.disk_cut)
            self.disk_central = uniform(self.rng,
                                        loc=-self.disk_cut,
                                        scale=2*self.disk_cut)
        else:
            self.disk_dist = exponential(self.rng,loc=0,scale=self.disk_scale)
    
    
    def _rvs(self,size=1):
        """
        

        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Samples from the simplified 1D Galaxy model.
        """
        draws = xp.empty(size)
        ## draw bulge with probaility beta, disk with probability (1-beta)
        mix_bit = (self.rng.uniform(size=size) <= self.beta)
        Nbulge = int(xp.sum(mix_bit,dtype='int'))
        Ndisk = draws.size - Nbulge
        if Nbulge > 0:
            draws[mix_bit] = self.x0 + self.bulge_dist.rvs(size=Nbulge)
        if Ndisk > 0:
            inv_mix_bit = xp.invert(mix_bit)
            dir_bit = (self.rng.uniform(size=size) <= 0.5)
            Nfar = int(xp.sum(dir_bit[inv_mix_bit],dtype='int'))
            Nnear = dir_bit[inv_mix_bit].size - Nfar
            if Nfar > 0:
                draws[inv_mix_bit*dir_bit] = self.x0 + self.disk_dist.rvs(size=Nfar)
            if Nnear > 0:
                draws[inv_mix_bit*xp.invert(dir_bit)] = self.x0 - self.disk_dist.rvs(size=Nnear)
        
        ## ensure distance measurement by taking absolute value
        draws = xp.abs(draws)
        
        return draws
    
    def _logpdf(self,x):
        """
        log PDF of the simplified 1D Galaxy model

        Parameters
        ----------
        x : numpy or cupy array
            Values at which to compute the logpdf.

        Returns
        -------
        (numpy or cupy array)
            Values of the Normal-exponential mixture model logPDF.

        """
        ## where x is distance from SSB, transform to Galactocentric
        x_towards = x - self.x0
        x_away = self.x0 - x
        
        ## will need a logsumexp here
        # logpdf_bulge_tw = xp.log(self.beta) - 0.5*(x_towards/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        # logpdf_bulge_aw = xp.log(self.beta) - 0.5*(x_away/self.bulge_scale)**2 - xp.log(self.bulge_scale) - _norm_pdf_logC - xp.log(2)
        logpdf_bulge_tw = xp.log(self.beta) + self.bulge_dist.logpdf(x_towards) - xp.log(2)
        logpdf_bulge_aw = xp.log(self.beta) + self.bulge_dist.logpdf(x_away) - xp.log(2)
        logpdf_disk_tw = xp.log(1-self.beta) + self.disk_dist.logpdf(x_towards) - xp.log(2)
        logpdf_disk_aw = xp.log(1-self.beta) + self.disk_dist.logpdf(x_away) - xp.log(2)
        # pdftowards = self.beta*() + (1-self.beta)*((1/self.disk_scale)*xp.exp(-xp.abs(x_towards/self.disk_scale)))
        
        
        return xsc.logsumexp(xp.array([logpdf_bulge_tw,logpdf_bulge_aw,logpdf_disk_tw,logpdf_disk_aw]),axis=0)        

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
        
        return sc.xlogy(self.a-1.0,x) - x/self.scale - sc.gammaln(self.a) - sc.xlogy(self.a,self.scale)

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
        
        logp = -(self.a+1) * xp.log(x) - sc.gammaln(self.a) - 1.0/x
        
        ## returns nan for x <= 0, so cast to -inf
        logp[xp.isnan(logp)] = -xp.inf
        
        return logp

class powerlaw(BaseDist):
    
    def __init__(self,rng,alpha,loc=0.0,scale=1.0,cast=False):
        """
        
        Power law distribution with PDF:
        
        $$f(x,\alpha) = x^{\alpha}$$
        
        Note: we diverge in convention from powerlaw as defined in scipy here, as
              they define p(x,alpha) as x^(alpha-1), which does not match the power law
              conventions used in GW astro and leads to confusion.

        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        alpha : float
            power law slope
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
        
        return self.loc + self.scale*self.rng.power(self.alpha+1,size=size)
    
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
        
        return xp.log(self.alpha+1) + sc.xlogy(self.alpha,(x-self.loc)/self.scale) - xp.log(self.scale)
    

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


class vector_marginal_t(BaseDist):
    
    def __init__(self,rng,mu0,N_realz,alpha,beta,cast=False):
        r"""
        
        Marginalized Normal-inverse-Gamma distribution. Equivalent to a location/scale t distribution.
        
        Vectorized to always consider a full spectrum.
        
        Used to compute p(Sgw | vector Sgw_hat(Lambda))
        
        It is possible to instantiate the object with known hyperparameters
        so that p(Sgw | vector Sgw_hat(Lambda)) can be computed as a function
        of vector Sgw_hat(Lambda) (which is what actually changes during sampling).
    
        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        mu0 : array
            Mean of the prior on the spctrum mean. In general, should be <= than the expected PSD
            to avoid biasing the marginal prior. Default is 1e-45. If passed as an array, should be of the same shape as fg_data_psd,
            and provide a prior mean for each frequency bin.
        N_realz : int
            The number of realizations to use to estimate the marginal Poisson uncertainty. Minimum 2.
        alpha : float or array
            Shape parameter for the inverse Gamma hyperprior on the marginal prior mean. 
            In general, the marginal prior is robust to choice of hp_alpha, provided it is roughly of order the typical
            foreground PSD value (within ~10 orders of magnitude). If passed as an array, 
            should be of the same shape as fg_data_psd, and provide a value of alpha for each frequency bin.
        beta : float or array
            Scale parameter for the inverse Gamma hyper prior on the marginal prior mean. 
            If hp_beta >> the typical PSD value, the set of simulation draws is essentially ignored in favor of a large prior. However,
            if hp_beta << the typical PSD value, the marginal prior will become inverted and disfavour the region where it should be peaked.
            Fore safety reasons, beta should be within ~2 orders of magnitude of the minimum expected PSD value.
            Recommended choice is to set beta to an array 1 order of magnitude below the noise PSD in each frequency bin.
            If passed as an array, should be of the same shape as fg_data_psd, and provide a value of beta for each frequency bin.
               
        Returns
        -------
        None.

        """
        super().__init__(cast=cast)
        
        self.rng = rng
        
        ## hyperprior on the (central limit) Poisson uncertainty
        self.mu0 = mu0
        self.alpha = alpha
        self.beta = beta
        if hasattr(self.mu0,"ndim") and self.mu0.ndim==1:
            self.mu0 = self.mu0[...,xp.newaxis]
        if hasattr(self.alpha,"ndim") and self.alpha.ndim==1:
            self.alpha = self.alpha[...,xp.newaxis]
        if hasattr(self.beta,"ndim") and self.beta.ndim==1:
            self.beta = self.beta[...,xp.newaxis]
        ## number of realizations
        self.N_realz = N_realz
        
        ## precompute some parameters of the marginal prior which only rely on N_realz and the hyperprior
        ## effective sample size is nuprime = nu + N realizations
        ## nu is not bound to be an integer (but must be >0), so let's intentionally downweight the contribution from mu0
        self.nu = 1e-10
        self.nuprime = self.nu + self.N_realz
        ## alphaprime = alpha + N/2
        self.alphaprime = self.alpha + self.N_realz/2
        ## generalized t degrees of freedom = 2*alphaprime
        self.df = 2*self.alphaprime
        
        ## set remaining marginal prior parameters to None
        self.muprime = None
        self.sigmaprime = None
        self.betaprime = None
        
    def update(self,theta_spec):
        """
        Update the parameters of the conditional t distribution to be the hyperposterior
        parameters after taking into account the draws from the simulation.

        Parameters
        ----------
        theta_specs : array
            Spectra drawn from the conditional population prior, of shape (N_realz,N_frequencies).

        """
        
        ## compute remaining arguments from the theta_spec draws
        
        ## per-frequency mean of the draws
        Sf_mean = xp.mean(theta_spec,axis=1)
        
        ## sum of the spectral deviationes squared (sum((S-Smean)^2))
        Sf_sum_dev2 = xp.sum((theta_spec-Sf_mean[:,None])**2,axis=1)
        
        ## compute conditional prior parameters
        self.muprime = (self.nu*self.mu0 + self.N_realz*Sf_mean)/(self.nu + self.N_realz)
        betaprime = self.beta + 0.5*Sf_sum_dev2 + 0.5*((self.nu*self.N_realz)/(self.nu+self.N_realz))*(Sf_mean-self.mu0)**2
        self.sigmaprime = (betaprime*(self.nuprime + 1))/(self.alphaprime*self.nuprime)
        return
    
    def _logpdf(self,x):
        '''
        Get the marginal t logpdf
        
        Note: xsc.poch is the Pochhammer symbol, i.e. Gamma(z+m)/Gamma(z) where Gamma is the Gamma function.
              Hence, poch(df/2,1/2) = Gamma((df+1)/2)/Gamma(df/2).

        Parameters
        ----------
        x : array
            Array of values at which to compute the logpdf. Should at minimum be o shape (1,Nfreqs).

        Returns
        -------
        logpdf
            Natural log of the conditional location/scale t distribution at x.

        '''
        ln_coeff = xp.log(xsc.poch(0.5*self.df, 0.5)) - 0.5*(xp.log(self.df) + xp.log(xp.pi)) - xp.log(self.sigmaprime)

        return ln_coeff + -0.5*(self.df+1)*xp.log1p((((x-self.muprime)/self.sigmaprime)**2)/self.df)

class vector_marginal_logt(BaseDist):
    
    def __init__(self,rng,mu0,N_realz,alpha,beta,cast=False):
        r"""
        
        Marginalized Normal-inverse-Gamma distribution. Equivalent to a location/scale t distribution.
        
        This variant follows the same math, but assumes our variable of interest is the logPSD.
        
        Vectorized to always consider a full spectrum.
        
        Used to compute p(Sgw | vector Sgw_hat(Lambda))
        
        It is possible to instantiate the object with known hyperparameters
        so that p(Sgw | vector Sgw_hat(Lambda)) can be computed as a function
        of vector Sgw_hat(Lambda) (which is what actually changes during sampling).
    
        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        mu0 : array
            Mean of the prior on the log10 spectrum mean. Heavily down-weighted, so largely arbitrary. 
            Should be roughly of order the typical log10(PSD).
            If passed as an array, should be of the same shape as fg_data_psd,
            and provide a prior mean for each frequency bin.
        N_realz : int
            The number of realizations to use to estimate the marginal Poisson uncertainty. Minimum 2.
        alpha : float or array
            Shape parameter for the inverse Gamma hyperprior on the marginal prior mean. 
            In general, the marginal prior is robust to choice of hp_alpha, provided it is roughly of order the typical
            foreground PSD value (within ~10 orders of magnitude). If passed as an array, 
            should be of the same shape as fg_data_psd, and provide a value of alpha for each frequency bin.
        beta : float or array
            Scale parameter for the inverse Gamma hyper prior on the marginal prior mean. 
            If hp_beta >> the typical PSD value, the set of simulation draws is essentially ignored in favor of a large prior. However,
            if hp_beta << the typical PSD value, the marginal prior will become inverted and disfavour the region where it should be peaked.
            Fore safety reasons, beta should be within ~2 orders of magnitude of the minimum expected PSD value.
            Recommended choice is to set beta to an array 1 order of magnitude below the noise PSD in each frequency bin.
            If passed as an array, should be of the same shape as fg_data_psd, and provide a value of beta for each frequency bin.
               
        Returns
        -------
        None.

        """
        super().__init__(cast=cast)
        
        self.rng = rng
        
        ## hyperprior on the (central limit) Poisson uncertainty
        self.mu0 = mu0
        self.alpha = alpha
        self.beta = beta
        if hasattr(self.mu0,"ndim") and self.mu0.ndim==1:
            self.mu0 = self.mu0[...,xp.newaxis]
        if hasattr(self.alpha,"ndim") and self.alpha.ndim==1:
            self.alpha = self.alpha[...,xp.newaxis]
        if hasattr(self.beta,"ndim") and self.beta.ndim==1:
            self.beta = self.beta[...,xp.newaxis]
        ## number of realizations
        self.N_realz = N_realz
        
        ## precompute some parameters of the marginal prior which only rely on N_realz and the hyperprior
        ## effective sample size is nuprime = nu + N realizations
        ## nu is not bound to be an integer (but must be >0), so let's intentionally downweight the contribution from mu0
        self.nu = 1e-10
        self.nuprime = self.nu + self.N_realz
        ## alphaprime = alpha + N/2
        self.alphaprime = self.alpha + self.N_realz/2
        ## generalized t degrees of freedom = 2*alphaprime
        self.df = 2*self.alphaprime
        
        ## set remaining marginal prior parameters to None
        self.muprime = None
        self.sigmaprime = None
        self.betaprime = None
        
    def update(self,theta_spec):
        """
        Update the parameters of the conditional t distribution to be the hyperposterior
        parameters after taking into account the draws from the simulation.

        Parameters
        ----------
        theta_specs : array
            Spectra drawn from the conditional population prior, of shape (N_realz,N_frequencies).

        """
        
        ## compute remaining arguments from the theta_spec draws
        
        ## per-frequency mean of the draws
        Sf_mean = xp.mean(xp.log10(theta_spec),axis=1)
        
        ## sum of the spectral deviationes squared (sum((S-Smean)^2))
        Sf_sum_dev2 = xp.sum((xp.log10(theta_spec)-Sf_mean[:,None])**2,axis=1)
        # import pdb; pdb.set_trace()
        ## compute conditional prior parameters
        self.muprime = (self.nu*self.mu0 + self.N_realz*Sf_mean)/(self.nu + self.N_realz)
        betaprime = self.beta + 0.5*Sf_sum_dev2 + 0.5*((self.nu*self.N_realz)/(self.nu+self.N_realz))*(Sf_mean-self.mu0)**2
        self.sigmaprime = (betaprime*(self.nuprime + 1))/(self.alphaprime*self.nuprime)
        return
    
    def _logpdf(self,x):
        '''
        Get the marginal t logpdf
        
        Note: xsc.poch is the Pochhammer symbol, i.e. Gamma(z+m)/Gamma(z) where Gamma is the Gamma function.
              Hence, poch(df/2,1/2) = Gamma((df+1)/2)/Gamma(df/2).

        Parameters
        ----------
        x : array
            Array of values at which to compute the logpdf. Should at minimum be o shape (1,Nfreqs).

        Returns
        -------
        logpdf
            Natural log of the conditional location/scale t distribution at x.

        '''
        ln_coeff = xp.log(xsc.poch(0.5*self.df, 0.5)) - 0.5*(xp.log(self.df) + xp.log(xp.pi)) - xp.log(self.sigmaprime) - xp.log10(x)

        return ln_coeff + -0.5*(self.df+1)*xp.log1p((((xp.log10(x)-self.muprime)/self.sigmaprime)**2)/self.df)


class marginal_poisson_gamma(BaseDist):
    
    def __init__(self,rng,alpha=3,beta=0.001,N_obs=None,N_hat=None,cast=False):
        r"""
        
        Marginalized mixed poisson-gamma distribution. Equivalent to a negative binomial.
        
        Used to compute p(N_obs | N_hat(Lambda))
        
        It is possible to instantiate the object with known N_obs, alpha, and beta parameters,
        so that p(N_obs | N_hat) can be computed as a function of Nhat (which is what actually 
        changes during sampling).
        
        Only one of N_obs or N_hat can be specified, and you must specify one of them.
    
        Parameters
        ----------
        rng : Generator
            numpy or cupy Generator object.
        alpha : float (optional)
            Shape parameter for the Gamma prior on the Poisson rate. Should be O(1).
        beta : float (optional)
            Scale parameter for the Gamma prior on the Poisson rate. Should be 1e-3 or less.
        N_obs : float (optional)
            Number of observed resolved binaries
        N_hat : float (optional)
            Number of predicted resolved binaries.
        
        Returns
        -------
        None.

        """
        super().__init__(cast=cast)
        
        self.rng = rng
        self.alpha = alpha
        self.beta = beta
        
        if N_obs is not None and N_hat is not None:
            raise RuntimeError("Only one of N_obs or N_hat can be specified.")
        elif N_obs is None and N_hat is None:
            raise RuntimeError("Either N_obs or N_hat must be specified.")
        
        if N_obs is not None:
            self.N_obs = N_obs
            self._logpmf = self._logpmf_of_N_hat
            self._rvs = self._rvs_error
        
        if N_hat is not None:
            self.N_hat = N_hat
            self._logpmf = self._logpmf_of_N_obs

    
    def _rvs_error(self,size=1):
        raise NotImplementedError("No .rvs method is available for this configuration.")
    
    def _rvs(self,size=1):
        """
        
        Random variable sampling.
        
        Note that you need to set self.N_hat to do this.    
        
        Parameters
        ----------
        size : (int or tuple), optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        draws : (numpy or cupy array)
            Poisson-distributed samples

        """
        n = self.alpha + xp.sum(self.N_hat)
        betaprime = self.beta+xp.atleast_1d(self.N_hat).shape[0]
        p = (betaprime)/(1+betaprime)
        
        return self.rng.negative_binomial(n=n,p=p,size=size)
    
    def _logpmf_of_N_hat(self,N_hat):
        """
        log PMF of the Negative Binomial marginalized mixed Poisson-Gamma distribution.
        
        Computed as a function of the model realization of the Poisson process

        Parameters
        ----------
        N_hat : numpy or cupy array of ints
            Values of N_hat at which to compute the log PMF.

        Returns
        -------
        (numpy or cupy array)
            Values of the Negative Binomial marginalized mixed Poisson-Gamma log PMF

        """
        alphaprime = self.alpha + xp.sum(N_hat)
        betaprime = self.beta + xp.atleast_1d(N_hat).shape[0]
        
        p = (betaprime)/(1+betaprime)
        
        coeff = sc.gammaln(alphaprime+self.N_obs) - sc.gammaln(self.N_obs+1) - sc.gammaln(alphaprime)
        
        return coeff + alphaprime*xp.log(p) + sc.xlog1py(self.N_obs, -p)
    
    def _logpmf_of_N_obs(self,N_obs):
        """
        log PMF of the Negative Binomial marginalized mixed Poisson-Gamma distribution.
        
        Computed as a function of the observed realization of the Poisson process.

        Parameters
        ----------
        N_obs : numpy or cupy array of ints
            Values of N_obs at which to compute the log PMF.

        Returns
        -------
        (numpy or cupy array)
            Values of the Negative Binomial marginalized mixed Poisson-Gamma log PMF

        """
        alphaprime = self.alpha + xp.sum(self.N_hat)
        betaprime = self.beta + xp.atleast_1d(self.N_hat).shape[0]
        
        p = (betaprime)/(1+betaprime)
        
        coeff = sc.gammaln(alphaprime+N_obs) - sc.gammaln(N_obs+1) - sc.gammaln(alphaprime)
        
        return coeff + alphaprime*xp.log(p) + sc.xlog1py(N_obs, -p)