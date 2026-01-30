#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Sep  9 13:51:06 2025

@author: Alexander W. Criswell

Here we store all the priors and likelihoods, hierarchical or otherwise.

"""

import os
try:
    if ('PELARGIR_GPU' in os.environ.keys()) and int(os.environ['PELARGIR_GPU']):
        import cupy as xp
        ## check for available devices
        if xp.cuda.is_available():
            print("GPU requested and available; running Pelargir population inference on GPU.")
            os.environ['SCIPY_ARRAY_API'] = '1'
            from cupyx.scipy import special as xsc
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

import distributions as st

class HierarchicalPrior:
    
    '''
    Generic class to handle the population-informed priors.
    
    Arguments
    -------------
    prior_dict (dict) : Dictionary of priors given as {'parameter_name':prior_function,...}
    conditional_map (func) : Function which returns the population-dependent priors given in prior_dict
                             conditioned on the current values of the population parameters given as pop_theta
    kwargs : Any additional values needed by conditional map. These will be added as attributes of the 
             HierarchicalPrior object, such that passing keyward_1=kwarg_1 will set self.keyword_1 = kwarg_1.
    
    '''
    
    def __init__(self,prior_dict,conditional_map,rng,**kwargs):
        ## prior dict of the form {parameter_name:prior_func}
        self.prior_dict = prior_dict
        ## conditional map is a function to condition the above priors on the current values of the population priors
        self.conditional_map = conditional_map
        ## set rng
        self.rng = rng
        ## set any additional kwargs needed by conditional_map function as object attributes
        for kw in kwargs:
            setattr(self,kw,kwargs[kw])
        
        return
    
    def condition(self,pop_theta):
        
        self.conditional_dict = self.conditional_map(pop_theta,self.prior_dict)
        
        return

    def sample_conditional(self,size=(1,)):
        '''
        Sample from the conditional prior.

        Parameters
        ----------
        size : tuple, optional
            Shape of samples to be returned. The default is (1,).

        Returns
        -------
        theta : array
            Samples from the conditional prior, of shape (Npar,*size).

        '''
        if type(size) is int:
            size = (size,)
        theta = xp.empty((len(self.conditional_dict.keys()),*size))
        for i, key in enumerate(self.conditional_dict.keys()):
            theta[i,...] = self.conditional_dict[key].rvs(size=size)
        return theta
    
    def conditional_logpdf(self,theta):
        '''
        Compute the conditional prior probability of a given draw.

        Parameters
        ----------
        theta : array
            N_par x ... array of samples.

        Returns
        -------
        logpdf : array
            Conditional logpdf at theta.

        '''
        logpdf = xp.empty_like(theta)
        for i, key in enumerate(self.conditional_dict.keys()):
            logpdf[...,i] = self.conditional_dict[key].logpdf(theta[...,i])
        
        return logpdf
        



class GalacticBinaryPrior(HierarchicalPrior):
    '''
    Population-informed GB prior. Assumes:
    - Gaussian-distributed masses
    - Power-law distributed orbital separations
    - Uniformly distributed inclinations (uniform in cos(i); not population-dependent)
    - (for now) broad Gaussian-distributed distances (TODO: update to an analytic Galaxy model)
    - (TODO: add sky localization parameters)
    - (TODO: add fdot)
    '''
    
    def __init__(self,rng,pop_params=['m_mu','m_sigma','rh_disk','r_bulge','q_bd','a_alpha']):
        
        ## set hyperparameters
        self.pop_params = pop_params
        
        self.prior_dict = {'m_1':st.truncnorm, ## in Msun
                           'm_2':st.truncnorm, ## in Msun
                           # 'd_L':st.truncnorm, ## in kpc
                           'd_L':st.gaussian_exponential_mixture, ## in kpc
                           'a':st.powerlaw ## in AU
        }
        
        ## set minimum allowed distance in kpc
        self.d_min = 1e-3 ## no GBs closer than the closest known star
        self.a_min = 1e-4 ## no binaries with a semimajor axis comparable to their radius
        self.a_max = 1e-2 ## no binaries outside of LISA's frequency range
        self.m_min = 0.17 ## lowest-mass observed white dwarf
        self.m_max = 1.44 ## no WDs with mass above the Chandrasekar limit
        
        ## set Galaxy model parameters
        self.galactic_center = 8 ## in kpc

        ## store rng
        self.rng = rng
        
        return
    
    def conditional_map(self,pop_theta_vec):
        """
        Helper function to align the parameter values and names if pop_theta is passed 
        to condition() as a list or array.

        Parameters
        ----------
        pop_theta_vec : iterable
            pop theta draw as an unlabelled vector.

        Returns
        -------
        pop_theta_dict : dict
            pop theta draw as a dictionary with parameter names as keys.

        """
        pop_theta_dict = {name:xp.array(val) for name,val in zip(self.pop_params,pop_theta_vec.tolist())}
        return pop_theta_dict
    
    def condition(self,pop_theta):
        '''
        Condition the resolved GB parameters on the population parameters.
        
        Arguments:
        ---------------
        pop_theta (dict) : The population parameter chains as produced by Eryn. Keys are population parameter names.
        '''
        
        if type(pop_theta) is not dict:
            pop_theta = self.conditional_map(pop_theta)
            
        self.conditional_dict = {}
        ## condition mass prior on current pop values for the mean and standard deviation
        #scipy's truncnorm definition truncates by the number of sigmas, not at a value
        # m_trunc_low = (self.m_min - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        # m_trunc_high = (self.m_max - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        self.conditional_dict['m_1'] = self.prior_dict['m_1'](self.rng,
                                                              a_min=self.m_min,
                                                              a_max=self.m_max,
                                                              loc=pop_theta['m_mu'],
                                                              scale=pop_theta['m_sigma'])
        ## m1 and m2 should come from the same distribution; we can label-switch later if we need to assert m1>m2.
        self.conditional_dict['m_2'] = self.prior_dict['m_2'](self.rng,
                                                              a_min=self.m_min,
                                                              a_max=self.m_max,
                                                              loc=pop_theta['m_mu'],
                                                              scale=pop_theta['m_sigma'])
        ## conditional distance prior with simple 1D Galaxy model
        self.conditional_dict['d_L'] = self.prior_dict['d_L'](self.rng,
                                                              x0=self.galactic_center,
                                                              bulge_scale=pop_theta['r_bulge'],
                                                              disk_scale=pop_theta['rh_disk'],
                                                              beta=pop_theta['q_bd']
                                                              )
        ## condition semimajor axis prior
        ## NOTE: I am defining this as p(a) ~ a^{alpha}
        self.conditional_dict['a'] = self.prior_dict['a'](self.rng,
                                                          pop_theta['a_alpha'],
                                                          loc=self.a_min, ## minimum
                                                          scale=self.a_max-self.a_min ## maximum
                                                         )
        return

class OldGalacticBinaryPrior(HierarchicalPrior):
    '''
    Population-informed GB prior. Assumes:
    - Gaussian-distributed masses
    - Power-law distributed orbital separations
    - Uniformly distributed inclinations (uniform in cos(i); not population-dependent)
    - (for now) broad Gaussian-distributed distances (TODO: update to an analytic Galaxy model)
    - (TODO: add sky localization parameters)
    - (TODO: add fdot)
    '''
    
    def __init__(self,rng,pop_params=['m_mu','m_sigma','d_gamma_a','d_gamma_b','a_alpha']):
        
        ## set hyperparameters
        self.pop_params = pop_params
        
        self.prior_dict = {'m_1':st.truncnorm, ## in Msun
                           'm_2':st.truncnorm, ## in Msun
                           # 'd_L':st.truncnorm, ## in kpc
                           'd_L':st.gamma, ## in kpc
                           'a':st.powerlaw ## in AU
        }
        
        ## set minimum allowed distance in kpc
        self.d_min = 1e-3 ## no GBs closer than the closest known star
        self.a_min = 1e-4 ## no binaries with a semimajor axis comparable to their radius
        self.a_max = 1e-2 ## no binaries outside of LISA's frequency range
        self.m_min = 0.17 ## lowest-mass observed white dwarf
        self.m_max = 1.44 ## no WDs with mass above the Chandrasekar limit

        ## store rng
        self.rng = rng
        
        return
    
    def conditional_map(self,pop_theta_vec):
        """
        Helper function to align the parameter values and names if pop_theta is passed 
        to condition() as a list or array.

        Parameters
        ----------
        pop_theta_vec : iterable
            pop theta draw as an unlabelled vector.

        Returns
        -------
        pop_theta_dict : dict
            pop theta draw as a dictionary with parameter names as keys.

        """
        pop_theta_dict = {name:xp.array(val) for name,val in zip(self.pop_params,pop_theta_vec.tolist())}
        return pop_theta_dict
    
    def condition(self,pop_theta):
        '''
        Condition the resolved GB parameters on the population parameters.
        
        Arguments:
        ---------------
        pop_theta (dict) : The population parameter chains as produced by Eryn. Keys are population parameter names.
        '''
        
        if type(pop_theta) is not dict:
            pop_theta = self.conditional_map(pop_theta)
            
        self.conditional_dict = {}
        ## condition mass prior on current pop values for the mean and standard deviation
        #scipy's truncnorm definition truncates by the number of sigmas, not at a value
        # m_trunc_low = (self.m_min - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        # m_trunc_high = (self.m_max - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        self.conditional_dict['m_1'] = self.prior_dict['m_1'](self.rng,
                                                              a_min=self.m_min,
                                                              a_max=self.m_max,
                                                              loc=pop_theta['m_mu'],
                                                              scale=pop_theta['m_sigma'])
        ## m1 and m2 should come from the same distribution; we can label-switch later if we need to assert m1>m2.
        self.conditional_dict['m_2'] = self.prior_dict['m_2'](self.rng,
                                                              a_min=self.m_min,
                                                              a_max=self.m_max,
                                                              loc=pop_theta['m_mu'],
                                                              scale=pop_theta['m_sigma'])
        self.conditional_dict['d_L'] = self.prior_dict['d_L'](self.rng,
                                                              a=pop_theta['d_gamma_a'],
                                                              scale = pop_theta['d_gamma_b']
                                                              )
        ## condition semimajor axis prior
        ## NOTE: I am defining this as p(a) ~ a^{alpha}
        self.conditional_dict['a'] = self.prior_dict['a'](self.rng,
                                                          pop_theta['a_alpha'],
                                                          loc=self.a_min, ## minimum
                                                          scale=self.a_max ## maximum
                                                         )
        return

class PopulationHyperPrior():
    '''
    Class for the actual hyperparameters.
    '''

    def __init__(self,rng,hyperprior_dict=None):

        '''.
        For now, set defaults but we can adjust later.
        '''

        if hyperprior_dict is None:

            hyperprior_dict = {'m_mu':st.uniform(rng,loc=0.2,scale=0.9),
                               'm_sigma':st.invgamma(rng,5),
                               'rh_disk':st.uniform(rng,loc=1,scale=9),
                               'r_bulge':st.uniform(rng,loc=0.05,scale=1.95),
                               'q_bd':st.uniform(rng,loc=0.01,scale=0.98),
                               'a_alpha':st.uniform(rng,loc=-0.5,scale=2.0)
                              }
        self.hyperprior_dict = hyperprior_dict
        return

    def sample(self,size=1):
        return {key:self.hyperprior_dict[key].rvs(size=size) for key in self.hyperprior_dict.keys()}
    
    def logpdf(self,theta):
        return xp.array([self.hyperprior_dict[key].logpdf(theta[i]) for i, key in enumerate(self.hyperprior_dict.keys())])
    
class OldPopulationHyperPrior():
    '''
    Class for the actual hyperparameters.
    '''

    def __init__(self,rng,hyperprior_dict=None):

        '''.
        For now, set defaults but we can adjust later.
        '''

        if hyperprior_dict is None:

            hyperprior_dict = {'m_mu':st.norm(rng,loc=0.6,scale=0.05),
                               'm_sigma':st.invgamma(rng,5),
                               'd_gamma_a':st.uniform(rng,loc=1,scale=9), ## these are pretty arbitrary
                               'd_gamma_b':st.uniform(rng,loc=1,scale=9), ## these are pretty arbitrary
                               'a_alpha':st.uniform(rng,0.25,1.0)
                              }
        self.hyperprior_dict = hyperprior_dict
        return

    def sample(self,size=1):
        return {key:self.hyperprior_dict[key].rvs(size=size) for key in self.hyperprior_dict.keys()}

# =============================================================================
# ABSTRACTED LIKELIHOODS
#   These are analytic likelihoods that we can sample against to abstract out 
#   the pieces of the analysis which would require running a Global Fit
# =============================================================================

## make some basic faux likelihoods for the GBs
class Likelihood():
    '''
    Base class for the analytic likelihood methods.
    '''

    def const_covar_gaussian_logpdf(self, theta, mu_vec, cov):
        """
        Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i.
        From Daniel W. on StackOverflow (https://stackoverflow.com/questions/48686934/numpy-vectorization-of-multivariate-normal)
        Args:
            X : shape (n, d)
                Data points
            means : shape (n, d)
                Mean vectors
            covariances : shape (n, d)
                Diagonal covariance matrices
        Returns:
            logpdfs : shape (n,)
                Log probabilities
        """
        _, d = theta.shape
        constant = d * xp.log(2 * xp.pi)
        log_determinants = xp.log(xp.prod(xp.diag(cov)))
        deviations = theta - mu_vec
        inverses = 1/xp.diag(cov)
        return -0.5 * (constant + log_determinants + xp.sum(deviations * inverses * deviations, axis=1))

    def array_gaussian_logpdf(self, theta_vec, mu_vec, sigma):
        """
        Array operation-based Gaussian log PDF, sans normalization.

        Parameters
        ----------
        theta_vec : array
            Proposed (model) spectrum.
        mu_vec : array
            Measured (data) spectrum.
        sigma : float or array
            Uncertainty of the Gaussian as standard deviation. If array, designates uncertainty in each
            frequency bin, and must be of same shape as theta_vec and mu_vec.

        Returns
        -------
        logpdf
            Unnormalized Gaussian log likelihood.

        """
        
        ## dropping this as it's just a normalizing constant
        # constant = 0.5 * xp.log(2 * xp.pi * sigma**2) 

        return - xp.sum((theta_vec - mu_vec)**2/(2*sigma**2))
    
    def vector_gaussian_logpdf(self, theta_vec, mu_vec, sigma):
        """
        Array operation-based Gaussian log PDF, sans normalization.

        Parameters
        ----------
        theta_vec : array
            Proposed (model) spectrum. Must be an array whose leading axis is frequency; trailing axes will be vectorized over.
        mu_vec : array
            Measured (data) spectrum. Will be cast to shape (Nf,1). Leading axis must be of same size as theta_vec.
        sigma : float or array
            Uncertainty of the Gaussian as standard deviation. If array, designates uncertainty in each
            frequency bin, and must be of same shape as mu_vec.

        Returns
        -------
        logpdf
            Unnormalized Gaussian log likelihood.

        """
        
        ## dropping this as it's just a normalizing constant
        # constant = 0.5 * xp.log(2 * xp.pi * sigma**2)
        
        ## force proper casting
        theta_vec = xp.atleast_1d(theta_vec)
        mu_vec = xp.atleast_1d(mu_vec)
        sigma = xp.atleast_1d(sigma)
        if theta_vec.ndim == 1:
            theta_vec = theta_vec[:,xp.newaxis]
        if mu_vec.ndim < theta_vec.ndim:
            for i in range(theta_vec.ndim-mu_vec.ndim):
                mu_vec = mu_vec[...,xp.newaxis]
        if sigma.ndim < theta_vec.ndim:
            for i in range(theta_vec.ndim-sigma.ndim):
                sigma = sigma[...,xp.newaxis]
                

        return - xp.sum((theta_vec - mu_vec)**2/(2*sigma**2),axis=0)
    
    def grid_lognormal_logpdf(self,theta_vec, mu_vec, sigma):
        """
        Array operation-based log normal log PDF, sans normalization.
    
        Note that this does not include a leading factor of 1/x, as we integrate over 
        a log-spaced grid and it would cancel out there.
    
        Parameters
        ----------
        theta_vec : array
            Grid of proposed (model) spectra. Must be an array whose leading axis is frequency.
        mu_vec : array
            Measured (data) spectrum. Will be cast to shape (Nf,1). Leading axis must be of same size as theta_vec.
        sigma : float or array
            Uncertainty of the Gaussian as standard deviation. If array, designates uncertainty in each
            frequency bin, and must be of same shape as mu_vec.
    
        Returns
        -------
        logpdf
            Unnormalized Gaussian log likelihood.
    
        """
        
        ## force proper casting
        theta_vec = xp.atleast_1d(theta_vec)
        mu_vec = xp.atleast_1d(mu_vec)
        sigma = xp.atleast_1d(sigma)
        if theta_vec.ndim == 1:
            theta_vec = theta_vec[:,xp.newaxis]
        if mu_vec.ndim < theta_vec.ndim:
            for i in range(theta_vec.ndim-mu_vec.ndim):
                mu_vec = mu_vec[...,xp.newaxis]
        if sigma.ndim < theta_vec.ndim:
            for i in range(theta_vec.ndim-sigma.ndim):
                sigma = sigma[...,xp.newaxis]
                
    
        return - (xp.log(theta_vec) - xp.log(mu_vec))**2/(2*sigma**2)
    
    ## NOTE, THIS IS A BASE 10 LOG NORMAL SO THAT WE CAN HAVE SIGMA IN DEX    
    # def  array_lognormal_logpdf(self,theta_vec,mu_vec,sigma):
    #      """
    #      Array operation-based base 10 log-normal log PDF. 
         
    #      Note that theta_vec and mu_vec MUST include instrumental noise to avoid the
    #      likelihood dropping to -infinity for spectra with zero power in any bin.

    #      Parameters
    #      ----------
    #      theta_vec : array
    #          Proposed (model) spectrum.
    #      mu_vec : array
    #          Measured (data) spectrum.
    #      sigma : float or array
    #          Uncertainty of the log-normal as standard deviation, given in dex. If array, 
    #          designates uncertainty in each frequency bin, and must be of same shape as theta_vec and mu_vec.

    #      Returns
    #      -------
    #      logpdf
    #          Base 10 log-normal log likelihood.

    #      """
    #      norm = xp.log10(xp.e) - xp.log(theta_vec*sigma*xp.sqrt(2*xp.pi))
    #      return xp.sum(-((xp.log10(theta_vec) - xp.log10(mu_vec))**2)/(2*sigma**2) + norm)
    
    # def vectorized_gaussian_logpdf(self, theta, mu_vec, cov_vec):
    #     """
    #     Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i.
    #     From Daniel W. on StackOverflow (https://stackoverflow.com/questions/48686934/numpy-vectorization-of-multivariate-normal)
    #     Args:
    #         X : shape (n, d)
    #             Data points
    #         means : shape (n, d)
    #             Mean vectors
    #         covariances : shape (n, d)
    #             Diagonal covariance matrices
    #     Returns:
    #         logpdfs : shape (n,)
    #             Log probabilities
    #     """
    #     _, d = theta.shape
    #     constant = d * xp.log(2 * xp.pi)
    #     log_determinants = xp.log(xp.prod(cov_vec, axis=1))
    #     deviations = theta - mu_vec
    #     inverses = 1 / cov_vec
    #     return -0.5 * (constant + log_determinants + xp.sum(deviations * inverses * deviations, axis=1))

class Old_Res_Astro_Likelihood(Likelihood):
    '''
    Resolved GB analytic likelihood class
    '''

    def __init__(self,rng,theta_true,cov,lims,sigma_of_f=False):
        '''
        theta_true are the true simulated parameter values, of shape N_res x N_theta
        sigma is the N_theta x N_theta (N_theta x N_theta x N_f) or covariance matrix
        sigma_of_f (bool) : Whether the provided covariance is a function of frequency 
        
        The GB_Likelihood object contains methods to estimate the population-informed
        posterior probability of the resolved binaries.
        
        Basic algorithm is to create a normal distribution object that can produce draws all parameters
        for each resolved binary in the simulated data, based on those binaries' true parameters
        + some scatter. At each iteration, take one draw of N_res x N_theta from the distribution,
        compute the (full vector) log likelihood of that draw via the distribution, then compute the
        (full vector) log prior of these samples via the population-informed prior. Sum these.
        
        '''
        
        ## create first multivariate norm object
        mv_obj = st.multivariate_normal(rng, theta_true, cov)
        
        ## calculate the observed means with scatter from true vals
        self.cov = cov
        self.lims = lims
        self.rng = rng
        self.mu_vec = mv_obj.rvs(size=1)
        self.initialize_bounded_draws(theta_true)
        self.mu_vec = self.bounded_draw_from_likelihood(draw=self.mu_vec)
        
        
        ## now create the dist object to draw from
        ## TODO -- might need to check mu_vec vs. cov shape for broadcasting
        self.analytic_likelihood = st.multivariate_normal(rng,self.mu_vec,self.cov)
        
        self.ln_prob = self.ln_prob_analytic
        
        return
    
    def initialize_bounded_draws(self,theta_true):
        
        # self.previous_draw = xp.empty_like(self.mu_vec)
        # for ii in range(self.mu_vec.shape[-1]):
        #     self.previous_draw[:,ii] = st.uniform(rng,loc=self.lims[ii,0],
        #                                           scale=self.lims[ii,1]-self.lims[ii,0]).rvs(self.mu_vec.shape[0])
        self.previous_draw = theta_true
        self.previous_draw.shape = (1,*self.previous_draw.shape)

        return
    
    def bounded_draw_from_likelihood(self,draw=None):
        
        
        ## draw from a multivariate normal distribution
        ## if the draw is outside of bounds, keep the previous value
        ## initialize the first set of previous values as uniform on the bounds
        if draw is None:
            draw = self.analytic_likelihood.rvs()
             
        new_shape = tuple([1 for i in range(draw.ndim-1)]+[self.lims.shape[0]])
        lower = self.lims[:,0].reshape(new_shape)
        upper = self.lims[:,1].reshape(new_shape)
        
        filt = xp.invert(xp.prod((draw >= lower)*(draw <= upper)*xp.isfinite(draw),axis=-1).astype(dtype=bool))
        if xp.any(xp.isinf(self.previous_draw)):
            import pdb; pdb.set_trace()
        if xp.any(filt):
            try:
                draw[...,filt,:] = self.previous_draw[...,filt,:]
            except:
                self.previous_draw.shape = draw.shape
                draw[...,filt,:] = self.previous_draw[...,filt,:]
        if xp.any(xp.isinf(draw)):
            import pdb; pdb.set_trace()

        # self.previous_draw = draw

        return draw
        
        

    
    
    def ln_prob_analytic(self,prior_obj):
        '''
        

        Parameters
        ----------
        prior_obj : GalacticBinaryPrior object
            Population-informed prior.

        Returns
        -------
        log_posterior : float
            Total combined (log) likelihood and prior for a draw from the 
            abstracted analytic resolved binary likelihood.

        '''
        
        draw = self.bounded_draw_from_likelihood()
        log_like = xp.sum(self.analytic_likelihood.logpdf(draw))
        ## orbital separation back to linear space
        # import pdb; pdb.set_trace()
        draw[...,:,-1] = 10**draw[...,:,-1]
        log_prior = xp.sum(prior_obj.conditional_logpdf(draw))
        # import pdb; pdb.set_trace()
        return log_like + log_prior

class Res_Astro_Likelihood(Likelihood):
    '''
    Resolved GB analytic likelihood class
    '''

    def __init__(self,rng,theta_true,sigma_of_f=False):
        '''
        theta_true are the true simulated parameter values, of shape N_res x N_theta
        sigma is the N_theta x N_theta (N_theta x N_theta x N_f) or covariance matrix
        sigma_of_f (bool) : Whether the provided covariance is a function of frequency 
        
        The GB_Likelihood object contains methods to estimate the population-informed
        posterior probability of the resolved binaries.
        
        Basic algorithm is to create a normal distribution object that can produce draws all parameters
        for each resolved binary in the simulated data, based on those binaries' true parameters
        + some scatter. At each iteration, take one draw of N_res x N_theta from the distribution,
        compute the (full vector) log likelihood of that draw via the distribution, then compute the
        (full vector) log prior of these samples via the population-informed prior. Sum these.
        
        '''
        
        ## create first multivariate norm object
        mv_obj = st.multivariate_normal(rng, theta_true, cov)
        
        ## calculate the observed means with scatter from true vals
        self.cov = cov
        self.lims = lims
        self.rng = rng
        self.mu_vec = mv_obj.rvs(size=1)
        self.initialize_bounded_draws(theta_true)
        self.mu_vec = self.bounded_draw_from_likelihood(draw=self.mu_vec)
        
        
        ## now create the dist object to draw from
        ## TODO -- might need to check mu_vec vs. cov shape for broadcasting
        self.analytic_likelihood = st.multivariate_normal(rng,self.mu_vec,self.cov)
        
        self.ln_prob = self.ln_prob_analytic
        
        return
    
    def initialize_bounded_draws(self,theta_true):
        
        # self.previous_draw = xp.empty_like(self.mu_vec)
        # for ii in range(self.mu_vec.shape[-1]):
        #     self.previous_draw[:,ii] = st.uniform(rng,loc=self.lims[ii,0],
        #                                           scale=self.lims[ii,1]-self.lims[ii,0]).rvs(self.mu_vec.shape[0])
        self.previous_draw = theta_true
        self.previous_draw.shape = (1,*self.previous_draw.shape)

        return
    
    def bounded_draw_from_likelihood(self,draw=None):
        
        
        ## draw from a multivariate normal distribution
        ## if the draw is outside of bounds, keep the previous value
        ## initialize the first set of previous values as uniform on the bounds
        if draw is None:
            draw = self.analytic_likelihood.rvs()
             
        new_shape = tuple([1 for i in range(draw.ndim-1)]+[self.lims.shape[0]])
        lower = self.lims[:,0].reshape(new_shape)
        upper = self.lims[:,1].reshape(new_shape)
        
        filt = xp.invert(xp.prod((draw >= lower)*(draw <= upper)*xp.isfinite(draw),axis=-1).astype(dtype=bool))
        if xp.any(xp.isinf(self.previous_draw)):
            import pdb; pdb.set_trace()
        if xp.any(filt):
            try:
                draw[...,filt,:] = self.previous_draw[...,filt,:]
            except:
                self.previous_draw.shape = draw.shape
                draw[...,filt,:] = self.previous_draw[...,filt,:]
        if xp.any(xp.isinf(draw)):
            import pdb; pdb.set_trace()

        # self.previous_draw = draw

        return draw
        
        

    
    
    def ln_prob_analytic(self,prior_obj):
        '''
        

        Parameters
        ----------
        prior_obj : GalacticBinaryPrior object
            Population-informed prior.

        Returns
        -------
        log_posterior : float
            Total combined (log) likelihood and prior for a draw from the 
            abstracted analytic resolved binary likelihood.

        '''
        
        draw = self.bounded_draw_from_likelihood()
        log_like = xp.sum(self.analytic_likelihood.logpdf(draw))
        ## orbital separation back to linear space
        # import pdb; pdb.set_trace()
        draw[...,:,-1] = 10**draw[...,:,-1]
        log_prior = xp.sum(prior_obj.conditional_logpdf(draw))
        # import pdb; pdb.set_trace()
        return log_like + log_prior

class Nres_Likelihood(Likelihood):
    '''
    N_res Poisson likelihood
    '''

    def __init__(self,N_res_obs):
        '''
        N_res_obs (Number of resolved binaries)
        
        '''
        
        ## note: we arbitrarily initialize an rng that won't be used here
        ## b/c we only use the marginal poisson-gamma pmf
        ## but need to provide an rng to initialize the object
        rng = xp.random.default_rng(1)

        self.N_res_obs = N_res_obs
        # self.base_dist = st.poisson(rng,lam=self.N_res_obs)
        self.base_dist = st.marginal_poisson_gamma(rng,N_obs=self.N_res_obs)
        self.ln_prob = self.ln_conditional_poisson_gamma

    # def ln_conditional_Poisson(self,N_res_theta):

    #     return -self.N_res_obs + N_res_theta*xp.log(N_res_obs) - xp.log(factorial(N_res_theta))
    
    ## (new) mixed poisson-gamma dist
    def ln_conditional_poisson_gamma(self,N_res_theta):
        """
        Conditional log marginal mixed Poisson-Gamma PMF.

        Parameters
        ----------
        N_res_theta : int
            Observed number of resolved binaries.

        Returns
        -------
        logPMF
            Marginal Poisson-Gamma likelihood of observing N_res_obs resolved GBs,
            conditioned on the population via the single-draw estimator N_res_theta.

        """

        return self.base_dist.logpmf(N_res_theta)
    
    ## (old) poisson dist
    def ln_conditional_Poisson(self,N_res_theta):
        """
        Conditional log Poisson PMF

        Parameters
        ----------
        N_res_theta : int
            Observed number of resolved binaries.

        Returns
        -------
        logPMF
            Poisson likelihood conditioned on the population of observing N_res_theta resolved GBs.

        """
        
        return self.base_dist.logpmf(N_res_theta)
    
    

class FG_Likelihood(Likelihood):
    '''
    Foreground analytic likelihood class
    '''

    def __init__(self,fg_data_psd,psd_cov,noise_data_psd,
                 Nreal=5,Ngrid=1000,
                 hp_mu0=None,hp_alpha=1,hp_beta=None):
        """
        

        Parameters
        ----------
        fg_data_psd : array
            Observed foreground PSD.
        psd_cov : array or float
            Standard deviation(s) of the log10-normal uncertainy on the total PSD.
        noise_data_psd : array
            LISA instrumental noise PSD.
        Nreal : int, optional
            The number of realizations to use to estimate the marginal Poisson uncertainty. Minimum 2.
        Ngrid : float, optional
            Number of points to use to numerically compute the convolution of the Poisson-marginalized
            population-informed conditional spectral prior with the PSD likelihood. Default 1000.
            The grid will be on [-5 sigma, + 5 sigma] in each frequency bin.
        hp_mu0 : float or array, optional
            Value(s) of the hyperprior mu_0 parameter, i.e. the mean of the Gaussian prior on the foreground PSD mean.
            Used for analytic marginalization over the Poisson variance in the
            foreground spectrum at a given point in the population parameter space. In general, should be <= than the expected PSD
            to avoid biasing the marginal prior. Default is 1e-45. If passed as an array, should be of the same shape as fg_data_psd,
            and provide a prior mean for each frequency bin.
        hp_alpha : float or array, optional
            Value(s) of the hyperprior alpha parameter, i.e. the shape of the Gamma prior on the variance of the Gaussian prior
            on the foreground PSD. Used for analytic marginalization over the Poisson variance in the
            foreground spectrum at a given point in the population parameter space. In general, the likelihood is robust to choice of
            hp_alpha, provided it is roughly of order the typical foreground PSD value (within ~10 orders of magnitude).
            Default is 1e-40. If passed as an array, should be of the same shape as fg_data_psd, and provide a value of alpha for each frequency bin.
        hp_beta : float or array, optional
            Value(s) of the hyperprior beta parameter, i.e. the scale of the Gamma prior on the variance of the Gaussian prior
            on the foreground PSD. Used for analytic marginalization over the Poisson variance in the
            foreground spectrum at a given point in the population parameter space. In general, hp_beta informs the marginal prior more the larger
            it is. If hp_beta >> the typical PSD value, the set of simulation draws is essentially ignored in favor of a large prior. However,
            if hp_beta << the typical PSD value, the marginal prior will become inverted and disfavour the region where it should be peaked.
            Fore safety reasons, hp_beta should be within ~2 orders of magnitude of the minimum expected PSD value.
            The default (None) sets hp_beta to an array 1 order of magnitude below the noise PSD in each frequency bin.
            If passed as an array, should be of the same shape as fg_data_psd, and provide a value of beta for each frequency bin.
        
        Returns
        -------
        None.

        """
        
        ## note: we arbitrarily initialize an rng that won't be used here
        ## b/c we only use the marginal t pdf
        ## but need to provide an rng to initialize the st.t object
        rng = xp.random.default_rng(1)
        
        
        ## calculate the observed means with scatter from true vals
        self.noise_psd = noise_data_psd
        self.mu_vec = fg_data_psd + noise_data_psd #st.multivariate_normal.rvs(mean=spec_data,
                                # cov=cov,size=1)
        self.noise_vec = noise_data_psd
        self.cov = psd_cov
        
        ## number of realizations
        self.Nreal = Nreal
        
        ## grid from -5sigma to + 5sigma with Ngrid points for each frequency bins
        ## self.cgrid is of shape (Nfreqs,Ngrid)
        # self.cgrid = xp.linspace(self.mu_vec - 5*self.cov, self.mu_vec + 5*self.cov, Ngrid).T
        
        ## log-spaced grid across region of interest
        self.cgrid = xp.logspace(-42,-33,Ngrid)[xp.newaxis,:]
        
        ## compute data log likelihood for the grid
        ## grid is constructed such that this is the same for every frequency bin
        ## (b/c the grid is rectangular in probability-frequency space, not amplitude-frequency space)
        ## so this is a 2D array of shape (1,Ngrid)
        # import pdb; pdb.set_trace()
        # self.ln_pgrid = self.vector_gaussian_logpdf(self.cgrid[0,:][xp.newaxis,:],self.mu_vec[0],xp.atleast_1d(self.cov)[0])[xp.newaxis,:]
        
        self.ln_pgrid = self.grid_lognormal_logpdf(self.cgrid,self.mu_vec, xp.atleast_1d(self.cov))
        
        ## chunk of code to build the arguments for the t-distribution as much as possible a priori
        
        ## hyperprior parameters
        
        ## prior mean as a function of frequency
        if hp_mu0 is None:
            ## we will likely to have to approach this slightly differently in the non-toy-model case
            self.spec_mu0 = -40 #noise_data_psd
        else:
            self.spec_mu0 = xp.atleast_1d(hp_mu0) ## arbitrary but should be << typical PSD value
        
        ## parameters of gamma prior on variance
        self.spec_alpha = xp.atleast_1d(hp_alpha) ## arbitrary but should of order the typical PSD value
        if hp_beta is None:
            ## we will likely to have to approach this slightly differently in the non-toy-model case
            self.spec_beta = 0.15 #*noise_data_psd
        else:
            self.spec_beta = xp.atleast_1d(hp_beta) ## should be within 2 orders of magnitude of the minimum PSD value
        
        
        ## initialize the marginal Normal-inverse-Gamma as a conditional t distribution
        self.conditional_t = st.vector_marginal_logt(rng,self.spec_mu0,self.Nreal,
                                           alpha=self.spec_alpha,beta=self.spec_beta)
        
        ## prior parameters
        
        # ## effective sample size is nuprime = nu + N realizations, nu=1 (least weight to prior)
        # self.spec_nuprime = 1 + self.Nreal
        # ## alphaprime = alpha + N/2
        # self.spec_alphaprime = self.spec_alpha + self.Nreal/2
        # ## generalized t degrees of freedom = 2*alphaprime
        # self.spec_dof = 2*self.spec_alphaprime
        

        ## then assign a function for taking in the theta_spec draws, computing the remaining terms,
        ## calling the student t logpdf, and convolve with self.lNgrid
        
        self.ln_prob = self.ln_prob_conditional_like
    

    def ln_prob_spec_like(self,theta_spec):
        return self.array_gaussian_logpdf(theta_spec+self.noise_vec,self.mu_vec,self.cov)

    def ln_prob_conditional_like(self,theta_spec):
        
        ## check that theta_spec is of the right shape
        if theta_spec.shape[1] != self.Nreal:
            import pdb; pdb.set_trace()
        
        ## update the marginal prior with the theta_spec draws
        self.conditional_t.update(theta_spec+self.noise_psd[:,None,None])
        
        
        # ## per-frequency mean of the draws
        # Sf_mean = xp.mean(theta_spec,axis=0) ## CHECK AXIS
        
        # ## sum of the spectral deviationes squared (sum((S-Smean)^2))
        # Sf_sum_dev2 = xp.sum((theta_spec-Sf_mean[:,None])**2,axis=0)
        
        # ## compute conditional prior parameters
        # muprime = (self.spec_mu0 + self.Nreal*Sf_mean)/(1 + self.Nreal)
        # betaprime = self.spec_beta + 0.5*Sf_sum_dev2 + 0.5*(self.Nreal/(1+self.Nreal))*(Sf_mean-self.mu0)**2
        # sigmaprime = (betaprime*(self.nuprime + 1))/(self.alphaprime*self.nuprime)

        # ## make the st.general_t object
        # conditional_t_prior = st.t(self.rng,mu=muprime,sigma=sigmaprime,dof=self.spec_dof)

        ## call the generalized t logpdf
        ln_conditional_prior = self.conditional_t.logpdf(self.cgrid) ## shape (Nfreqs,Ngrid)
        
        ## convolve over grid and sum conditional loglike over frequencies
        ## the grid is log-spaced, so there should be a factor of the un-logged grid amplitude here
        ## but we implicitly cancel this out with the leadng 1/x missing from ln_pgrid
        loglike = xp.sum(xsc.logsumexp(ln_conditional_prior+self.ln_pgrid,axis=1))
        
        return loglike