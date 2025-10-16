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
        else:
            print("GPU requested but no device is available. Defaulting to CPU.")
            import numpy as xp
    else:
        print("Running Pelargir population inference on CPU.")
        import numpy as xp
except:
    print("An error occurred in initializing GPU functionality. Defaulting to CPU.")
    import numpy as xp

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

    def sample_conditional(self,N=1):

        theta = xp.empty((len(self.conditional_dict.keys()),N))
        for i, key in enumerate(self.conditional_dict.keys()):
            theta[i,:] = self.conditional_dict[key].rvs(size=N)
        return theta
        



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
    
    ## NOTE, THIS IS A BASE 10 LOG NORMAL SO THAT WE CAN HAVE SIGMA IN DEX    
    def  array_lognormal_logpdf(self,theta_vec,mu_vec,sigma):
         """
         Array operation-based base 10 log-normal log PDF. 
         
         Note that theta_vec and mu_vec MUST include instrumental noise to avoid the
         likelihood dropping to -infinity for spectra with zero power in any bin.

         Parameters
         ----------
         theta_vec : array
             Proposed (model) spectrum.
         mu_vec : array
             Measured (data) spectrum.
         sigma : float or array
             Uncertainty of the log-normal as standard deviation, given in dex. If array, 
             designates uncertainty in each frequency bin, and must be of same shape as theta_vec and mu_vec.

         Returns
         -------
         logpdf
             Base 10 log-normal log likelihood.

         """
         norm = xp.log10(xp.e) - xp.log(theta_vec*sigma*xp.sqrt(2*xp.pi))
         return xp.sum(-((xp.log10(theta_vec) - xp.log10(mu_vec))**2)/(2*sigma**2) + norm)
    
    def vectorized_gaussian_logpdf(self, theta, mu_vec, cov_vec):
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
        log_determinants = xp.log(xp.prod(cov_vec, axis=1))
        deviations = theta - mu_vec
        inverses = 1 / cov_vec
        return -0.5 * (constant + log_determinants + xp.sum(deviations * inverses * deviations, axis=1))

class GB_Likelihood(Likelihood):
    '''
    GB analytic likelihood class
    '''

    def __init__(self,theta_true,cov,sigma_of_f=False):
        '''
        theta_true are the true simulated parameter values, of shape N_res x N_theta
        sigma is the N_theta x N_theta (N_theta x N_theta x N_f) or covariance matrix
        sigma_of_f (bool) : Whether the provided covariance is a function of frequency
        '''
        
        if not sigma_of_f:
            ## calculate the observed means with scatter from true vals
            self.mu_vec = xp.array([st.multivariate_normal.rvs(mean=theta_true[ii,:],
                                                               cov=cov,size=1) for ii in range(theta_true.shape[0])])
            self.cov = cov
            self.ln_prob = self.ln_prob_const_sigma
        else:
            self.mu_vec = st.multivariate_normal.rvs(mean=theta_true,cov=cov,size=1)
            self.cov_vec = cov
            self.ln_prob = self.ln_prob_sigma_of_f
            raise(NotImplementedError)
    
    # def ln_prob(self,theta):
    #     return -0.5*(theta - self.mu_vec).T @ xp.inv(self.cov) @ (theta - self.mu_vec)
    def ln_prob_const_sigma(self,theta):
        return self.const_covar_gaussian_logpdf(theta,self.mu_vec,self.cov)
    def ln_prob_sigma_of_f(self,theta):
        return self.vectorized_gaussian_logpdf(theta,self.mu_vec,self.cov_vec)

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

    def __init__(self,fg_data_psd,psd_cov,noise_data_psd,sigma_of_f=False):
        """
        

        Parameters
        ----------
        fg_data_psd : array
            Observed foreground PSD.
        psd_cov : array or float
            Standard deviation(s) of the log10-normal uncertainy on the total PSD.
        noise_data_psd : array
            LISA instrumental noise PSD.
        sigma_of_f : bool, optional
            Whether the PSD uncertainty is a function of frequency. The default is False.
            (Not yet implemented)

        Returns
        -------
        None.

        """
        
        
        if not sigma_of_f:
            ## calculate the observed means with scatter from true vals
            self.mu_vec = fg_data_psd + noise_data_psd #st.multivariate_normal.rvs(mean=spec_data,
                                    # cov=cov,size=1)
            self.noise_vec = noise_data_psd
            self.cov = psd_cov
            self.ln_prob = self.ln_prob_const_sigma
        else:
            self.mu_vec = fg_data_psd + noise_data_psd ## st.multivariate_normal.rvs(mean=theta_true,cov=cov,size=1)
            self.cov_vec = psd_cov
            self.ln_prob = self.ln_prob_sigma_of_f
            raise(NotImplementedError)
    

    def ln_prob_const_sigma(self,theta_spec):
        return self.array_gaussian_logpdf(theta_spec+self.noise_vec,self.mu_vec,self.cov)
    def ln_prob_sigma_of_f(self,theta_spec):
        return self.vectorized_gaussian_logpdf(theta_spec,self.mu_vec,self.cov_vec)

