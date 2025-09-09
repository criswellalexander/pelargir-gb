#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Sep  9 13:51:06 2025

@author: Alexander W. Criswell

Here we store all the priors and likelihoods, hierarchical or otherwise.

"""

import os
import cupy as cp
import numpy as np
xp = np
if xp == cp:
    os.environ['SCIPY_ARRAY_API'] = 1

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

        theta = np.empty((len(self.conditional_dict.keys()),N))
        for i, key in enumerate(self.conditional_dict.keys()):
            theta[i,:] = self.conditional_dict[key].rvs(N,random_state=self.rng)
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
    
    def __init__(self,rng):
        
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
    
    def condition(self,pop_theta):
        '''
        Condition the resolved GB parameters on the population parameters.
        
        Arguments:
        ---------------
        pop_theta (dict) : The population parameter chains as produced by Eryn. Keys are population parameter names.
        '''
        
        self.conditional_dict = {}
        ## condition mass prior on current pop values for the mean and standard deviation
        #scipy's truncnorm definition truncates by the number of sigmas, not at a value
        m_trunc_low = (self.m_min - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        m_trunc_high = (self.m_max - pop_theta['m_mu'][-1])/pop_theta['m_sigma'][-1]
        self.conditional_dict['m_1'] = self.prior_dict['m_1'](self.rng,
                                                              a=m_trunc_low,
                                                              b=m_trunc_high,
                                                              loc=pop_theta['m_mu'][-1],
                                                              scale=pop_theta['m_sigma'][-1])
        ## m1 and m2 should come from the same distribution; we can label-switch later if we need to assert m1>m2.
        self.conditional_dict['m_2'] = self.prior_dict['m_2'](self.rng,
                                                              a=m_trunc_low,
                                                              b=m_trunc_high,
                                                              loc=pop_theta['m_mu'][-1],
                                                              scale=pop_theta['m_sigma'][-1])
        ## ensure minimum distance is preserved; 
        ## scipy's truncnorm definition truncates by the number of sigmas, not at a value
        # d_trunc = (self.d_min - pop_theta['d_mu'][-1])/pop_theta['d_sigma'][-1] 
        # self.conditional_dict['d_L'] = self.prior_dict['d_L'](a=d_trunc,
        #                                                       b=np.inf,
        #                                                       loc=pop_theta['d_mu'][-1],
        #                                                       scale=pop_theta['d_sigma'][-1]
        #                                                       )
        self.conditional_dict['d_L'] = self.prior_dict['d_L'](self.rng,
                                                              a=pop_theta['d_gamma_a'],
                                                              scale = pop_theta['d_gamma_b']
                                                              )
        ## condition semimajor axis prior
        ## NOTE: I am defining this as p(a) ~ a^{alpha}
        ## adding 1 because scipy defines the power law as p(a) ~ a^{alpha - 1} for some reason
        self.conditional_dict['a'] = self.prior_dict['a'](self.rng,
                                                          pop_theta['a_alpha']+1,
                                                          loc=self.a_min, ## minimum
                                                          scale=self.a_max ## maximum
                                                         )
        return