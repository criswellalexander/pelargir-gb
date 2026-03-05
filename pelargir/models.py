'''

File to house the population model classes.

'''
## numpy/cupy switch
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
import legwork as lw
import astropy.units as u
from tqdm import tqdm

from utils import get_amp_freq, to_numpy, lisa_noise_psd
from thresholding import SNR_Threshold
from inference import PopulationHyperPrior, GalacticBinaryPrior, FG_Likelihood, Nres_Likelihood, Res_Astro_Likelihood

class PopModel():
    '''
    Class to house the overall population model.
    '''

    def __init__(self,Ntot,rng,hyperprior='default',
                 fbins='default',Tobs=4*u.yr,Nsamp=1,
                 Nreal=1,block_after=4,
                 thresholding="SNR",threshold_val=7.0,
                 res_rng=None,res_scatter=True,res_dynamic_scatter=True):
        """
        GB population model. Houses the mechanics of drawning GB populations from conditional
        population priors and computing the likelihood of the data given the draw(s).

        Parameters
        ----------
        Ntot : int
            Total number of binaries in the Galaxy. Fixed for now.
        rng : Generator object
            RNG. xp.random.default_rng or other Generator.
        hyperprior : {dict, string, HyperPrior}, optional
            The hyperprior to use. Can be a dict of {parameter:dist}, 'default', which initializes a
            new HyperPrior instance, or a HyperPrior instance. The default is 'default'.
        fbins : {str, array}, optional
            Frequency bins, as an array. The default is 'default' (bin widths of 1e-5 Hz, on [1e-4,5e-3]).
        Tobs : astropy.Quantity, optional
            LISA observation duration. The default is 4*u.yr.
        Nsamp : int, optional
            Number of times to run the population model. The default is 1.
        Nreal : int, optional
            Number of realizations to draw per call to the population model. The default is 1.
            If Nreal > 1, calls to the model will return arrays with trailing dimension Nreal.
        block_after : int, optional
            Number of frequency bins to perform serial operations on before switching to blocked operations.
        thresholding : str, optional
            How to threshold between resolved/unresolved binaries. Only "SNR" is implemented for now.
            The default is "SNR".
        threshold_val : float, optional
            Threshold SNR dividing resolved and unresolved binaries. The default is 7.0.
        res_rng : Generator object
            RNG used for the abstract resolved binary likelihood. 
            xp.random.default_rng or other Generator. Default None (uses input of rng).
        res_scatter: bool
            For the resolved GB likelihood.
            Whether to draw, at initialization a new parameter vector from a Gaussian centered at theta_true to 
            simulate sampling over the parameter likelihood. Default True.
        res_dynamic_scatter : bool
            For the resolved GB likelihood.
            Whether to apply scatter at every likelihood call, or just at initialization. Default True.
        
        Returns
        -------
        None.

        """
        
        if type(hyperprior) is str and hyperprior == 'default':
            self.hyperprior = PopulationHyperPrior(rng)
        elif type(hyperprior) is PopulationHyperPrior:
            self.hyperprior = hyperprior
        elif type(hyperprior) is dict:
            self.hyperprior = PopulationHyperPrior(rng,hyperprior_dict=hyperprior)
        else:
            raise TypeError("Unknown option for 'hyperprior' given. \
                             Can be 'default' or an instantiated PopulationHyperPrior object.")
        
        self.hpar_names = [key for key in self.hyperprior.hyperprior_dict.keys()]
        
        self.Npar = len(self.hpar_names)
        
        self.N = int(Ntot)
        
        self.Nreal = Nreal
        
        self.gbprior = GalacticBinaryPrior(rng,Nreal=self.Nreal)
        
        if res_rng is None:
            self.res_rng = rng
        else:
            self.res_rng = res_rng
        self.scatter = res_scatter
        self.dynamic_scatter = res_dynamic_scatter

        if type(fbins) is str and fbins == 'default':
            self.bin_width = 1e-5
            self.dur_eff = 1/self.bin_width
            self.fbins = xp.arange(1e-4,5e-3,self.bin_width)
        else:
            self.fbins = fbins
            self.bin_width = self.fbins[1] - self.fbins[0]
            self.dur_eff = 1/self.bin_width
        
        self.fmax = self.fbins.max()

        self.Tobs = Tobs.to(u.s).value
        
        ## get the approximate (and, for now, fixed) LISA instrumental noise PSD
        self.approx_lisa_psd = xp.asarray(lisa_noise_psd(self.fbins))
        
        ## need some casting here to make this numpy/cupy agnostic
        self.approx_lisa_rx = xp.asarray(lw.psd.approximate_response_function(to_numpy(self.fbins)*u.Hz,
                                                                              19.09*u.mHz).value)

        self.Nsamp = Nsamp
        
        if (thresholding == "SNR") or (thresholding == "snr"):
            self.thresher = SNR_Threshold(self.fbins, self.approx_lisa_psd, self.approx_lisa_rx, block_after=block_after)
            self.thresh_val = threshold_val
        else:
            raise NotImplementedError("Only SNR thresholding is currently supported.")
        
        ## GPU/CPU agnostic
        gpu_flag = ('PELARGIR_GPU' in os.environ.keys()) and int(os.environ['PELARGIR_GPU'])
        eryn_flag = ('PELARGIR_ERYN' in os.environ.keys()) and int(os.environ['PELARGIR_ERYN'])
        if gpu_flag and eryn_flag:
            self.cast = xp.asnumpy
            self.invcast = xp.asarray
        else:
            self.cast = xp.asarray
            self.invcast = xp.asarray
        
        return

    def construct_likelihood(self,data,theta_lims='default',**fg_kwargs):
        '''
        Wrapper to build all the likelihoods
        
        Arguments
        ------------------
        data : dict
            Data dictionary, consisting of {'fg':foreground spectrum,
                                            'fg_sigma':spectrum uncertainty,
                                            'Nres':number of resolved binaries,
                                            'noise':noise spectrum (optional),
                                            'gb_thetas':array of shape (Nres,Ntheta) with simulated resGB params,
                                             }
        **fg_kwargs : kwargs, optional
            Keyword arguments to pass to construct_fg_likelihood (hp_mu0, hp_alpha, hp_beta)
        '''

        fg_data = data['fg']
        fg_sigma =data['fg_sigma']
        N_res_data = data['Nres']
        gb_thetas = data['gb_thetas']
        
        if 'noise' in data.keys():
            noise = data['noise']
        else:
            noise='default'
        
        self.construct_fg_likelihood(fg_data,fg_sigma,noise_psd=noise,**fg_kwargs)
        self.construct_Nres_likelihood(N_res_data)
        self.construct_res_astro_likelihood(self.res_rng,gb_thetas,
                                            scatter=self.scatter,dynamic_scatter=self.dynamic_scatter)

        return
    
    def construct_fg_likelihood(self,fg_psd,psd_sigma,noise_psd='default',**hp_kwargs):
        """
        Method to attach the foreground likelihood to the PopModel,

        Parameters
        ----------
        fg_psd : array
            Data foreground PSD.
        
        psd_sigma : float or array
            Standard deviation of the log-normal uncertainty on the joint noise+foreground PSD.
            Currently can only be a float. IMPLEMENT IN FUTURE: per-frequency uncertainty as array arg.
        
        noise_psd : str or array, optional
            LISA instrumental noise PSD. Default ('default') will use the simple Robson+19 approximate LISA PSD.
            Otherwise it should be an array of noise PSD values at the same frequencies as fg_psd.
        **hp_kwargs : kwargs, optional
            Hyperprior keyword arguments for FG_Likelihood (hp_mu0, hp_alpha, hp_beta).
        
        Returns
        -------
        None.

        """
        if (type(noise_psd) is str) and (noise_psd == 'default'):
            noise_psd = self.approx_lisa_psd
        

        self.fg_like = FG_Likelihood(fg_psd,psd_sigma,noise_psd,Nreal=self.Nreal,**hp_kwargs)
        self.fg_ln_prob = self.fg_like.ln_prob

        return

    def construct_Nres_likelihood(self,N_res_obs):
        '''
        Method to attach the Poisson likelihood for the number of resolved binaries to the PopModel
        '''
        self.Nres_like = Nres_Likelihood(N_res_obs)
        self.N_res_ln_prob = self.Nres_like.ln_prob

        return
    
    def construct_res_astro_likelihood(self,rng,theta_true,scatter=True,dynamic_scatter=True,
                                       override_dims=False,**kwargs):
        """
        Method to attach the abstracted likelihood on the resolved binary astro parameters.

        Parameters
        ----------
        rng : Generator
            DESCRIPTION.
        theta_true : array of shape (Nres,Ntheta)
            True values of the GB parameters.
        scatter : bool, optional
            Whether to draw a new parameter vector from a Gaussian centered at theta_true to 
            simulate sampling over the parameter likelihood. Default True.
        dynamic_scatter : bool, optional
            Whether to apply scatter at every likelihood call, or just at initialization. Default True.
        override_dims : bool, opional
            Whether to force override of the error which is raised if Nres < N_theta
        **kwargs : kwargs
            Keyword arguments for utils.scatter_thetas()

        Raises
        ------
        ValueError
            If the number of resolved binaries is < the number of parameters, an error will be raised.
            This is to avoid accidental passing of an array with shape (N_theta,N_res).
            This can be overridden by setting override_dims=True.

        Returns
        -------
        None.

        """
        
        if not override_dims and (theta_true.squeeze().shape[0] < theta_true.squeeze().shape[1]):
            raise ValueError("theta_draw must be of shape (Nres,N_theta) but array of shape {} was passed.\
                              If you want to have more parameters than binaries, set override_dims=True.".format(theta_true.shape))
        
        self.res_astro_like = Res_Astro_Likelihood(rng,theta_true,scatter=scatter,dynamic_scatter=dynamic_scatter,**kwargs)
        self.res_astro_ln_prob = self.res_astro_like.ln_prob
        
        return
    
    def fg_N_ln_prob(self,pop_theta,return_spec=False,branch_supps=None,inds=None,branch_name='model_0'):
        """
        Function to get the model probability conditioned on only 
        the per-bin foreground amplitude and the total number of resolved binaries.
        
        We draw N_realizations from the model and analytically marginalize over the uncertainty
        in the (unknown) underlying Poisson processes.

        Eventually we can extend this to per-bin N_res

        Parameters
        ----------
        pop_theta : array
            Input state of the population parameters.
        return_spec : bool, optional
            Whether to return the foreground spectrum, along with frequencies and number of resolved binaries. The default is False.
        branch_supps : eryn.state.BranchSupplemental, optional
            Branch supplemental, for carrying the foreground spectrum and N_res as Eryn latent variables. The default is None.
        inds : tuple, optional
            Indices where to update branch_supplemental. Eryn handles this automagically, but it's sometimes useful to pass these manually.
            Default is None (all provided dims save for the last)
        
        Returns
        -------
        loglike : array or float
            Log likelihood at proposed point.
        astro_info : list, optional
            List of latent astrophysical information, given as [frequencies, foreground PSD, N_res].
            Only returned if return_spec is set to True.

        """
        
        
        # ## unpack data
        # N_res_obs = data['N_res']
        # fg_obs = data['fg']

        ## call the population model
        fbins, fg_psd, N_res = self.run_model(pop_theta)

        ## call the fg likelihood
        ln_p_fg = self.fg_ln_prob(fg_psd)

        ln_p_Nres = self.N_res_ln_prob(N_res)
        
        if branch_supps is not None:
            if type(branch_supps) is dict:
                branch_supps = branch_supps[branch_name]
            if type(inds) is dict:
                inds = inds[branch_name]
            # import pdb; pdb.set_trace()
            if inds is not None:
                branch_supps.holder['spectra'][*inds] = to_numpy(fg_psd)
                branch_supps.holder['Nres'][*inds] = to_numpy(N_res)
            else:
                branch_supps[0]['spectra'][...] = to_numpy(fg_psd)
                branch_supps[0]['Nres'][...] = to_numpy(N_res)
        
        if return_spec:
            return self.cast(ln_p_fg + ln_p_Nres), [to_numpy(fbins[1:]),to_numpy(fg_psd[1:]),to_numpy(N_res)]
        else:
            return self.cast(ln_p_fg + ln_p_Nres)
    
    def ln_prob(self,pop_theta,return_spec=False,branch_supps=None,inds=None,branch_name='model_0'):
        """
        Function to get the model probability conditioned on 
        the per-bin foreground amplitude, the total number of resolved binaries, and the
        astrophyical parameters of the resolved binaries.
        
        We draw N_realizations from the model and analytically marginalize over the uncertainty
        in the (unknown) underlying Poisson processes.

        Eventually we can extend this to per-bin N_res

        Parameters
        ----------
        pop_theta : array
            Input state of the population parameters.
        return_spec : bool, optional
            Whether to return the foreground spectrum, along with frequencies and number of resolved binaries. The default is False.
        branch_supps : eryn.state.BranchSupplemental, optional
            Branch supplemental, for carrying the foreground spectrum and N_res as Eryn latent variables. The default is None.
        inds : tuple, optional
            Indices where to update branch_supplemental. Eryn handles this automagically, but it's sometimes useful to pass these manually.
            Default is None (all provided dims save for the last)
        
        Returns
        -------
        loglike : array or float
            Log likelihood at proposed point.
        astro_info : list, optional
            List of latent astrophysical information, given as [frequencies, foreground PSD, N_res].
            Only returned if return_spec is set to True.

        """
        
        
        # ## unpack data
        # N_res_obs = data['N_res']
        # fg_obs = data['fg']
        ## call the population model
        fbins, fg_psd, N_res = self.run_model(pop_theta)

        ## call the fg likelihood
        ln_p_fg = self.fg_ln_prob(fg_psd)
        
        ## call the Poisson term likelihood
        ln_p_Nres = self.N_res_ln_prob(N_res)
        
        ## call the resolved binary likelihood
        ln_p_res_astro = self.res_astro_ln_prob(self.gbprior)
        
        ln_p_tot = ln_p_fg + ln_p_Nres + ln_p_res_astro
        
        # import pdb; pdb.set_trace()
        
        if branch_supps is not None:
            if type(branch_supps) is dict:
                branch_supps = branch_supps[branch_name]
            if type(inds) is dict:
                inds = inds[branch_name]
            if inds is not None:
                branch_supps.holder['spectra'][inds] = to_numpy(xp.moveaxis(fg_psd,-1,0)[...,xp.newaxis])
                branch_supps.holder['Nres'][inds] = to_numpy(xp.moveaxis(N_res,-1,0)[:,xp.newaxis,:,xp.newaxis])
            else:
                branch_supps[0]['spectra'][...] = to_numpy(fg_psd)
                branch_supps[0]['Nres'][...] = to_numpy(N_res)
        
        if return_spec:
            return self.cast(ln_p_tot), [to_numpy(fbins[1:]),to_numpy(fg_psd[1:]),to_numpy(N_res)]
        else:
            return self.cast(ln_p_tot)
    
    def reweight_foreground(self,coarsegrained_foreground):
        """
        Utility function to account for coarsegrained binning.
        The coarsegrained foreground power needs to be weighted by
        1/(delta f) to produce a correctly-normalized PSD.
        Parameters
        ----------
        coarsegrained_foreground : array
            The coarsegrained foreground amplitudes.

        Returns
        -------
        reweighted_foreground : array
            The foreground PSD.

        """
        
        return self.bin_width**(-1) * coarsegrained_foreground
    
    def run_model(self,pop_theta=None,return_extras=False):
        """
        Run the population model

        Parameters
        ----------
        pop_theta : {array, dict, list}, optional
            The population parameter draw. The default is None (samples from the attached hyperprior).
            Can be a dict of {hyperparameter_name:value} or an array/list of hyperparameter values
            in the same order as self.hpar_names.
        return_extras : bool, optional
            If True, the resolved binary indices and all binary parameters will also be returned.

        Returns
        -------
        fs : array
            Frequencies at which foreground_PSD is evaluated.
        foreground_PSD : array
            Foreground PSD for each realization.
        N_res : array
            Number of resolved binaries for each realization.
        res_idx : array
            Indices of the resolved binaries in galaxy_draw
        galaxy_draw : array
            Astrophysical parameters of all binaries.

        """
        
        # import pdb; pdb.set_trace()
        ## draw pop hyperparameters
        if pop_theta is None:
            # theta_shape = (1,)
            pop_theta = self.hyperprior.sample(1)
        elif (type(pop_theta) is xp.ndarray) or (type(pop_theta) is np.ndarray):
            # theta_shape = (1,)
            pop_theta = {key:xp.atleast_1d(val) for key, val in zip(self.hpar_names,pop_theta.T)}
        elif type(pop_theta) is list:
            # theta_shape = pop_theta[0].shape
            pop_theta = {key:xp.atleast_1d(val) for key, val in zip(self.hpar_names,pop_theta)}
        
        ## condition the astro parameter distributions on the hyperprior draw
        self.gbprior.condition(pop_theta)
        
        ## draw a sample galaxy
        ## of shape (N-realz,N,Npar)
        galaxy_draw = self.gbprior.sample_conditional(self.N)

        ## convert to phenomenological space
        amp_draws, fgw_draws = get_amp_freq(galaxy_draw)

        ## form array
        obs_draws = xp.array([fgw_draws,amp_draws]) ## 2 x N x Nreal x Nparallel
        
        ## sort into resolved and unresolved binaries
        if not return_extras:
            N_res, coarsegrain_fg = self.thresher.block_array_sort(obs_draws,
                                                                    self.fbins,
                                                                    snr_thresh=self.thresh_val)
        else:
            N_res, coarsegrain_fg, res_idx = self.thresher.serial_array_sort(obs_draws,
                                                                    self.fbins,
                                                                    snr_thresh=self.thresh_val,get_indices=True)
        ## reweight power spectral density back to density at observation frequencies
        foreground_psd = self.reweight_foreground(coarsegrain_fg)
        
        ## lowest bin is not accurate, discard,fbins=lowf_bins
        if not return_extras:
            return self.fbins[1:], foreground_psd[1:,...], N_res
        else:
            return self.fbins[1:], foreground_psd[1:,...], N_res, res_idx, galaxy_draw
    
    def sample_partial_likelihood(self,save_spec=False):
        """
        

        Parameters
        ----------
        save_spec : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        chain : array
            Parameter draws and associated log likelihood.
        
        fs : array
            [IF save_spec is True] Foreground spectrum frequencies
        specs : list of array
            [IF save_spec is True] Associated foreground spectra
        Ns : list of int
            [IF save_spec is True] Associated counts of resolved binaries

        """

        new_chain = xp.empty((len(self.hyperprior.hyperprior_dict)+1,self.Nsamp)) ## last column is for the likelihood
        if hasattr(self,'chain'):
            self.chain = xp.append(self.chain,new_chain,axis=1)
        else:
            self.chain = new_chain

        specs = []
        Ns = []
        if save_spec:
            for ii in tqdm(range(self.Nsamp)):
                draw = self.hyperprior.sample(1)
                self.chain[:-1,ii] = xp.array([draw[key] for key in draw.keys()]).flatten()
                self.chain[-1,ii], astro_result = self.fg_N_ln_prob(draw,return_spec=True)
                specs.append(astro_result[1])
                Ns.append(astro_result[2])
            fs = astro_result[0]
            return self.chain, fs, specs, Ns
        else:
            for ii in tqdm(range(self.Nsamp)):
                draw = self.hyperprior.sample(1)
                self.chain[:-1,ii] = xp.array([draw[key] for key in draw.keys()]).flatten()
                self.chain[-1,ii] = self.fg_N_ln_prob(draw)
        
            
            return self.chain
    
    def sample_likelihood(self,save_spec=False):
        """
        

        Parameters
        ----------
        save_spec : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        chain : array
            Parameter draws and associated log likelihood.
        
        fs : array
            [IF save_spec is True] Foreground spectrum frequencies
        specs : list of array
            [IF save_spec is True] Associated foreground spectra
        Ns : list of int
            [IF save_spec is True] Associated counts of resolved binaries

        """

        new_chain = xp.empty((len(self.hyperprior.hyperprior_dict)+1,self.Nsamp)) ## last column is for the likelihood
        if hasattr(self,'chain'):
            self.chain = xp.append(self.chain,new_chain,axis=1)
        else:
            self.chain = new_chain

        specs = []
        Ns = []
        if save_spec:
            for ii in tqdm(range(self.Nsamp)):
                draw = self.hyperprior.sample(1)
                self.chain[:-1,ii] = xp.array([draw[key] for key in draw.keys()]).flatten()
                self.chain[-1,ii], astro_result = self.ln_prob(draw,return_spec=True)
                specs.append(astro_result[1])
                Ns.append(astro_result[2])
            fs = astro_result[0]
            return self.chain, fs, specs, Ns
        else:
            for ii in tqdm(range(self.Nsamp)):
                draw = self.hyperprior.sample(1)
                self.chain[:-1,ii] = xp.array([draw[key] for key in draw.keys()]).flatten()
                self.chain[-1,ii] = self.ln_prob(draw)
        
            
            return self.chain