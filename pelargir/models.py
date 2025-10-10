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
from inference import PopulationHyperPrior, GalacticBinaryPrior, FG_Likelihood, Nres_Likelihood

class PopModel():
    '''
    Class to house the overall population model.
    '''

    def __init__(self,Ntot,rng,hyperprior='default',
                 fbins='default',Tobs=4*u.yr,Nsamp=1,
                 thresholding="SNR",threshold_val=7.0,
                 thresh_compute_frac=1.0):
        
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
        
        self.gbprior = GalacticBinaryPrior(rng)

        self.N = int(Ntot)

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
            self.thresher = SNR_Threshold(self.fbins, self.approx_lisa_psd, self.approx_lisa_rx)
            self.thresh_val = threshold_val
            self.tc_frac = thresh_compute_frac
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

    def construct_likelihood(self,data):
        '''
        Wrapper to build all the likelihoods
        '''

        fg_data = data['fg']
        fg_sigma =data['fg_sigma']
        N_res_data = data['Nres']
        
        if 'noise' in data.keys():
            noise = data['noise']
        else:
            noise='default'

        self.construct_fg_likelihood(fg_data,fg_sigma,noise_psd=noise)
        self.construct_Nres_likelihood(N_res_data)

        return
    
    def construct_fg_likelihood(self,fg_psd,psd_sigma,noise_psd='default'):
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

        Returns
        -------
        None.

        """
        if (type(noise_psd) is str) and (noise_psd == 'default'):
            noise_psd = self.approx_lisa_psd
        

        self.fg_like = FG_Likelihood(fg_psd,psd_sigma,noise_psd)
        self.fg_ln_prob = self.fg_like.ln_prob

        return

    def construct_Nres_likelihood(self,N_res_obs):
        '''
        Method to attach the Poisson likelihood for the number of resolved binaries to the PopModel
        '''
        self.Nres_like = Nres_Likelihood(N_res_obs)
        self.N_res_ln_prob = self.Nres_like.ln_prob

        return
    
    def fg_N_ln_prob(self,pop_theta,return_spec=False):
        '''
        Function to get the model probability conditioned on only 
        the per-bin foreground amplitude and the total number of resolved binaries

        Eventually we can extend this to per-bin N_res
        '''
        # ## unpack data
        # N_res_obs = data['N_res']
        # fg_obs = data['fg']

        ## call the population model
        fbins, fg_psd, N_res = self.run_model(pop_theta)

        ## call the fg likelihood
        ln_p_fg = self.fg_ln_prob(fg_psd)

        ln_p_Nres = self.N_res_ln_prob(N_res)

        if return_spec:
            return self.cast(ln_p_fg + ln_p_Nres), [self.cast(fbins[1:]),self.cast(fg_psd[1:]),self.cast(N_res)]
        else:
            return self.cast(ln_p_fg + ln_p_Nres)
    
    def reweight_foreground(self,coarsegrained_foreground):
        """
        Utility function to account for coarsegrained binning.
        The coarsegrained foreground PSD needs to be rebinned to the 
        original frequency resolution

        Parameters
        ----------
        coarsegrained_foreground : array
            The coarsegrained foreground amplitudes.

        Returns
        -------
        reweighted_foreground : array
            The foreground amplitudes at the original frequency resolution.

        """
        
        return (self.Tobs / self.bin_width**(-1))*coarsegrained_foreground
    
    def run_model(self,pop_theta=None):

        ## draw pop hyperparameters
        if pop_theta is None:
            pop_theta = self.hyperprior.sample(1)
        elif (type(pop_theta) is list) or (type(pop_theta) is xp.ndarray) or (type(pop_theta) is np.ndarray):
            pop_theta = {key:xp.atleast_1d(val) for key, val in zip(self.hpar_names,pop_theta)}

        ## condition the astro parameter distributions on the hyperprior draw
        self.gbprior.condition(pop_theta)

        ## draw a sample galaxy
        galaxy_draw = self.gbprior.sample_conditional(self.N)

        ## convert to phenomenological space
        amp_draws, fgw_draws = get_amp_freq(galaxy_draw)

        ## form array
        obs_draws = xp.array([fgw_draws,amp_draws])
        
        ## sort into resolved and unresolved binaries
        N_res, coarsegrain_fg_amp = self.thresher.serial_array_sort(obs_draws,
                                                                    self.fbins,
                                                                    snr_thresh=self.thresh_val,
                                                                    compute_frac=self.tc_frac)
        
        ## reweight power spectral density back to density at observation frequencies
        fg_psd = self.reweight_foreground(coarsegrain_fg_amp)

        ## lowest bin is not accurate, discard,fbins=lowf_bins
        return self.fbins[1:], fg_psd[1:], N_res
    
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