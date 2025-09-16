# """
# File to house the rapid array sorting algorithm and inevitable variants.
# """
import numpy as xp
# import cupy as xp


class SNR_Threshold:

    def __init__(self,fs,noisePSD,LISA_rx,duration=1.262e8):
        '''
        
        Arguments
        -------------
        fs (array) : Array of data frequencies
        noisePSD (array) : The LISA noise PSD at frequencies of fs
        LISA_rx (array) : The frequency-domain LISA response function
        duration (array) : The LISA mission duration. Default 4 years (1.262e8 s).
        
        
        Returns
        -------
        None.

        '''
        
        self.noisePSD = noisePSD
        self.duration = duration
        self.LISA_rx = LISA_rx

        ## deal with unclipped Fourier frequencies if needed
        if fs[0] == 0:
            fs = fs[1:]
            self.noisePSD = self.noisePSD[1:]

        ## bin the binaries by frequency
        ## first, find which frequency bin each binary is in
        self.delf = fs[1] - fs[0]
        
        self.duration_eff = 1/self.delf ## effective duration for new frequency resolution

        return


    def calc_Nij(self, A, lowamp_PSD, noisePSD):
        '''
        Make the per-frequency SNR vector (dim 1xN_dwd)

        Arguments
        ------------
        A (float array)      : Sorted (ascending) DWD amplitudes
        noisePSD (float)     : Level of the noise PSD in the relevant frequency bin (i.e., S_n(f))
        lowamp_PSD (float)   : Level of the low-amplitude contribution to the foreground PSD in the relevant frequency bin
        '''
        return xp.sqrt(self.duration*A**2/((noisePSD + lowamp_PSD + self.duration_eff * (xp.cumsum(A**2) - A**2) )))

    def coarsegrain_bin(self,binaries,fs):
        
        '''
        Sort the binaries into their proper (coarse-grained) frequency bins.

        Parameters
        ----------
        binaries : TYPE
            DESCRIPTION.
        fs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        dwd_fs = binaries[0,:]
        dwd_amps = binaries[1,:]

        f_idx = xp.digitize(dwd_fs,xp.concatenate((fs-0.5*self.delf,xp.array(fs[-1]+0.5*self.delf).reshape(1,))))
        
        return dwd_amps, f_idx


    def per_frequency_array_sort(self,ii,dwd_amps,f_idx,snr_thresh=7,compute_frac=0.3):
        """
        
        Parameters
        ----------
        ii : int
            Frequency index
        dwd_amps : array
            The binary amplitudes.
        f_idx : array
            Binned binary frequency indices.
        snr_thresh : float, optional
            SNR threshold from resolved to unresolved. The default is 7.
        compute_frac : float, optional
            Fraction of the binaries for which we perform explicit thresholding, from the top. 
            The remainder will be assumed to be unresolved. The default is 0.3.

        Returns
        -------
        None.

        """
        
        ## select for the binaries in bin ii
        fbin_mask_i = xp.array(f_idx == ii)
        ## grab the corresponding amplitudes
        fbin_amps_i = dwd_amps[fbin_mask_i]*xp.sqrt(self.LISA_rx[ii]) ## sqrt because we square the amplitudes to get Sgw
        ## sort descending
        fbin_sort_i = xp.argsort(fbin_amps_i)
        sorted_fbin_amps_i = fbin_amps_i[fbin_sort_i]
        
        
        ## check that there are binaries in the bin, and skip if not
        if len(sorted_fbin_amps_i) != 0:
            
            ## if compute_frac is not 1, only perform computations for a portion of the binaries
            if compute_frac != 1.0:
                hightail_filt = sorted_fbin_amps_i > sorted_fbin_amps_i[int((1-compute_frac)*len(sorted_fbin_amps_i))]
                hightail_idx = xp.where(hightail_filt)
                lowamp_PSD = self.duration_eff*xp.sum(sorted_fbin_amps_i[xp.invert(hightail_filt)]**2)
    
                compute_amps = sorted_fbin_amps_i[hightail_filt]
            else:
                lowamp_PSD = 0.0
                compute_amps = sorted_fbin_amps_i
            
            ## compute the thresholding
            fbin_Nij = self.calc_Nij(compute_amps,lowamp_PSD,self.noisePSD[ii])

            res_mask_i = xp.zeros(len(sorted_fbin_amps_i),dtype='bool')
            
            ## threshold and store number of resolved binaries
            ## the argmin call addresses the fact that Nij >= snr_thresh can result in an array 
            ## with structure (e.g.) [False, False, False,  True,  True, False, False,  True,  True,  True]
            ## but only the systems after the last False
            ## (i.e. with amplitudes greated than the highest-amplitude unresolved binary)
            ## are in fact resolved. (order of sorted_fbin_amps_i is low -> high)
            snr_filt = fbin_Nij>=snr_thresh
            loudest_unres_idx = snr_filt[::-1].argmin()
            if compute_frac != 1.0:
                res_mask_i[hightail_idx][-loudest_unres_idx:] = snr_filt[-loudest_unres_idx:]
                fbin_res = xp.sum(res_mask_i[hightail_idx][-loudest_unres_idx:])
            else:
                res_mask_i[-loudest_unres_idx:] = snr_filt[-loudest_unres_idx:]
                fbin_res = xp.sum(res_mask_i[-loudest_unres_idx:])

            res_mask_i_resort = res_mask_i[fbin_sort_i]
            ## foreground amplitude
            foreground_amp = xp.sum(fbin_amps_i[xp.invert(res_mask_i_resort)]**2)
        else:
            fbin_res = 0
            foreground_amp = 0.0
        
        return fbin_res, foreground_amp
    
    
    def serial_array_sort(self,binaries,fs,snr_thresh=7,compute_frac=0.1):
        '''
        Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.
        
        As opposed to rapid_array_sort, serial_array_sort is serial across frequency bins

        Arguments
        -----------
        binaries (array)      : array with binary info. Will rephrase arguments in terms of the specific needed components later.
        fs (float array)      : data frequencies
        snr_thresh (float)    : the SNR threshold to condition resolved vs. unresolved on
        compute_frac (float)  : Percent (from top) of sources in a given bin to perform the calculations on. 
                                Must be 0 < q < 1.

        Returns
        -----------
        foreground_amp (array) : Stochastic foreground from unresolved sources, evaluated at fs_full.
        N_res (int)            : Number of resolved DWDs
        
        '''
        ## bin out the binaries by frequency
        dwd_amps, f_idx = self.coarsegrain_bin(binaries, fs)
        
        # frequency-dimension
        Nf = len(fs)
        
        foreground_amp = xp.zeros(Nf)
        Nres_f = xp.zeros(Nf,dtype='int')
        
        for ii in range(Nf):
            Nres_f[ii], foreground_amp[ii] = self.per_frequency_array_sort(ii,dwd_amps,f_idx,
                                                                           snr_thresh=snr_thresh,
                                                                           compute_frac=compute_frac)
        # =============================================================================
        # FOR NOW (only care about Nres, not specifics)
        # =============================================================================
        Nres = xp.sum(Nres_f)
        
        return Nres, foreground_amp
        

    def rapid_array_sort(self,binaries,fs,snr_thresh=7,compute_frac=0.1):
        '''
        Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.
        
        NOTE --- NOT CURRENTLY RECOMMENDED DUE TO RAM/ALLOCATION INEFFICIENCY
            
            While this function in principle allows for completely data-parallel array calculations on GPU,
            its current RAM and allocation costs due to zero-padding the binaries x frequencies array
            exceed feasible usage on most --- if not all --- GPUs. Use serial_array_sort for now.

        Arguments
        -----------
        binaries (array) : array with binary info. Will rephrase arguments in terms of the specific needed components later.
        fs (float array) : data frequencies
        snr_thresh (float)    : the SNR threshold to condition resolved vs. unresolved on
        compute_frac (float : Percent (from top) of sources in a given bin to perform the calculations on. Must be 0 < q < 1.

        Returns
        -----------
        foreground_amp (array) : Stochastic foreground from unresolved sources, evaluated at fs_full.
        N_res (int)            : Number of resolved DWDs
        
        '''
        
        ## bin out the binaries by frequency
        dwd_amps, f_idx = self.coarsegrain_bin(binaries, fs)
        
        # frequency-dimension
        Nf = len(fs)
        
        fbin_masks = [xp.array(f_idx == ii) for ii in range(Nf)]
        fbin_amps = [dwd_amps[fbin_masks[ii]]*xp.sqrt(self.LISA_rx[ii]) for ii in range(Nf)]
        
        ## can probably do some optimization here; I don't think I can prove that the first bin **always** has
        ## the most binaries, but it should be one of the first few bins in most cases
        dims = [xp.sum(fbin_masks[ii]) for ii in range(Nf)]
        
        ## instantiate the array as zeros so we don't have to fill in later
        binned_array = xp.zeros((xp.max(dims),Nf))

        ## and fill it in where needed with the ragged data
        for ii in range(Nf):
            binned_array[xp.arange(dims[ii]),ii] = fbin_amps[ii]
        
        ## now we can apply argsort to the entire array in a data parallel way
        sorted_idx = xp.argsort(binned_array,axis=0)
        
        sorted_array = xp.take_along_axis(binned_array,sorted_idx,axis=0)
        
        if compute_frac != 1.0:
            ## only perform calculations on upper [compute_frac] of the array
            thinned_array = sorted_array[:int(sorted_array.shape[0]*compute_frac),:]
            
            lowamp_PSD = self.duration_eff*xp.sum(sorted_array[int(sorted_array.shape[0]*compute_frac):,:]**2)
        else:
            thinned_array = sorted_array
            lowamp_PSD = xp.zeros(Nf)
        
        Nij = self.calc_Nij(thinned_array, lowamp_PSD, self.noisePSD)
        
        ## filter to resolved sources
        res_filt = Nij >= snr_thresh
        
        foreground_amp = xp.sum(thinned_array[xp.invert(res_filt)]**2,axis=0)

        # =============================================================================
        # FOR NOW (only care about Nres, not specifics)
        # =============================================================================
        Nres_f = xp.sum(res_filt,axis=0)
        
        
        return Nres_f, foreground_amp
        
  