# """
# File to house the rapid array sorting algorithm and inevitable variants.
# """
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


    def calc_Nij(self, A, noisePSD):
        '''
        Make the per-frequency SNR vector (dim 1xN_dwd)

        Arguments
        ------------
        A (float array)      : Sorted (ascending) DWD amplitudes
        noisePSD (float)     : Level of the noise PSD in the relevant frequency bin (i.e., S_n(f))
        '''
        return xp.sqrt(self.duration*A**2/((noisePSD + self.duration_eff * (xp.cumsum(A**2,axis=0) - A**2) )))

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


    def per_frequency_array_sort(self,amp_arr_i,Sn_i,snr_thresh=7):
        """
        
        Parameters
        ----------
        amp_arr_i : array
            The binary amplitudes for one frequency bin, of shape (:,Nrealizations,Nparallel)
        Sn_i : float
            The noise PSD in the frequency bin.
        snr_thresh : float, optional
            SNR threshold from resolved to unresolved. The default is 7.

        Returns
        -------
        None.

        """
        ## sort descending
        fbin_sort_i = xp.argsort(amp_arr_i,axis=0)
        sorted_amps_i = xp.take_along_axis(amp_arr_i, fbin_sort_i, axis=0)
        
        ## check that there are binaries in the bin, and skip if not
        if sorted_amps_i.shape[0] != 0:
            
            ## compute the thresholding
            fbin_Nij = self.calc_Nij(sorted_amps_i,Sn_i)

            ## threshold and store number of resolved binaries
            ## the multiply/subtract + argmax call addresses the fact that Nij >= snr_thresh can result in 
            ## an array with structure (e.g.) [False, False,  True, False, False,  True,  True]
            ## but only the systems after the last False
            ## (i.e. with amplitudes greated than the highest-amplitude unresolved binary)
            ## are in fact resolved. (order of sorted_fbin_amps_i is low -> high)
            snr_filt = fbin_Nij>=snr_thresh
            
            ## this is a very silly solution, but does work
            ## we multiply the boolean array by its indices along axis 0
            ## then by sliding the array and subtractin we can create a filter 
            ## which only registers as True only for the resolved binaries
            ## [0 0 1 0 0 1 1 1] -> [0 2 0 0 5 6 7] - [0 0 2 0 0 5 6] = [0 2 -2 0 5 1 1]
            ## argmax then returns index 4, and we filter to values > 4.
            ## As the original index 4 has to be zero to yield this result, the only
            ## entries >4 will be those after the final zero in the original array
            tilt_filt = snr_filt*xp.arange(snr_filt.shape[0])[:,None,None]
            res_filt = tilt_filt > xp.argmax(tilt_filt[1:,...]-tilt_filt[:-1,...],axis=0)
            
            fbin_res = xp.sum(res_filt,axis=0)
            foreground_amp = xp.sum((sorted_amps_i*xp.invert(res_filt))**2)
        else:
            fbin_res = xp.zeros(amp_arr_i.shape[1:],dtype='int')
            foreground_amp = xp.zeros(amp_arr_i.shape[1:])
        
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
        ## check binaries.shape to handle trailing axes
        ## force it to have shape (2,Ndraws,Nrealz,Nparallel)
        if binaries.ndim == 2:
            binaries_4d = binaries[:,:,xp.newaxis,xp.newaxis]
        elif binaries.ndim == 3: 
            binaries_4d = binaries[:,:,:,xp.newaxis]
        elif binaries.ndim == 4:
            binaries_4d = binaries
        else:
            raise ValueError("Invalid shape. Binaries can be of shapes \
                             (2,Ndraws), (2,Ndraws,Nrealz), or (2,Ndraws,Nrealz,Nparallel)")
        ## useful dims
        Nr = binaries_4d.shape[2] # realizations
        Np = binaries_4d.shape[3] # parallel
        
        amp_list = []
        f_idx_list = []
        
        ## loop over parallelization
        for pj in range(Np):
            ## loop over realizations
            for ri in range(Nr):
                amps_ij, f_idx_ij = self.coarsegrain_bin(binaries[...,ri,pj], fs)
                amp_list.append(amps_ij)
                f_idx_list.append(f_idx_ij)
        
        # frequency-dimension
        Nf = len(fs)
        
        ## initialize arrays of shape (Nf,Nrealz,Nparallel)
        foreground_amp = xp.zeros((Nf,Nr,Np))
        Nres_f = xp.zeros((Nf,Nr,Np),dtype='int')
        
        
        for ii in range(Nf):
            ## loop over realizations, parallelization to do setup
            amps_ii = [] ## amplitudes in fbin ii
            Ns_ii = []## total counts
            for jj, amps_all_jj in enumerate(amp_list):
                amps_ii.append(amps_all_jj[xp.array(f_idx_list[jj] == ii)])
                Ns_ii.append(len(amps_ii[jj]))
            ## instantiate array of shape (max(Ns_ii),Nr,Np)
            amp_arr_ii = xp.zeros((xp.max(Ns_ii),Nr,Np))
            ## assign values
            ## loop over parallelization
            for pj in range(Np):
                ## loop over realizations
                for ri in range(Nr):
                    ## multiply by the LISA response
                    ## sqrt because we square the amplitudes to get Sgw
                    amp_arr_ii[:len(amps_ii[pj*Nr+ri]),ri,pj] = amps_ii[pj*Nr+ri]*xp.sqrt(self.LISA_rx[ii])
            
            ## we now have an array-operation-ready frequency bin! run the thresher:
            Nres_f[ii,...], foreground_amp[ii,...] = self.per_frequency_array_sort(amp_arr_ii,
                                                                                   self.noisePSD[ii],
                                                                                   snr_thresh=snr_thresh,
                                                                                   compute_frac=compute_frac)
        
        # =============================================================================
        # FOR NOW (only care about Nres, not specifics)
        # =============================================================================
        Nres = xp.sum(Nres_f,axis=0)
        
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
        
  