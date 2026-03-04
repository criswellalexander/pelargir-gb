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

from cupyx.profiler import benchmark


class SNR_Threshold:

    def __init__(self,fs,noisePSD,LISA_rx,duration=1.262e8,block_after=None):
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
        self.block_after = block_after

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
        dwd_fs = binaries[0,...]
        dwd_amps = binaries[1,...]

        f_idx = xp.digitize(dwd_fs,xp.concatenate((fs-0.5*self.delf,xp.array(fs[-1]+0.5*self.delf).reshape(1,))))
        
        return dwd_amps, f_idx


    def per_frequency_array_sort(self,amp_arr_i,Sn_i,snr_thresh=7,return_indices=False):
        """
        
        Parameters
        ----------
        amp_arr_i : array
            The binary amplitudes for one frequency bin, of shape (:,Nrealizations,Nparallel)
        Sn_i : float
            The noise PSD in the frequency bin.
        snr_thresh : float, optional
            SNR threshold from resolved to unresolved. The default is 7.
        return_indices : bool
            Whether to return the resolved binary indices. The default is False.

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
            if sorted_amps_i.shape[0] > 1:
                tilt_filt = snr_filt*xp.arange(snr_filt.shape[0])[:,None,None]
                res_filt = tilt_filt > xp.argmax(tilt_filt[1:,...]-tilt_filt[:-1,...],axis=0)
            else:
                ## cases with 1 binary
                res_filt = snr_filt
            fbin_res = xp.sum(res_filt,axis=0)
            foreground_amp = xp.sum((sorted_amps_i*xp.invert(res_filt))**2,axis=0)
        else:
            res_filt = []
            fbin_res = xp.zeros(amp_arr_i.shape[1:],dtype='int')
            foreground_amp = xp.zeros(amp_arr_i.shape[1:])
        
        if not return_indices:
            return fbin_res, foreground_amp
        
        else:
            res_idx = fbin_sort_i[res_filt]
            return fbin_res, foreground_amp, res_idx
            
    
    
    def serial_array_sort(self,binaries,fs,snr_thresh=7,force_shape=False,get_indices=False):
        '''
        Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.
        
        As opposed to rapid_array_sort, serial_array_sort is serial across frequency bins

        Arguments
        -----------
        binaries (array)      : Array with binary info. Should be of shape (2,Ndraws,Nreal,Nparallel), where the 1st axis is (frequency,amplitude).
        fs (float array)      : Data frequencies.
        snr_thresh (float)    : The SNR threshold to condition resolved vs. unresolved on.
        force_shape (bool)    : Turn off safety checks related to the shape of binaries.
        get_indices (bool)    : Whether to track the resolved binary indices. Default False.

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
        
        ## for now, only allow returning indices for Nr=Np=1
        # if get_indices:
        #     assert binaries.ndim == 2
        
        ## useful dims
        Nr = binaries_4d.shape[2] # realizations
        Np = binaries_4d.shape[3] # parallel
        
        ## throw an error if there are more realizations or parallel threads than binaries
        if not force_shape:
            if Nr > binaries.shape[1] or Np > binaries.shape[1]:
                raise RuntimeError("Number of realizations is {} and number of parallel operations is {}, but there are only {} binaries. \
                                    This seems suspect...".format(Nr,Np,binaries.shape[0]))

        amp_list = []
        f_idx_list = []
        
        
        amps, f_idx = self.coarsegrain_bin(binaries_4d, fs)
        # ## loop over parallelization
        # for pj in range(Np):
        #     ## loop over realizations
        #     for ri in range(Nr):
        #         amps_ij, f_idx_ij = self.coarsegrain_bin(binaries_4d[...,ri,pj], fs)
        #         amp_list.append(amps_ij)
        #         f_idx_list.append(f_idx_ij)
        #         import pdb; pdb.set_trace()
        
        # frequency-dimension
        Nf = len(fs)
        
        ## initialize arrays of shape (Nf,Nrealz,Nparallel)
        foreground_amp = xp.zeros((Nf,Nr,Np))
        Nres_f = xp.zeros((Nf,Nr,Np),dtype='int')
        if get_indices:
            res_idx_list = [[] for item in range(Nf)]
        
        for ii in range(Nf):
        #     ## loop over realizations, parallelization to do setup
        #     amps_ii = [] ## amplitudes in fbin ii
        #     Ns_ii = []## total counts
        #     for jj, amps_all_jj in enumerate(amp_list):
        #         amps_ii.append(amps_all_jj[xp.array(f_idx_list[jj] == ii)])
        #         Ns_ii.append(len(amps_ii[jj]))
        #     ## instantiate array of shape (max(Ns_ii),Nr,Np)
        #     amp_arr_ii = xp.zeros((max(Ns_ii),Nr,Np))
        #     ## assign values
        #     ## loop over parallelization
        #     for pj in range(Np):
        #         ## loop over realizations
        #         for ri in range(Nr):
        #             ## multiply by the LISA response
        #             ## sqrt because we square the amplitudes to get Sgw
        #             amp_arr_ii[:len(amps_ii[pj*Nr+ri]),ri,pj] = amps_ii[pj*Nr+ri]*xp.sqrt(self.LISA_rx[ii])
            
            in_fbin_ii = xp.equal(f_idx,ii)
            Ns_ii = xp.sum(in_fbin_ii,axis=0)
            Nmax_ii = xp.max(Ns_ii)
            zpad_filt_ii = xp.greater(Ns_ii,xp.arange(Nmax_ii)[:,None,None])
            amp_arr_ii = xp.zeros((int(Nmax_ii),Nr,Np))
            for pj in range(Np):
                for ri in range(Nr):
                    amp_arr_ii[zpad_filt_ii[:,ri,pj],ri,pj] = amps[:,ri,pj][in_fbin_ii[:,ri,pj]]*xp.sqrt(self.LISA_rx[ii])
                    
            ## we now have an array-operation-ready frequency bin! run the thresher:
            
            if not get_indices:
                Nres_f[ii,...], foreground_amp[ii,...] = self.per_frequency_array_sort(amp_arr_ii,
                                                                                       self.noisePSD[ii],
                                                                                       snr_thresh=snr_thresh)
            else:
                Nres_f[ii,...], foreground_amp[ii,...], res_idx_ii = self.per_frequency_array_sort(amp_arr_ii,
                                                                                                        self.noisePSD[ii],
                                                                                                        snr_thresh=snr_thresh,
                                                                                                        return_indices=True)
                res_idx_list[ii] = res_idx_ii
        
        # =============================================================================
        # FOR NOW (only care about Nres, not specifics)
        # =============================================================================
        # import pdb; pdb.set_trace()
        Nres = xp.sum(Nres_f[1:,...],axis=0)
        
        ## if this is 1D, flatten the output
        if Nr==1 and Np==1:
            Nres = Nres.squeeze()
            foreground_amp = foreground_amp.squeeze()
        
        if not get_indices:
            return Nres, foreground_amp
        else:
            res_idx = [idx for block in res_idx_list for idx in block]
            return Nres, foreground_amp, res_idx
    
    
    def block_array_sort(self,binaries,fs,snr_thresh=7,force_shape=False,get_indices=False,
                         block_after=None):
        '''
        Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.
        
        As opposed to rapid_array_sort, serial_array_sort is serial across frequency bins

        Arguments
        -----------
        binaries (array)      : Array with binary info. Should be of shape (2,Ndraws,Nreal,Nparallel), where the 1st axis is (frequency,amplitude).
        fs (float array)      : Data frequencies.
        snr_thresh (float)    : The SNR threshold to condition resolved vs. unresolved on.
        force_shape (bool)    : Turn off safety checks related to the shape of binaries.
        get_indices (bool)    : Whether to track the resolved binary indices. Default False.

        Returns
        -----------
        foreground_amp (array) : Stochastic foreground from unresolved sources, evaluated at fs_full.
        N_res (int)            : Number of resolved DWDs
        
        '''
        if block_after is None:
            block_after = self.block_after
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
        
        ## for now, only allow returning indices for Nr=Np=1
        if get_indices:
            assert binaries.ndim == 2
        
        ## useful dims
        Nr = binaries_4d.shape[2] # realizations
        Np = binaries_4d.shape[3] # parallel
        
        ## throw an error if there are more realizations or parallel threads than binaries
        if not force_shape:
            if Nr > binaries.shape[1] or Np > binaries.shape[1]:
                raise RuntimeError("Number of realizations is {} and number of parallel operations is {}, but there are only {} binaries. \
                                    This seems suspect...".format(Nr,Np,binaries.shape[0]))
        
        
        amps, f_idx = self.coarsegrain_bin(binaries_4d, fs)

        # frequency-dimension
        Nf = len(fs)
        
        ## initialize arrays of shape (Nf,Nrealz,Nparallel)
        foreground_amp = xp.zeros((Nf,Nr,Np))
        Nres_f = xp.zeros((Nf,Nr,Np),dtype='int')
        if get_indices:
            raise ValueError("Tracking indices is not supported for the block array sort. Use serial_array_sort() instead.")
        
        ## low-f bins; do in serial but avoid calcs on bottom 95%
        for ii in range(block_after):
            
            # in_fbin_ii = xp.equal(f_idx,ii)
            # Ns_ii = xp.sum(in_fbin_ii,axis=0)
            # Nmax_ii = xp.max(Ns_ii)
            # zpad_filt_ii = xp.greater(Ns_ii,xp.arange(Nmax_ii)[:,None,None])
            # pre_arr_ii = xp.zeros((int(Nmax_ii),Nr,Np))
            # pre_arr_ii[zpad_filt_ii] = amps[in_fbin_ii]
            
            # ## avoid sorting vast majority of low-amplitude systems
            # quant_ii = xp.quantile(pre_arr_ii,0.95,axis=0)
            # minquant_ii = xp.min(quant_ii)
            # quantfilt_ii = xp.greater(pre_arr_ii,minquant_ii)
            # Nkeep_ii = xp.sum(quantfilt_ii,axis=0)
            # Nkeepmax_ii = xp.max(Nkeep_ii)
            # keep_filt_ii = xp.greater(Nkeep_ii,xp.arange(Nkeepmax_ii)[:,None,None])
            # amp_arr_ii = xp.zeros((int(Nkeepmax_ii),Nr,Np))
            # amp_arr_ii[keep_filt_ii] = pre_arr_ii[quantfilt_ii]*xp.sqrt(self.LISA_rx[ii])
            
            # in_fbin_ii = xp.equal(f_idx,ii)
            # Ns_ii = xp.sum(in_fbin_ii,axis=0)
            # Nmax_ii = xp.max(0.1*Ns_ii)
            # # zpad_filt_ii = xp.greater(Ns_ii,xp.arange(Nmax_ii)[:,None,None])
            # amp_arr_ii = xp.zeros((int(Nmax_ii),Nr,Np))
            # Sgw_ii = xp.zeros((Nr,Np))
            # for pj in range(Np):
            #     for ri in range(Nr):
            #         pre_arr_ii = amps[:,ri,pj][in_fbin_ii[:,ri,pj]]
            #         ## avoid sorting vast majority of low-amplitude systems
            #         quantfilt_ii = xp.greater(pre_arr_ii,xp.quantile(pre_arr_ii,0.95))
            #         import pdb; pdb.set_trace()
            #         # qmax_ii = int(xp.sum(quantfilt_ii,dtype='int32'))
            #         filt_arr_ii = pre_arr_ii[quantfilt_ii]       
            #         amp_arr_ii[:,ri,pj][:filt_arr_ii.size] = filt_arr_ii
            #         ## confusion noise from the discarded systems
            #         Sgw_ii[ri,pj] = self.duration_eff*self.LISA_rx[ii]*xp.sum(pre_arr_ii[xp.invert(quantfilt_ii)],axis=0)
            
            in_fbin_ii = xp.equal(f_idx,ii)
            Ns_ii = xp.sum(in_fbin_ii,axis=0)
            Nmax_ii = xp.max(Ns_ii)
            zpad_filt_ii = xp.greater(Ns_ii,xp.arange(Nmax_ii)[:,None,None])
            amp_arr_ii = xp.zeros((int(Nmax_ii),Nr,Np))
            for pj in range(Np):
                for ri in range(Nr):
                    amp_arr_ii[zpad_filt_ii[:,ri,pj],ri,pj] = amps[:,ri,pj][in_fbin_ii[:,ri,pj]]*xp.sqrt(self.LISA_rx[ii])
                    
            
            ## we now have an array-operation-ready frequency bin! run the thresher:
            Nres_f[ii,...], foreground_amp[ii,...] = self.per_frequency_array_sort(amp_arr_ii,
                                                                                       self.noisePSD[ii],
                                                                                       snr_thresh=snr_thresh)
        
        ## do all remaining bins simultaneously
        fbin_masks = [xp.equal(f_idx,ii) for ii in range(block_after,Nf)]
        counts = [xp.sum(fbin_masks[ii],axis=0) for ii in range(Nf-block_after)]
        max_counts = xp.max(xp.array(counts))
        amp_arr = xp.zeros((int(max_counts),Nf-block_after,Nr,Np))
        
        for ii in range(Nf-block_after):
            jj = ii + block_after
            
            # amp_idx = xp.greater(counts[ii],xp.arange(max_counts)[:,None,None])
            # amp_arr[:,ii,...][amp_idx] = amps[fbin_masks[ii]]*xp.sqrt(self.LISA_rx[jj])
        
            # in_fbin_ii = xp.equal(f_idx,ii)
            # Ns_ii = xp.sum(in_fbin_ii,axis=0)
            # Nmax_ii = xp.max(Ns_ii)
            # zpad_filt_ii = xp.greater(Ns_ii,xp.arange(Nmax_ii)[:,None,None])
            # amp_arr_ii = xp.zeros((int(Nmax_ii),Nr,Np))
            for pj in range(Np):
                for ri in range(Nr):
                    amp_arr[:counts[ii][ri,pj],ii,ri,pj] = amps[:,ri,pj][fbin_masks[ii][:,ri,pj]]*xp.sqrt(self.LISA_rx[jj])
        
        ## sort descending
        fbin_sort = xp.argsort(amp_arr,axis=0)
        sorted_amps = xp.take_along_axis(amp_arr, fbin_sort, axis=0)
        
        fbin_Nij = self.calc_Nij(sorted_amps, self.noisePSD[None,block_after:,None,None])
        
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
        if sorted_amps.shape[0] > 1:
            tilt_filt = snr_filt*xp.arange(snr_filt.shape[0])[:,None,None,None]
            res_filt = tilt_filt > xp.argmax(tilt_filt[1:,...]-tilt_filt[:-1,...],axis=0)
        else:
            ## cases with 1 binary
            res_filt = snr_filt
        Nres_f[block_after:,...] = xp.sum(res_filt,axis=0)
        foreground_amp[block_after:,...] = xp.sum((sorted_amps*xp.invert(res_filt))**2,axis=0)
        
        # =============================================================================
        # FOR NOW (only care about Nres, not specifics)
        # =============================================================================
        Nres = xp.sum(Nres_f[1:,...],axis=0)
        
        ## if this is 1D, flatten the output
        if Nr==1 and Np==1:
            Nres = Nres.squeeze()
            foreground_amp = foreground_amp.squeeze()
        
        return Nres, foreground_amp
    
    def rapid_array_sort(self,binaries,fs,snr_thresh=7):
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
        
  