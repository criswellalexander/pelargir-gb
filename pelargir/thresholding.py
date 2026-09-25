# """
# File to house the rapid array sorting algorithm and inevitable variants.
# """
from backend import xp


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

        ## band-wide response-weighted noise floor: min_f( noisePSD(f)/LISA_rx(f) )
        self.min_sens = xp.min(self.noisePSD / self.LISA_rx)

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

    def naive_snr_max(self, amp):
        '''
        Sound (best-case) upper bound on a binary's SNR, needing no frequency-bin
        assignment. Evaluates the noise floor at self.min_sens = min_f(noisePSD(f)/LISA_rx(f)) --
        the band-wide lowest-noise, highest-response point -- and drops the confusion
        term entirely. Both relaxations only ever raise the SNR estimate relative to
        the true (per-bin, confusion-inclusive) value, so a binary with
        naive_snr_max < snr_thresh is guaranteed unresolved, regardless of which
        frequency bin it actually falls in or how crowded that bin is.

        This is a further-relaxed relative of calc_Nij with the confusion term dropped
        (i.e. SNR w.r.t. instrumental noise only, evaluated at the binary's own bin);
        naive_snr_max relaxes that once more by using the band-wide minimum of
        noisePSD/LISA_rx in place of the binary's own bin, so it needs no digitize/
        binning step at all.

        Arguments
        ------------
        amp (array) : Response-free GW strain amplitude(s), e.g. from utils.get_amp_freq.
        '''
        return xp.sqrt(self.duration*amp**2/self.min_sens)

    def naive_snr_survives(self, amp, snr_thresh=7):
        '''
        Boolean mask: True means naive_snr_max does not rule this binary out (it must
        go through the real per-bin thresholding); False means guaranteed unresolved.

        snr_thresh should match the value passed to serial_array_sort/block_array_sort.
        '''
        return self.naive_snr_max(amp) >= snr_thresh

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

        ## digitize (right=False) returns #{k : edges[k] <= x}. Using the bin UPPER
        ## edges as the boundaries means a binary anywhere in
        ## [fs[k]-0.5*delf, fs[k]+0.5*delf) gets f_idx = k, so f_idx indexes fs --
        ## and hence self.noisePSD and self.LISA_rx -- directly. Sources below the
        ## band collect in bin 0 (discarded downstream by PopModel.run_model);
        ## sources above it get f_idx = Nf and are dropped by the range(Nf) loops.
        f_idx = xp.digitize(dwd_fs,fs+0.5*self.delf)
        
        return dwd_amps, f_idx


    def prefilter_and_partial_foreground(self,binaries,fs,snr_thresh=7):
        '''
        Bins every binary (cheap, O(Ndraws) digitize via coarsegrain_bin) and splits
        them into "survivors" (must go through the real per-bin sort/threshold) and
        "guaranteed unresolved" (naive_snr_survives is False). The latter's power is
        scatter-added into a per-bin foreground accumulator instead of being sorted.

        foreground_amp_partial is meant to be passed to serial_array_sort/
        block_array_sort as extra_confusion_psd, after trimming binaries down to the
        survivors, so their removal doesn't change the survivors' SNR estimates.

        Arguments
        -----------
        binaries (array) : Array with binary info, of shape (2,Ndraws), (2,Ndraws,Nrealz),
            or (2,Ndraws,Nrealz,Nparallel) -- same convention as serial_array_sort/
            block_array_sort.
        fs (float array) : Data frequencies.
        snr_thresh (float) : Should match what will be passed to serial_array_sort/
            block_array_sort downstream, for naive_snr_max's bound to be meaningful.

        Returns
        -----------
        survive_mask (bool array) : True = must go through the real sort. Shape
            (Ndraws,Nrealz,Nparallel), squeezed to (Ndraws,) if Nrealz==Nparallel==1.
        foreground_amp_partial (array) : Squared response-weighted amplitude of
            dropped, in-band binaries, summed per frequency bin. Shape (Nf,Nrealz,Nparallel),
            squeezed to (Nf,) if Nrealz==Nparallel==1 -- same convention as the
            foreground_amp returned by serial_array_sort/block_array_sort (bin 0
            included; out-of-band binaries excluded).
        '''
        if binaries.ndim == 2:
            binaries_4d = binaries[:,:,xp.newaxis,xp.newaxis]
        elif binaries.ndim == 3:
            binaries_4d = binaries[:,:,:,xp.newaxis]
        elif binaries.ndim == 4:
            binaries_4d = binaries
        else:
            raise ValueError("Invalid shape. Binaries can be of shapes \
                             (2,Ndraws), (2,Ndraws,Nrealz), or (2,Ndraws,Nrealz,Nparallel)")

        amps, f_idx = self.coarsegrain_bin(binaries_4d, fs)
        Nf = len(fs)
        Nr, Np = amps.shape[1], amps.shape[2]

        survive_mask = self.naive_snr_survives(amps, snr_thresh)
        in_band = f_idx < Nf
        dropped_mask = xp.logical_and(xp.logical_not(survive_mask), in_band)

        foreground_amp_partial = xp.zeros((Nf, Nr, Np))
        for pj in range(Np):
            for ri in range(Nr):
                sel = dropped_mask[:, ri, pj]
                f_idx_sel = f_idx[:, ri, pj][sel]
                ## cupy's bincount fails on empty input even with minlength set
                if f_idx_sel.size == 0:
                    continue
                weighted_amp_sq_sel = amps[:, ri, pj][sel]**2 * self.LISA_rx[f_idx_sel]
                foreground_amp_partial[:, ri, pj] = xp.bincount(f_idx_sel, weights=weighted_amp_sq_sel,
                                                                 minlength=Nf)

        if Nr==1 and Np==1:
            survive_mask = survive_mask.squeeze()
            foreground_amp_partial = foreground_amp_partial.squeeze()

        return survive_mask, foreground_amp_partial


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
            ## Nij >= snr_thresh can result in an array with structure (e.g.)
            ## [False, False,  True, False, False,  True,  True]
            ## but only the systems after the last False
            ## (i.e. with amplitudes greater than the highest-amplitude unresolved binary)
            ## are in fact resolved (Eq. 17 of arXiv:2604.03390). Order is low -> high.
            snr_filt = fbin_Nij>=snr_thresh

            ## index of the last sub-threshold binary in the bin; -1 if every binary passes
            idx = xp.arange(snr_filt.shape[0])[:,None,None]
            last_sub = xp.max(xp.where(snr_filt, -1, idx), axis=0)
            res_filt = idx > last_sub
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
            
    
    
    def serial_array_sort(self,binaries,fs,snr_thresh=7,force_shape=False,get_indices=False,
                          extra_confusion_psd=None):
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
        extra_confusion_psd (array) : Optional per-bin confusion power (shape
            (Nf,Nrealz,Nparallel)) folded into the noise floor used by calc_Nij; see
            prefilter_and_partial_foreground. Default None (noisePSD used as-is).

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

        amps, f_idx = self.coarsegrain_bin(binaries_4d, fs)
        Ntot = amps.shape[0]
        binary_inds = xp.arange(Ntot)[:,None,None]
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

        ## fold in extra confusion power, if any; see prefilter_and_partial_foreground.
        if extra_confusion_psd is None:
            noisePSD_eff = xp.broadcast_to(self.noisePSD[:,None,None], (Nf,Nr,Np))
        else:
            noisePSD_eff = self.noisePSD[:,None,None] + self.duration_eff*extra_confusion_psd.reshape(Nf,Nr,Np)

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
                                                                                       noisePSD_eff[ii],
                                                                                       snr_thresh=snr_thresh)
            else:
                Nres_f[ii,...], foreground_amp[ii,...], res_idx_ii = self.per_frequency_array_sort(amp_arr_ii,
                                                                                                        noisePSD_eff[ii],
                                                                                                        snr_thresh=snr_thresh,
                                                                                                        return_indices=True)
                res_idx_list[ii] = binary_inds[in_fbin_ii][res_idx_ii]
        
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
                         block_after=None,extra_confusion_psd=None):
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
        extra_confusion_psd (array) : Optional per-bin confusion power (shape
            (Nf,Nrealz,Nparallel)) folded into the noise floor used by calc_Nij; see
            prefilter_and_partial_foreground. Default None (noisePSD used as-is).

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

        ## fold in extra confusion power, if any; see prefilter_and_partial_foreground.
        if extra_confusion_psd is None:
            noisePSD_eff = xp.broadcast_to(self.noisePSD[:,None,None], (Nf,Nr,Np))
        else:
            noisePSD_eff = self.noisePSD[:,None,None] + self.duration_eff*extra_confusion_psd.reshape(Nf,Nr,Np)

        ## initialize arrays of shape (Nf,Nrealz,Nparallel)
        foreground_amp = xp.zeros((Nf,Nr,Np))
        Nres_f = xp.zeros((Nf,Nr,Np),dtype='int')
        if get_indices:
            raise ValueError("Tracking indices is not supported for the block array sort. Use serial_array_sort() instead.")

        ## low-f bins; do in serial but avoid calcs on bottom 95%
        for ii in range(block_after):

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
                                                                                       noisePSD_eff[ii],
                                                                                       snr_thresh=snr_thresh)
        
        ## do all remaining bins simultaneously
        fbin_masks = [xp.equal(f_idx,ii) for ii in range(block_after,Nf)]
        counts = [xp.sum(fbin_masks[ii],axis=0) for ii in range(Nf-block_after)]
        max_counts = xp.max(xp.array(counts))
        amp_arr = xp.zeros((int(max_counts),Nf-block_after,Nr,Np))
        
        for ii in range(Nf-block_after):
            jj = ii + block_after

            for pj in range(Np):
                for ri in range(Nr):
                    amp_arr[:counts[ii][ri,pj],ii,ri,pj] = amps[:,ri,pj][fbin_masks[ii][:,ri,pj]]*xp.sqrt(self.LISA_rx[jj])
        
        ## sort descending
        fbin_sort = xp.argsort(amp_arr,axis=0)
        sorted_amps = xp.take_along_axis(amp_arr, fbin_sort, axis=0)
        
        fbin_Nij = self.calc_Nij(sorted_amps, noisePSD_eff[None,block_after:,:,:])
        
        ## threshold and store number of resolved binaries
        ## only the systems after the last sub-threshold binary in each bin are resolved
        ## (Eq. 17 of arXiv:2604.03390); see per_frequency_array_sort.
        ## Zero-padded entries sort first and always fail the threshold, so they are never resolved.
        snr_filt = fbin_Nij>=snr_thresh

        idx = xp.arange(snr_filt.shape[0])[:,None,None,None]
        if snr_filt.shape[0] > 0:
            last_sub = xp.max(xp.where(snr_filt, -1, idx), axis=0)
            res_filt = idx > last_sub
        else:
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
        
  