# """
# File to house the rapid array sorting algorithm and inevitable variants.
# """
import numpy as np


class snr_threshold:

    def __init__(self,noisePSD,LISA_rx,duration,duration_eff):
        '''
        
        Arguments
        -------------
        
        
        
        Returns
        -------
        None.

        '''
        
        self.noisePSD = noisePSD
        self.duration = duration
        self.duration_eff = duration_eff
        self.LISA_rx = LISA_rx

        return


    def calc_Nij(self, A, lowamp_PSD):
        '''
        Make the per-frequency SNR vector (dim 1xN_dwd)

        Arguments
        ------------
        A (float array)      : Sorted (ascending) DWD amplitudes
        noisePSD (float)     : Level of the noise PSD in the relevant frequency bin (i.e., S_n(f))
        lowamp_PSD (float)   : Level of the low-amplitude contribution to the foreground PSD in the relevant frequency bin
        '''
        return np.sqrt(self.duration*A**2/((self.noisePSD + lowamp_PSD + self.duration_eff * (np.cumsum(A**2) - A**2) )))

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
        
        
        
        return


    def rapid_array_sort(self,binaries,fs,snr_thresh=7,compute_frac=0.1):
        '''
        Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.

        Arguments
        -----------
        binaries (dataframe) : df with binary info. Will rephrase arguments in terms of the specific needed components later.
        fs (float array) : data frequencies
        snr_thresh (float)    : the SNR threshold to condition resolved vs. unresolved on
        quantile (float : Percent (from bottom) of sources in a given bin to assume are unresolved. Must be 0 < q < 1.

        Returns
        -----------
        foreground_amp (array) : Stochastic foreground from unresolved sources, evaluated at fs_full.
        N_res (int)            : Number of resolved DWDs
        res_idx (array)        : Indices of the binaries dataframe for resolved DWDs.
        unres_idx (array)      : Indices of the binaries dataframe for unresolved DWDs.
        '''
        # dwd_fs = np.array(binaries['fs'])
        # dwd_amps = np.array(binaries['hs'])

        dwd_fs = binaries[0,:]
        dwd_amps = binaries[1,:]

        dwd_idx = np.arange(len(dwd_amps))
        ## constrain to frequencies where we have a noise curve
        fs_noise = fs ## lazy
        fs_full = fs ## lazy
        if fs_noise[0] == 0:
            fs_noise = fs_noise[1:]
            noisePSD = noisePSD[1:]
        noise_f_mask = (fs_full>=fs_noise.min()) & (fs_full<=fs_noise.max())
        fs_full = fs_full[noise_f_mask]
        ## find which noise frequency corresponds to each frequency bin
    #     noise_f_idx = np.digitize(fs_full,fs_noise-(fs_noise[1]-fs_noise[0])/2)

        ## bin the binaries by frequency
        ## first, find which frequency bin each binary is in
        delf = fs_full[1] - fs_full[0]
        f_idx = np.digitize(dwd_fs,fs_full+0.5*delf)
        duration_eff = 1/delf ## effective duration for new frequency resolution


        ## now created a ragged list of arrays of varying sizes, corresponding to N_dwd(f_i)
        ## each entry is an array containing the indices of the DWDs in that bin, sorted by ascending amplitude*
        ##     * under the current assumption of uniform responses, this is equivalent to sorting by the naive SNR
        ##       (!! -- we will need to refine this in future)
        fbin_res_list = []
        foreground_amp = np.zeros(len(fs_full))
        iter_range = len(fs_full)
        for i in range(iter_range):
            fbin_mask_i = np.array(f_idx == i)
            fbin_amps_i = dwd_amps[fbin_mask_i]*np.sqrt(LISA_rx[i]) ## sqrt because we square the amplitudes to get Sgw
            fbin_sort_i = np.argsort(fbin_amps_i)
            re_sort_i = np.argsort(fbin_sort_i) ## this will allow us to later return to the original order
            sorted_fbin_amps_i = fbin_amps_i[fbin_sort_i]
            if len(sorted_fbin_amps_i) != 0:
                hightail_filt = sorted_fbin_amps_i > sorted_fbin_amps_i[int((1-compute_frac)*len(sorted_fbin_amps_i))]
                # print(np.sum(hightail_filt)/len(sorted_fbin_amps_i))
                hightail_idx = np.where(hightail_filt)
                lowamp_idx = np.where(np.invert(hightail_filt))
                # bin_amps_i[fbin_sort_i] > np.quantile(fbin_amps_i[fbin_sort_i],0.9)
                lowamp_PSD = duration_eff*np.sum(wts*sorted_fbin_amps_i[np.invert(hightail_filt)]**2)
                # print(lowamp_PSD,noisePSD[i])

                high_tail = sorted_fbin_amps_i[hightail_filt]

                fbin_Nij = rebin_calc_Nij(high_tail,noisePSD[i],lowamp_PSD,wts,duration,duration_eff)
                # if fbin_Nij.size > 0:
                    # print(np.max(fbin_Nij))
                res_mask_i = np.zeros(len(sorted_fbin_amps_i),dtype='bool')
                res_mask_i[hightail_idx] = fbin_Nij>=snr_thresh
                # print(np.sum(res_mask_i))
                res_mask_i_resort = res_mask_i[re_sort_i]
                fbin_res_list.append(dwd_idx[fbin_mask_i][res_mask_i_resort])

                foreground_amp[i] = np.sum(fbin_amps_i[np.invert(res_mask_i_resort)]**2)
            else:
                foreground_amp[i] = 0

        ##unpack the binned list
        res_idx = np.array([],dtype=int)
        for i, arr in enumerate(fbin_res_list):
            res_idx = np.append(res_idx,arr)
        N_res = len(res_idx)
        unres_idx = np.isin(dwd_idx,res_idx,invert=True)

        return foreground_amp, fs_full, N_res, res_idx, unres_idx
