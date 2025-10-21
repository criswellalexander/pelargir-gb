#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:12:16 2025

@author: Alexander W. Criswell

Houses the custom Eryn backend subclass.
"""

from eryn.backends import Backend
import numpy as np


class SupplementalBackend(Backend):
    """
    An advanced backend that stores the chain in memory, alongside the values of hierarchichal or latent variables
    stored via the BranchSupplemental utility.

    Parameters
    ----------
    branch_supp_dims : dict
        Dictionary whose keys are the keys of the branch supplemental dict, and whose values
        give the expected dimensions of the corresponding supplemental infomation to store in the backend.
        These should NOT include the base dimensions (ntemps,nwalkers,nleaves), only the additional
        supplemental dimensions.
    **kwargs : kwargs
        Keyword arguments to pass to Backend().
    
    """
    
    def __init__(self, branch_supp_dims, **kwargs):
        
        ## initialize backend with any kwargs
        super().__init__(**kwargs)
       
        ## store the branch supplemental references/dimensions
        self.branch_supp_dims = branch_supp_dims
        self.branch_supp_names = [key for key in branch_supp_dims.keys()]
        ## make sure these are all tuples, even if 1D
        for supp_name in self.branch_supp_names:
            if type(self.branch_supp_dims[supp_name]) is not tuple:
                self.branch_supp_dims[supp_name] = (self.branch_supp_dims[supp_name],)
    
    
    ## slightly modified reset to also clear/set up the supplemental chain
    ## note that reset is also called by ensemble() to set up the backend
    def reset(self,
              nwalkers,
              ndims,
              nleaves_max=1,
              ntemps=1,
              branch_names=None,
              nbranches=1,
              rj=False,
              moves=None,
              **info,):
        """
        Clear the state of the chain and empty the backend.
        If self.initialized is False, this sets up the backend.

        Args:
            nwalkers (int): The size of the ensemble (per temperature).
            ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
            nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
                It should have the same length as the number of branches.
                If ``dict``, keys should be the branch names and values the associated maximal leaf value.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            nbranches (int, optional): Number of branches. This is only used if ``branch_names is None``.
                (default: ``1``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            moves (list, optional): List of all of the move classes input into the sampler.
                (default: ``None``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend.

        """
        # store inputs for later resets
        self.reset_args = (nwalkers, ndims)
        self.reset_kwargs = dict(
            nleaves_max=nleaves_max,
            ntemps=ntemps,
            branch_names=branch_names,
            rj=rj,
            moves=moves,
            info=info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.rj = rj

        # turn things into lists/dicts if needed
        if branch_names is not None:
            if isinstance(branch_names, str):
                branch_names = [branch_names]

            elif not isinstance(branch_names, list):
                raise ValueError("branch_names must be string or list of strings.")

        else:
            branch_names = ["model_{}".format(i) for i in range(nbranches)]

        nbranches = len(branch_names)

        if isinstance(ndims, int):
            assert len(branch_names) == 1
            ndims = {branch_names[0]: ndims}

        elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
            assert len(branch_names) == len(ndims)
            ndims = {bn: nd for bn, nd in zip(branch_names, ndims)}

        elif isinstance(ndims, dict):
            assert len(list(ndims.keys())) == len(branch_names)
            for key in ndims:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in ndims but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("ndims is to be a scalar int, list or dict.")

        if isinstance(nleaves_max, int):
            assert len(branch_names) == 1
            nleaves_max = {branch_names[0]: nleaves_max}

        elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
            assert len(branch_names) == len(nleaves_max)
            nleaves_max = {bn: nl for bn, nl in zip(branch_names, nleaves_max)}

        elif isinstance(nleaves_max, dict):
            assert len(list(nleaves_max.keys())) == len(branch_names)
            for key in nleaves_max:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in nleaves_max but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("nleaves_max is to be a scalar int, list, or dict.")

        self.nbranches = len(branch_names)

        self.branch_names = branch_names
        self.ndims = ndims
        self.nleaves_max = nleaves_max

        self.iteration = 0

        # setup all the holder arrays
        self.accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)
        self.swaps_accepted = np.zeros((self.ntemps - 1,), dtype=self.dtype)
        if self.rj:
            self.rj_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)

        else:
            self.rj_accepted = None

        # chains are stored in dictionaries
        self.chain = {
            name: np.empty(
                (
                    0,
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[name],
                    self.ndims[name],
                ),
                dtype=self.dtype,
            )
            for name in self.branch_names
        }

        # inds correspond to leaves used or not
        self.inds = {
            name: np.empty(
                (0, self.ntemps, self.nwalkers, self.nleaves_max[name]), dtype=bool
            )
            for name in self.branch_names
        }
        
        ## chain_supplemental mimics the chain, but is a nested dict of branch_name:{supplemental_name:supp_chain}
        if self.nbranches > 1:
            raise Warning("nbranches is > 1, which is not yet supported for the SupplemetalBackend.\
                           Supplementals are assumed to be associated with the branch at position 0 ({}).".format(self.branch_names[0]))
        self.chain_supplemental = {name:{supp_name:np.empty((0,
                                                             self.ntemps,
                                                             self.nwalkers,
                                                             self.nleaves_max[name],
                                                             *self.branch_supp_dims[supp_name]),
                                                            dtype=self.dtype)
                                         for supp_name in self.branch_supp_names}
                                   for name in self.branch_names}
        

        # log likelihood and prior
        self.log_like = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)

        # temperature ladder
        self.betas = np.empty((0, self.ntemps), dtype=self.dtype)

        self.blobs = None

        self.random_state = None
        self.initialized = True

        # store move specific information
        if moves is not None:
            # setup info and keys
            self.move_info = {}
            self.move_keys = []
            for move in moves:
                # prepare information dictionary
                self.move_info[move] = {
                    "acceptance_fraction": np.zeros(
                        (self.ntemps, self.nwalkers), dtype=self.dtype
                    )
                }

                # update the move keys to keep proper order
                self.move_keys.append(move)

        else:
            self.move_info = None
    
    
    def get_chain_supplemental(self, **kwargs):
        """
        Get the stored chain of supplemental values.

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            dict: MCMC samples
                The dictionary contains np.ndarrays of supplemental samples
                across the branches.

        """
        return self.get_value("chain_supplemental", **kwargs)
    
    def grow(self, ngrow, blobs):
        """
        Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        # determine the number of entries in the chains
        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)
        {
            key: (self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key])
            for key in self.branch_names
        }

        # temperary addition to chains
        a = {
            key: np.empty(
                (i, self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key]),
                dtype=self.dtype,
            )
            for key in self.branch_names
        }
        # combine with original chain
        self.chain = {
            key: np.concatenate((self.chain[key], a[key]), axis=0) for key in a
        }

        # temporary addition to inds
        a = {
            key: np.empty(
                (i, self.ntemps, self.nwalkers, self.nleaves_max[key]), dtype=bool
            )
            for key in self.branch_names
        }
        # combine with original inds
        self.inds = {key: np.concatenate((self.inds[key], a[key]), axis=0) for key in a}
        
        ## temporary addition for chain_supplemental
        a = {key:{supp_key:np.empty((i,self.ntemps, self.nwalkers, self.nleaves_max[key], *self.branch_supp_dims[supp_key]),dtype=self.dtype)
                  for supp_key in self.branch_supp_names}
             for key in self.branch_names}
        ## combine with original chain_supplemental
        self.chain_supplemental = {key:{supp_key:np.concatenate((self.chain_supplemental[key][supp_key],a[key][supp_key]), axis=0)
                                        for supp_key in self.branch_supp_names}
                                   for key in self.branch_names}
        
        # temporary addition for log likelihood
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log likelihood
        self.log_like = np.concatenate((self.log_like, a), axis=0)

        # temporary addition for log prior
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log prior
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)

        # temporary addition for betas
        a = np.empty((i, self.ntemps), dtype=self.dtype)
        # combine with original betas
        self.betas = np.concatenate((self.betas, a), axis=0)

        if blobs is not None:
            dt = np.dtype((blobs.dtype, blobs.shape[2:]))
            # temporary addition for blobs
            a = np.empty((i, self.ntemps, self.nwalkers), dtype=dt)
            # combine with original blobs
            if self.blobs is None:
                self.blobs = a
            else:
                self.blobs = np.concatenate((self.blobs, a), axis=0)

    
    def save_step(self,
                  state,
                  accepted,
                  rj_accepted=None,
                  swaps_accepted=None,
                  moves_accepted_fraction=None):
        """
        Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            rj_accepted (ndarray, optional): An array of the number of accepted steps
                for the reversible jump proposal for each walker.
                If :code:`self.rj` is True, then rj_accepted must be an array with
                :code:`rj_accepted.shape == accepted.shape`. If :code:`self.rj`
                is False, then rj_accepted must be None, which is the default.
            swaps_accepted (ndarray, optional): 1D array with number of swaps accepted
                for the in-model step. (default: ``None``)
            moves_accepted_fraction (dict, optional): Dict of acceptance fraction arrays for all of the
                moves in the sampler. This dict must have the same keys as ``self.move_keys``.
                (default: ``None``)

        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            rj_accepted=rj_accepted,
            swaps_accepted=swaps_accepted,
        )

        # save the coordinates, inds, and supplementals
        for key, model in state.branches.items():
            self.inds[key][self.iteration] = model.inds
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = model.coords * model.inds[:, :, :, None]

            inds_all = np.repeat(model.inds, model.coords.shape[-1], axis=-1).reshape(
                model.inds.shape + (model.coords.shape[-1],)
            )
            coords_in[~inds_all] = self.store_missing_leaves
            self.chain[key][self.iteration] = coords_in
            
            ## also save the supplemental
            for supp_key in self.chain_supplemental[key].keys():
                self.chain_supplemental[key][supp_key][self.iteration] = state.branches_supplemental[key][0][supp_key]

        # save higher level quantities
        self.log_like[self.iteration, :, :] = state.log_like
        self.log_prior[self.iteration, :, :] = state.log_prior
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        if state.betas is not None:
            self.betas[self.iteration, :] = state.betas

        self.accepted += accepted

        if swaps_accepted is not None:
            self.swaps_accepted += swaps_accepted
        if self.rj:
            self.rj_accepted += rj_accepted

        # moves
        if moves_accepted_fraction is not None:
            if self.move_info is None:
                raise ValueError(
                    """moves_accepted_fraction was passed, but moves_info was not initialized. Use the moves kwarg 
                    in the reset function."""
                )

            # update acceptance fractions
            for move_key in self.move_keys:
                self.move_info[move_key]["acceptance_fraction"][:] = (
                    moves_accepted_fraction[move_key]
                )

        self.random_state = state.random_state
        self.iteration += 1