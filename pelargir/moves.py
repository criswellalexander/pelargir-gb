#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:07:57 2025

@author: Alexander W. Criswell

Houses custom Eryn moves.

"""

import numpy as np
from copy import deepcopy
from eryn.state import State
from eryn.moves import DistributionGenerate, Move


def make_PriorMove(eryn_prior,name='model_0'):
    """
    Wrapper function to make a Metropolis-Hasting prior draw move

    Parameters
    ----------
    eryn_prior : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is 'model_0'.

    Returns
    -------
    PriorMove : TYPE
        DESCRIPTION.

    """
    
    PriorMove = DistributionGenerate({'model_0':eryn_prior})
    return PriorMove


class PoissonMove(Move):
    '''
    A static Poisson move. This will remain at the current state in the population parameter space, but force reevaluation of the likelihood,
    thereby allowing the sampler to marginalize over the intrinsic Poisson scatter in the likelihood.
    '''

    def __init__(self, **kwargs):

        Move.__init__(self, **kwargs)


    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """
        
        This is essentially a wrapper; the method will return the current state as the new proposal.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        # set up all initial holders
        q = {}
        factors = {}
        new_inds = {}
        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords
            }

        # iterate through branches and propose new points where inds == True
        for i, (name, coords, inds) in enumerate(
            zip(
                branches_coords.keys(),
                branches_coords.values(),
                branches_inds.values(),
            )
        ):
            # copy over previous info
            ntemps, nwalkers, _, _ = coords.shape
            q[name] = coords.copy()
            new_inds[name] = inds.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # add coordinates for new leaves
            inds_here = np.where(inds == True)
            # num_inds_change = len(inds_here[0])

            old_points = coords[inds_here]

            q[name][inds_here] = old_points

        return q, factors

    def setup(self, branches_coords):
        """Any setup for the proposal.

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """
    
    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        
        self.setup(state.branches_coords)

        # get all branch names for gibbs setup
        all_branch_names = list(state.branches.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # in case there are no leaves yet
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # iterate through gibbs setup
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # setup supplemental information
            if not np.all(
                np.asarray(list(state.branches_supplemental.values())) == None
            ):
                new_branch_supps = deepcopy(state.branches_supplemental)
            else:
                new_branch_supps = None

            if state.supplemental is not None:
                new_supps = deepcopy(state.supplemental)
            else:
                new_supps = None

            # setup information according to gibbs info
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            # if no walkers are actually being proposed
            if not at_least_one_proposal:
                continue

            self.current_model = model
            self.current_state = state

            # Get the move-specific proposal.
            q, factors = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                branches_inds=inds_going_for_proposal,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # order everything properly
            q, _, new_branch_supps = self.ensure_ordering(
                list(state.branches.keys()), q, state.branches_inds, new_branch_supps
            )

            # if not wrapping with mutliple try (normal route)
            if not hasattr(self, "mt_ll") or not hasattr(self, "mt_lp"):
                # Compute prior of the proposed position
                logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

                self.fix_logp_gibbs(
                    branch_names_run, inds_run, logp, state.branches_inds
                )

                # Compute the lnprobs of the proposed position.
                # Can adjust supplementals in place
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=state.branches_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

            else:
                # if already computed in multiple try
                logl = self.mt_ll
                logp = self.mt_lp
                new_blobs = None

            # get log posterior
            # logP = self.compute_log_posterior(logl, logp)

            # # get previous information
            # prev_logl = state.log_like

            # prev_logp = state.log_prior

            # # takes care of tempering
            # prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # # difference
            # lnpdiff = factors + logP - prev_logP

            # 100% acceptance
            accepted = np.ones((ntemps, nwalkers),dtype=bool)

            # Update the parameters
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=new_blobs,
                inds=state.branches_inds,
                supplemental=new_supps,
                branch_supplemental=new_branch_supps,
            )

            state = self.update(state, new_state, accepted)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1
            

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)
        return state, accepted