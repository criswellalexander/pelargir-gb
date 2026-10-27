"""
Shared fixtures: a small toy dataset and a FlowPosterior on a small randomized flax flow (used by
tests/test_flow_inference.py and tests/test_flow_hmc.py).
"""
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest


## 10 modelled bins on 0.11-0.29 mHz (flows.data.model_fbins(1e-4, 3e-4, 2e-5)), two bands of 5;
## spelled out so this file imports nothing from pelargir (gwenv-1 has no flax)
FBINS = np.arange(1e-4 - 1e-5, 3e-4 + 1e-5, 2e-5)
## a small galaxy whose log10 lambda_tot (5.78) is inside the hyperprior
LAMBDA = 6e5


@pytest.fixture(scope="session")
def toy():
    pytest.importorskip("flax")
    from pelargir.inference import flow_posterior as fi
    return fi.simulate_toy_data(fi.FIDUCIAL, 7.0, LAMBDA, FBINS, seed=3)


@pytest.fixture(scope="session")
def post(toy):
    '''A FlowPosterior on a small flax flow with transforms fitted around the toy spectrum and random weights.'''
    from pelargir import backend
    jax = backend.import_jax()
    from flax import nnx
    from pelargir.flows import data as fd, flax_base as ff
    from pelargir.inference import flow_posterior as fi
    rng = np.random.default_rng(0)
    n = 400
    ctx = fd.sample_context(rng, n)
    psd = toy.truth_S*10**(0.05*rng.standard_normal((n, toy.fs.size)))
    nres = rng.poisson(np.maximum(toy.nres_f, 1), (n, toy.fs.size))
    ts = fd.TrainingSet(ctx, nres.astype(np.int32), psd, np.full(n, LAMBDA), FBINS, {})
    em = ff.BandedFlowEmulator(fd.make_bands(ts.fs, bins_per_band=5), FBINS, flow_num_layers=4, hidden_size=16)
    for j, f in enumerate(em.flows):
        f.fit_transforms(*fd.band_view(ts, f.band), rng)
        ## small random weights, so the flow is not its base distribution
        state = nnx.state(f.model, nnx.Param)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        keys = jax.random.split(jax.random.key(20 + j), len(leaves))
        nnx.update(f.model, jax.tree_util.tree_unflatten(treedef, [0.05*jax.random.normal(k, l.shape, l.dtype)
                                                                    for k, l in zip(keys, leaves)]))
    return fi.FlowPosterior(em, toy)


def near_truth_points(post, M, seed=1, scale=0.02):
    '''M points near the truth (theta rows inside the prior, y = log10 S_gw).'''
    rng = np.random.default_rng(seed)
    th, y = (np.asarray(a) for a in post.truth)
    th = np.repeat(th, M, axis=0)*(1 + scale*rng.standard_normal((M, 8)))
    y = np.repeat(y, M, axis=0) + scale*rng.standard_normal((M, y.shape[1]))
    return th, y


