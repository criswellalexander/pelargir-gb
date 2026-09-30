"""
The cupy and jax backends give the same results on an identical numpy draw: run_pelargir's
simulated-data step (serial_array_sort on cupy against the JAX thresholder on host numpy) and
PopModel's likelihood terms (the cupy likelihood classes against jax_likelihood.ln_like).
Each backend runs in its own process (tests/backend_route_worker.py). Needs cupy with a GPU and jax.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import gammaln

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_route_worker.py")


def run(backend, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PELARGIR_")}
    env["PELARGIR_BACKEND"] = backend
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, WORKER, *args], env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]


def cupy_gpu_available():
    res = subprocess.run([sys.executable, "-c", "import cupy; assert cupy.cuda.is_available()"],
                         capture_output=True, timeout=120)
    return res.returncode == 0


pytestmark = pytest.mark.skipif(not cupy_gpu_available(), reason="cupy with a CUDA GPU is not available")


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    pytest.importorskip("jax")
    tmp = tmp_path_factory.mktemp("routes")
    run("numpy", "draw", str(tmp/"inputs.npz"))
    for b in ("cupy", "jax"):
        run(b, "route", str(tmp/"inputs.npz"), str(tmp/(b + ".npz")))
    return np.load(tmp/"inputs.npz"), np.load(tmp/"cupy.npz"), np.load(tmp/"jax.npz")


def rel(a, b):
    return float(np.max(np.abs(a - b)/np.abs(b)))


def test_workers_used_their_backends(routes):
    _, c, j = routes
    assert str(c['xp']) == 'cupy' and str(j['xp']) == 'numpy'


def test_simulated_data_step_agrees(routes):
    d, c, j = routes
    assert int(j['nres']) == int(c['nres']) == int(d['data_nres'])
    assert_array_equal(j['res_idx'], c['res_idx'])
    assert_allclose(j['fg'], c['fg'], rtol=1e-12, atol=0)
    print("\nsimulated data: N_res {} (both); res_idx identical ({} binaries); fg max rel diff {:.2e}".format(
        int(c['nres']), c['res_idx'].size, rel(j['fg'], c['fg'])))


def test_likelihood_terms_agree(routes):
    d, c, j = routes
    ## the NegBin log-gamma terms (~1e5) cancel to O(10): compare at their float64 precision
    nb_atol = 1e-14*np.max(gammaln(3 + d['nres'].sum(axis=0) + int(d['data_nres'])))
    assert_allclose(j['ln_fg'], c['ln_fg'], rtol=1e-12)
    assert_allclose(j['ln_nres'], c['ln_nres'], rtol=0, atol=nb_atol)
    assert_allclose(j['ln_res'], c['ln_res'], rtol=1e-12)
    tot_j = j['ln_fg'] + j['ln_nres'] + j['ln_res']
    tot_c = c['ln_fg'] + c['ln_nres'] + c['ln_res']
    assert_allclose(tot_j, tot_c, rtol=1e-12, atol=nb_atol)
    print("\nlikelihood max rel diffs: fg {:.2e}, N_res {:.2e} (abs {:.2e}), resolved {:.2e}, total {:.2e}".format(
        rel(j['ln_fg'], c['ln_fg']), rel(j['ln_nres'], c['ln_nres']), float(np.max(np.abs(j['ln_nres'] - c['ln_nres']))),
        rel(j['ln_res'], c['ln_res']), rel(tot_j, tot_c)))
