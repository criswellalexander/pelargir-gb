"""
Backend selection (pelargir/backend.py). The backend binds once per process, so each
case runs in a fresh subprocess with the PELARGIR_* variables cleared.
"""
import os
import subprocess
import sys

import pytest

## the repository root, so the subprocesses import this checkout whether or not it is installed
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


def run(code, **env_vars):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PELARGIR_")}
    env.update(env_vars)
    prelude = "import sys; sys.path.insert(0, {!r})\n".format(REPO_DIR)
    return subprocess.run([sys.executable, "-c", prelude + code], env=env,
                          capture_output=True, text=True, timeout=120)


def cupy_gpu_available():
    res = run("import cupy; assert cupy.cuda.is_available()")
    return res.returncode == 0


def test_default_is_numpy():
    res = run("import pelargir.models as models; from pelargir import backend; print('BACKEND=' + backend.BACKEND, backend.xp.__name__, backend.CUPY_GPU)")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=numpy numpy False" in res.stdout


def test_env_var_selects_backend():
    res = run("from pelargir import backend; print('BACKEND=' + backend.BACKEND)", PELARGIR_BACKEND="NumPy")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=numpy" in res.stdout


def test_unknown_backend_raises():
    res = run("import pelargir.models", PELARGIR_BACKEND="foo")
    assert res.returncode != 0
    assert "Unknown pelargir backend 'foo'" in res.stderr


def test_stale_pelargir_gpu_raises():
    res = run("import pelargir.models", PELARGIR_GPU="1")
    assert res.returncode != 0
    assert "PELARGIR_GPU is no longer used" in res.stderr


def test_set_backend_after_initialization_raises():
    res = run("import pelargir.models as models; from pelargir import backend; backend.set_backend('cupy')")
    assert res.returncode != 0
    assert "already initialized as 'numpy'" in res.stderr


def test_set_backend_same_name_after_initialization_is_a_no_op():
    res = run("import pelargir.models as models; from pelargir import backend; backend.set_backend('numpy'); print('ok')")
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_jax_backend_uses_host_numpy_and_never_imports_cupy():
    pytest.importorskip("jax")
    res = run("import pelargir.models as models; from pelargir import backend, jax_likelihood\n"
              "print('BACKEND=' + backend.BACKEND, models.xp.__name__, backend.CUPY_GPU, 'cupy' in sys.modules)",
              PELARGIR_BACKEND="jax")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=jax numpy False False" in res.stdout


def test_jax_backend_runs_on_cpu_with_a_warning():
    pytest.importorskip("jax")
    code = ("import numpy as np, jax\n"
            "from pelargir import backend, jax_population as jp\n"
            "from pelargir.inference import GalacticBinaryPrior\n"
            "from pelargir.utils import lisa_noise_psd\n"
            "fb = np.arange(9e-5, 3e-4, 2e-5)\n"
            "N, fg = jp.jax_forward_model(jax.random.key(0), np.array([[0.6, 0.15, 3.31, 0.75, 0.33, 0.5]]),\n"
            "    np.array([[20000]]), jp.prior_bounds(GalacticBinaryPrior(None)), fb + 1e-5,\n"
            "    lisa_noise_psd(fb, cpu=True), np.ones(fb.size), 1.262e8, 5e4)\n"
            "print('devices', jax.devices(), 'Nres', int(N), 'fg', bool(np.all(np.isfinite(fg))))")
    res = run(code, PELARGIR_BACKEND="jax", JAX_PLATFORMS="cpu")
    assert res.returncode == 0, res.stderr
    assert "found no GPU" in res.stderr
    assert "CpuDevice" in res.stdout and "fg True" in res.stdout


@pytest.mark.skipif(not cupy_gpu_available(), reason="cupy with a CUDA GPU is not available")
def test_jax_under_the_cupy_backend_leaves_cupy_able_to_compile_kernels():
    """JAX initializing CUDA before cupy has loaded NVRTC breaks later cupy kernel compiles."""
    pytest.importorskip("jax")
    res = run("from pelargir import backend; backend.set_backend('cupy'); xp = backend.xp\n"
              "jax = backend.import_jax(); print('devices', jax.devices())\n"
              "print('kernel', float((xp.exp(xp.arange(4.0))*3).sum()))")
    assert res.returncode == 0, res.stderr
    assert "kernel" in res.stdout


@pytest.mark.skipif(not cupy_gpu_available(), reason="cupy with a CUDA GPU is not available")
def test_set_backend_before_import_selects_cupy():
    res = run("from pelargir import backend; backend.set_backend('cupy')\n"
              "import pelargir.models as models; print('BACKEND=' + backend.BACKEND, models.xp.__name__, backend.CUPY_GPU)")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=cupy cupy True" in res.stdout
