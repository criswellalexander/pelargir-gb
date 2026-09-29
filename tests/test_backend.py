"""
Backend selection (pelargir/backend.py). The backend binds once per process, so each
case runs in a fresh subprocess with the PELARGIR_* variables cleared.
"""
import os
import subprocess
import sys

import pytest

PELARGIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pelargir")


def run(code, **env_vars):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PELARGIR_")}
    env.update(env_vars)
    prelude = "import sys; sys.path.insert(0, {!r})\n".format(PELARGIR_DIR)
    return subprocess.run([sys.executable, "-c", prelude + code], env=env,
                          capture_output=True, text=True, timeout=120)


def cupy_gpu_available():
    res = run("import cupy; assert cupy.cuda.is_available()")
    return res.returncode == 0


def test_default_is_numpy():
    res = run("import models; import backend; print('BACKEND=' + backend.BACKEND, backend.xp.__name__, backend.GPU)")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=numpy numpy False" in res.stdout


def test_env_var_selects_backend():
    res = run("import backend; print('BACKEND=' + backend.BACKEND)", PELARGIR_BACKEND="NumPy")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=numpy" in res.stdout


def test_unknown_backend_raises():
    res = run("import models", PELARGIR_BACKEND="foo")
    assert res.returncode != 0
    assert "Unknown pelargir backend 'foo'" in res.stderr


def test_stale_pelargir_gpu_raises():
    res = run("import models", PELARGIR_GPU="1")
    assert res.returncode != 0
    assert "PELARGIR_GPU is no longer used" in res.stderr


def test_set_backend_after_initialization_raises():
    res = run("import models, backend; backend.set_backend('cupy')")
    assert res.returncode != 0
    assert "already initialized as 'numpy'" in res.stderr


def test_set_backend_same_name_after_initialization_is_a_no_op():
    res = run("import models, backend; backend.set_backend('numpy'); print('ok')")
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


@pytest.mark.skipif(not cupy_gpu_available(), reason="cupy with a CUDA GPU is not available")
def test_jax_backend_leaves_cupy_able_to_compile_kernels():
    """JAX initializing CUDA before cupy has loaded NVRTC breaks later cupy kernel compiles."""
    pytest.importorskip("jax")
    res = run("import backend; xp = backend.xp; import jax\n"
              "print('devices', jax.devices())\n"
              "print('kernel', float((xp.exp(xp.arange(4.0))*3).sum()))", PELARGIR_BACKEND="jax")
    assert res.returncode == 0, res.stderr
    assert "kernel" in res.stdout


@pytest.mark.skipif(not cupy_gpu_available(), reason="cupy with a CUDA GPU is not available")
def test_set_backend_before_import_selects_cupy():
    res = run("import backend; backend.set_backend('cupy')\n"
              "import models; print('BACKEND=' + backend.BACKEND, models.xp.__name__, backend.GPU)")
    assert res.returncode == 0, res.stderr
    assert "BACKEND=cupy cupy True" in res.stdout
