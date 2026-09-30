"""
Array-backend selection for pelargir.

The backend is one of "numpy", "cupy", or "jax" and is chosen once per process,
before any other pelargir module is imported (those modules bind `xp` at import):

    os.environ["PELARGIR_BACKEND"] = "cupy"      # or: export PELARGIR_BACKEND=cupy
    # or
    from pelargir import backend; backend.set_backend("cupy")

The default is "numpy". Under "jax", `xp` is host numpy and the JAX modules (forward
model, thresholder, likelihood, flows' simulator) run on JAX's default device, a GPU
if JAX has one (otherwise a warning is issued and JAX runs on the CPU); cupy is not
needed. Under "cupy", `xp` is cupy on a CUDA GPU.

JAX kernels can also be used under the cupy backend (arrays pass by DLPack). cupy
must then load its NVRTC (compile a kernel) before JAX initializes its CUDA backend,
otherwise JAX's bundled CUDA libraries shadow cupy's and cupy kernel compilation
fails; import_jax() does this, and code that uses cupy and jax together outside
pelargir must do the same.

Exports (resolved on first access): BACKEND, xp, xsc (scipy.special or
cupyx.scipy.special), CUPY_GPU (True only for "cupy", i.e. when xp arrays live on
the GPU).
"""
import os
import warnings

_VALID = ("numpy", "cupy", "jax")
_EXPORTS = ("BACKEND", "xp", "xsc", "CUPY_GPU")
_requested = None
_state = None


def _validate(name):
    name = str(name).lower()
    if name not in _VALID:
        raise ValueError("Unknown pelargir backend {!r}; choose one of {}.".format(name, _VALID))
    return name


def set_backend(name):
    '''
    Choose the array backend. Must be called before any other pelargir module is imported.

    Arguments
    -----------
    name (str) : "numpy", "cupy", or "jax".
    '''
    global _requested
    name = _validate(name)
    if _state is not None:
        if _state["BACKEND"] != name:
            raise RuntimeError("pelargir backend already initialized as {!r}; call set_backend({!r}) "
                               "before importing other pelargir modules.".format(_state["BACKEND"], name))
        return
    _requested = name


def _initialize():
    if _requested is not None:
        name = _requested
    elif "PELARGIR_BACKEND" in os.environ:
        name = _validate(os.environ["PELARGIR_BACKEND"])
    elif "PELARGIR_GPU" in os.environ:
        raise RuntimeError("PELARGIR_GPU is no longer used; set PELARGIR_BACKEND to one of {} "
                           "(or call backend.set_backend) instead.".format(_VALID))
    else:
        name = "numpy"

    if name == "cupy":
        os.environ["SCIPY_ARRAY_API"] = "1"
        try:
            import cupy as xp
            from cupyx.scipy import special as xsc
        except ImportError as err:
            raise ImportError("pelargir backend 'cupy' requires cupy, which could not be imported.") from err
        if not xp.cuda.is_available():
            raise RuntimeError("pelargir backend 'cupy' requires a CUDA GPU, but cupy reports none.")
    else:
        import numpy as xp
        import scipy.special as xsc
        if name == "jax":
            jax = _load_jax()
            if not any(dev.platform == "gpu" for dev in jax.devices()):
                warnings.warn("pelargir backend 'jax' found no GPU; JAX will run on {}.".format(jax.devices()),
                              stacklevel=2)

    print("Running Pelargir population inference with the {} backend.".format(name))
    return {"BACKEND": name, "xp": xp, "xsc": xsc, "CUPY_GPU": name == "cupy"}


def _load_jax(cupy_module=None):
    if cupy_module is not None:
        ## compile one cupy kernel so cupy loads its own NVRTC before JAX initializes CUDA
        cupy_module.arange(2, dtype=cupy_module.float64).sum()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax
    except ImportError as err:
        raise ImportError("pelargir's JAX functionality requires jax, which could not be imported.") from err
    jax.config.update("jax_enable_x64", True)
    ## prefix-stable draws: the first N values of a shape-(n,) draw don't depend on n,
    ## so a galaxy's binaries are the same at any padded size (jax_population.py)
    jax.config.update("jax_threefry_partitionable", True)
    return jax


def import_jax():
    '''
    Import jax configured for pelargir (x64, no GPU memory preallocation). Under the cupy
    backend, cupy loads its NVRTC first. Use this instead of importing jax directly.
    '''
    xp = __getattr__("xp")
    return _load_jax(xp if xp.__name__ == "cupy" else None)


def __getattr__(attr):
    global _state
    if attr in _EXPORTS:
        if _state is None:
            _state = _initialize()
        return _state[attr]
    raise AttributeError("module 'backend' has no attribute {!r}".format(attr))
