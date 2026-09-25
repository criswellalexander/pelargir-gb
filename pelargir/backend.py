"""
Array-backend selection for pelargir.

The backend is one of "numpy", "cupy", or "jax" and is chosen once per process,
before any other pelargir module is imported (those modules bind `xp` at import):

    os.environ["PELARGIR_BACKEND"] = "cupy"      # or: export PELARGIR_BACKEND=cupy
    # or
    import backend; backend.set_backend("cupy")

The default is "numpy". Under "jax", `xp` is cupy; JAX is used only for the JAX
thresholder. cupy must load its NVRTC (compile a kernel) before JAX initializes
its CUDA backend: otherwise JAX's bundled CUDA libraries shadow cupy's and cupy
kernel compilation fails. The "jax" setup below does this; code that uses cupy
and jax together outside pelargir must do the same.

Exports (resolved on first access): BACKEND, xp, xsc (scipy.special or
cupyx.scipy.special), GPU (True for "cupy" and "jax").
"""
import os

_VALID = ("numpy", "cupy", "jax")
_EXPORTS = ("BACKEND", "xp", "xsc", "GPU")
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

    if name == "numpy":
        import numpy as xp
        import scipy.special as xsc
    else:
        os.environ["SCIPY_ARRAY_API"] = "1"
        try:
            import cupy as xp
            from cupyx.scipy import special as xsc
        except ImportError as err:
            raise ImportError("pelargir backend {!r} requires cupy, which could not be imported.".format(name)) from err
        if not xp.cuda.is_available():
            raise RuntimeError("pelargir backend {!r} requires a CUDA GPU, but cupy reports none.".format(name))
        if name == "jax":
            ## compile one cupy kernel so cupy loads its own NVRTC before JAX initializes CUDA
            xp.arange(2, dtype=xp.float64).sum()
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            try:
                import jax
            except ImportError as err:
                raise ImportError("pelargir backend 'jax' requires jax, which could not be imported.") from err
            jax.config.update("jax_enable_x64", True)
            if not any(dev.platform == "gpu" for dev in jax.devices()):
                raise RuntimeError("pelargir backend 'jax' requires a GPU-enabled jax; found devices {}.".format(jax.devices()))

    print("Running Pelargir population inference with the {} backend.".format(name))
    return {"BACKEND": name, "xp": xp, "xsc": xsc, "GPU": name in ("cupy", "jax")}


def __getattr__(attr):
    global _state
    if attr in _EXPORTS:
        if _state is None:
            _state = _initialize()
        return _state[attr]
    raise AttributeError("module 'backend' has no attribute {!r}".format(attr))
