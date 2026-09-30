"""
PELARGIR: Population Estimation for LISA in A Reverse-jump Global Inference Regime (arXiv:2604.03390).

Submodules are not imported here: importing any of them binds the array backend, so choose it
first if you want something other than numpy:

    from pelargir import backend
    backend.set_backend("jax")         # or "cupy"; or set PELARGIR_BACKEND
    from pelargir.models import PopModel

The jax backend needs jax but not cupy (host numpy glue, JAX kernels on JAX's default device);
the cupy backend needs cupy and a CUDA GPU.
"""
__version__ = "0.3.0"
