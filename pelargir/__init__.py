"""
PELARGIR: Population Estimation for LISA in A Reverse-jump Global Inference Regime (arXiv:2604.03390).

Submodules are not imported here: importing any of them binds the array backend, so choose it
first if you want something other than numpy:

    from pelargir import backend
    backend.set_backend("cupy")        # or "jax"; or set PELARGIR_BACKEND
    from pelargir.models import PopModel
"""
__version__ = "0.3.0"
