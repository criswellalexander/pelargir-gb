# pelargir-gb
Population Estimation for LISA in A Reverse-jump Global Inference Regime (PELARGIR).  A prototype for population inference for LISA in a Global Fit setting. Prototype implementation of the formalism described in [arXiv:2604.03390](https://arxiv.org/abs/2604.03390)

## Installation

From a clone of the repository:

```bash
pip install -e .              # numpy backend only
pip install -e ".[dev]"       # everything: JAX, flows (torch, zuko), pytest, and CUDA 12 builds of cupy and JAX
```

Extras can also be combined individually: `jax`, `flows`, `test`, and one of `cuda12` / `cuda13` (cupy and JAX GPU builds for that CUDA major version; the two conflict, so on a CUDA-13-only machine use `".[jax,flows,test,cuda13]"` instead of `dev`). In a conda/mamba environment that already provides cupy and a CUDA-enabled JAX, install with `--no-deps` so pip does not replace them.

## Usage

The array backend (`numpy`, `cupy` or `jax`) is chosen once per process, before any other pelargir module is imported:

```python
from pelargir import backend
backend.set_backend("jax")          # or set PELARGIR_BACKEND=jax
from pelargir.models import PopModel
```

Command-line tools installed with the package:

- `pelargir-run`: population inference with Eryn (`pelargir/scripts/run_pelargir.py`)
- `pelargir-make-flow-set`, `pelargir-train-flows`, `pelargir-validate-flows`: the flow emulator's training set, training and validation.

Tests: `pytest` from the repository root.
