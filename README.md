# pelargir-gb
Population Estimation for LISA in A Reverse-jump Global Inference Regime (PELARGIR).  A prototype for population inference for LISA in a Global Fit setting. Prototype implementation of the formalism described in [arXiv:2604.03390](https://arxiv.org/abs/2604.03390)

## Installation

From a clone of the repository:

```bash
pip install -e .              # numpy backend only
pip install -e ".[dev]"       # jax backend on a CUDA 12+ GPU, flows (torch, zuko) and pytest; no cupy
```

Extras:

| Extra | Installs | For |
|---|---|---|
| `jax` | jax (CPU) | the jax backend on the CPU |
| `jax-cuda12`, `jax-cuda13` | jax with CUDA 12 / 13 | the jax backend on a GPU |
| `cupy-cuda12`, `cupy-cuda13` | cupy-cuda12x / cupy-cuda13x | the cupy backend (these two conflict) |
| `flows` | torch, zuko | the zuko flow emulator (`pelargir.flows.zuko_base`) |
| `flax` | flax, distrax, optax | the JAX flow emulator (`pelargir.flows.flax_base`), differentiable for NUTS |
| `test` | pytest | the test suite |
| `dev` | `jax`, `jax-cuda12`, `flows`, `flax`, `test` | development |

The jax backend does not need cupy: its array glue is host numpy and the JAX kernels run on JAX's default device (with a warning if JAX finds no GPU). In a conda/mamba environment that already provides cupy and a CUDA-enabled JAX, install with `--no-deps` so pip does not replace them.

## Usage

The array backend (`numpy`, `cupy` or `jax`) is chosen once per process, before any other pelargir module is imported:

```python
from pelargir import backend
backend.set_backend("jax")          # or set PELARGIR_BACKEND=jax
from pelargir.models.population import PopModel
```

Package layout:

- `pelargir.models`: the population model (`population.PopModel`) and the SNR thresholders (`thresholding`, `jax_thresholding`)
- `pelargir.sim`: the JAX forward model (`jax_population`)
- `pelargir.inference`: the analytic priors and likelihoods (`analytic_inference`), their JAX port (`jax_likelihood`), the probability distributions, and the hierarchical flow-emulator prior and the posterior built on it (`flow_inference`) with its samplers (`flow_hmc`, `flow_eryn`, `flow_gibbs`)
- `pelargir.flows`: the flow emulators' data layer (`data`) and the zuko and flax flow bases (`zuko_base`, `flax_base`)
- `pelargir.sampler_utils`: Eryn helpers (`backends.SupplementalBackend`, `moves`)
- top level: `backend` (array backend), `utils`, `plotting`, and the command-line `scripts`

Command-line tools installed with the package:

- `pelargir-run`: population inference with Eryn (`pelargir/scripts/run_pelargir.py`)
- `pelargir-make-flow-set`, `pelargir-train-flows`, `pelargir-validate-flows`: the flow emulator's training set, training and validation. Training takes `-f/--flow-base zuko|flax`; the flax base's architecture is drawn in `docs/flax_flow_architecture.svg`.

Every script logs timestamped progress to stderr at INFO, with progress bars on its long loops; results go to stdout. `-v/--verbose` adds DEBUG messages (e.g. every training epoch), and `-q/--quiet` keeps warnings only and hides pelargir's bars.

Tests: `pytest` from the repository root.
