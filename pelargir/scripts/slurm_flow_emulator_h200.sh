#!/bin/bash
#SBATCH --job-name=pelargir-flows
#SBATCH -e error-pelargir.lisa
#SBATCH -o out-pelargir.lisa
#SBATCH --account=taylor_group_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# ==================================================================================================
# pelargir flow emulator on one H200: generate the training set with the JAX forward model, train
# one flow per band, and validate the flows against fresh simulator draws.
#
#   sbatch slurm_flow_emulator_h200.sh                                  # production run
#   sbatch --export=ALL,SMOKE=1 slurm_flow_emulator_h200.sh             # tiny end-to-end check
#   sbatch --export=ALL,OUTDIR=/path/run1,N_DRAWS=200000 slurm_flow_emulator_h200.sh
#   sbatch --export=ALL,FLOW_BASE=flax slurm_flow_emulator_h200.sh            # the flax flow base
#
# Edit the #SBATCH account/partition (and --gres, if ACCRE names the H200s differently) and the
# environment block below before the first submission.
#
# Everything goes to $OUTDIR. Each stage is skipped if its output already exists, and training-set
# generation saves every chunk as it completes, so resubmitting with the same OUTDIR resumes an
# interrupted or timed-out job (the result is identical to an uninterrupted run).
#
#   $OUTDIR/provenance/job_<id>/  per submission: git commit, status and diff, a tarball of pelargir/,
#                         GPU and package versions, config
#   $OUTDIR/chunks/       per-chunk simulator output (resume state)
#   $OUTDIR/train.npz     flows.TrainingSet (per-bin N_res and S_gw on fbins[1:], one row per galaxy)
#   $OUTDIR/emulator/     trained BandedFlowEmulator, losses.json, losses.png
#   $OUTDIR/validation/   validation.json and per-point plots
#   $OUTDIR/logs/         one log per stage (appended across resubmissions)
#
# Timing: locally (8 GB RTX 4070) the forward model below 1 mHz costs ~150 ms per galaxy at the
# default lambda_tot prior, so 1e5 draws x 5 realizations would take ~21 h there. H200 throughput
# is unmeasured: run SMOKE=1 first, then read the per-chunk ETA in logs/make.log.
# ==================================================================================================
set -euo pipefail

## ---- environment (edit for ACCRE) ----
## an environment with pelargir installed (pip install -e ".[dev]" in the checkout), e.g.
## ENV_SETUP="module load miniforge; source activate gwenv-1"
ENV_SETUP=${ENV_SETUP:-}
PYTHON=${PYTHON:-python}
## the checkout, for the provenance record only
PELARGIR_DIR=${PELARGIR_DIR:-$HOME/pelargir-gb}
## optional: directory holding a libnvrtc that cupy can load (the problem run_pelargir's --fixlib works around)
CUDA_LIB_DIR=${CUDA_LIB_DIR:-}

## ---- run configuration (override with sbatch --export=ALL,VAR=value) ----
OUTDIR=${OUTDIR:-$PWD/flow_emulator_run}
N_DRAWS=${N_DRAWS:-500000}     ## hyperprior draws
N_REAL=${N_REAL:-10}            ## realizations per draw
FMIN=${FMIN:-1e-4}
FMAX=${FMAX:-1e-3}
FBIN=${FBIN:-2e-5}
SEED=${SEED:-1}
CHUNK=${CHUNK:-2048}           ## draws per saved chunk
## galaxies x padded size per jitted batch; ~90 B of GPU memory per binary, so 8e8 is ~72 GB
MAX_BINARIES=${MAX_BINARIES:-8e8}
BINS_PER_BAND=${BINS_PER_BAND:-5}
FLOW_BASE=${FLOW_BASE:-zuko}      ## zuko (torch) or flax (JAX)
FLOW_DTYPE=${FLOW_DTYPE:-float64}  ## flax only
N_EPOCHS=${N_EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-3}
N_SIM=${N_SIM:-200}            ## simulator realizations per validation point
N_FLOW=${N_FLOW:-2000}         ## flow samples per validation point
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    N_DRAWS=16; N_REAL=2; CHUNK=8; N_EPOCHS=1; N_SIM=20; N_FLOW=200
    OUTDIR=${OUTDIR}_smoke
fi

eval "$ENV_SETUP"
if [ -n "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
## JAX's default allocator fragments across the many padded galaxy sizes; the scripts also default to this
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
PROV="$OUTDIR/provenance/job_${SLURM_JOB_ID:-local_$(date +%s)}"
mkdir -p "$PROV" "$OUTDIR/logs"

stamp() { echo "[$(date '+%F %T')] $*"; }

## ---- provenance ----
{
    echo "job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
    for v in OUTDIR N_DRAWS N_REAL FMIN FMAX FBIN SEED CHUNK MAX_BINARIES BINS_PER_BAND FLOW_BASE FLOW_DTYPE N_EPOCHS BATCH_SIZE LR \
             N_SIM N_FLOW SMOKE XLA_PYTHON_CLIENT_ALLOCATOR; do echo "$v=${!v}"; done
} > "$PROV/config.txt"
git -C "$PELARGIR_DIR" rev-parse HEAD > "$PROV/git_commit.txt"
git -C "$PELARGIR_DIR" status --short > "$PROV/git_status.txt"
git -C "$PELARGIR_DIR" diff HEAD > "$PROV/git_diff.patch"
## the diff misses untracked files, so also keep the code that ran
tar czf "$PROV/pelargir_source.tgz" -C "$PELARGIR_DIR" --exclude='__pycache__' --exclude='.ipynb_checkpoints' pelargir
nvidia-smi > "$PROV/nvidia_smi.txt" 2>&1 || true
"$PYTHON" -c "import numpy, scipy, jax, torch, zuko; print('numpy', numpy.__version__, '| scipy', scipy.__version__, \
'| jax', jax.__version__, '| torch', torch.__version__, torch.version.cuda, '| zuko', zuko.__version__)" \
    > "$PROV/versions.txt" 2>&1
stamp "config: $(tr '\n' ' ' < "$PROV/config.txt")"

## ---- 1. training set ----
if [ -f "$OUTDIR/train.npz" ]; then
    stamp "train.npz exists; skipping generation"
else
    stamp "generating $N_DRAWS x $N_REAL galaxies"
    "$PYTHON" -m pelargir.scripts.make_flow_training_set "$OUTDIR/train.npz" --chunk_dir "$OUTDIR/chunks" \
        --n_draws "$N_DRAWS" --n_real "$N_REAL" --fmin "$FMIN" --fmax "$FMAX" --fbin "$FBIN" --seed "$SEED" \
        --chunk "$CHUNK" --max_binaries_per_batch "$MAX_BINARIES" 2>&1 | tee -a "$OUTDIR/logs/make.log"
fi

## ---- 2. training ----
if [ -f "$OUTDIR/emulator/emulator.pt" ]; then
    stamp "emulator exists; skipping training"
else
    stamp "training"
    "$PYTHON" -m pelargir.scripts.train_flow_emulator "$OUTDIR/train.npz" "$OUTDIR/emulator" \
        --flow-base "$FLOW_BASE" --dtype "$FLOW_DTYPE" \
        --bins_per_band "$BINS_PER_BAND" --n_epochs "$N_EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
        --seed "$SEED" --device cuda 2>&1 | tee -a "$OUTDIR/logs/train.log"
fi

## ---- 3. validation ----
if [ -f "$OUTDIR/validation/validation.json" ]; then
    stamp "validation exists; skipping"
else
    stamp "validating"
    "$PYTHON" -m pelargir.scripts.validate_flow_emulator "$OUTDIR/emulator" "$OUTDIR/validation" \
        --n_sim "$N_SIM" --n_flow "$N_FLOW" --device cuda 2>&1 | tee -a "$OUTDIR/logs/validate.log"
fi
stamp "done: $OUTDIR"
