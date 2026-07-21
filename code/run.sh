#!/bin/bash
#SBATCH --job-name=egnn-refine
#SBATCH --output=logs/%x_%j.out      # stdout -> logs/egnn-refine_<jobid>.out
#SBATCH --error=logs/%x_%j.err       # stderr
#SBATCH --partition=u22              # the only partition on turing
#SBATCH --gres=gpu:1                 # request 1 GPU (turing nodes have gpu:4, node10 gpu:8)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00              # walltime HH:MM:SS

set -euo pipefail

# --- Move to the directory sbatch was run from (should be the `code/` folder) -
# NOTE: in a SLURM batch job $0 points at a spool copy, not your script, so we
# use SLURM_SUBMIT_DIR (set to wherever you ran `sbatch`). Falls back to the
# script dir when run directly with `bash run.sh`.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# --- Environment --------------------------------------------------------------
# Uncomment / adjust the module loads for your cluster:
# module load python/3.10
# module load cuda/11.8

# Create a venv once, reuse it afterwards.
if [ ! -d "../venv" ]; then
    python3 -m venv ../venv
    source ../venv/bin/activate
    pip install --upgrade pip
    # Install a CUDA-12.x torch FIRST so pip doesn't pull the default CUDA-13
    # build, which the node's 12.8 driver can't run (falls back to CPU).
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
else
    source ../venv/bin/activate
fi

# --- Run ----------------------------------------------------------------------
# main.py sweeps radius x n_layers internally -> one call runs all 9 trials.
python main.py \
    --pdb_dir ./PDB-CIF \
    --af_dir ./AlphaFold-CIF \
    --epochs 3 \
    --lr 1e-3 \
    --n_layers 4 6 8 \
    --batch_size 1 \
    --hidden_nf 64 \
    --radius 8 10 12 \
    --mask_schedule 0.10 \
    --use_lm_emb \
    --emb_dir ./embeddings \
    --seed 42

echo "Done."
