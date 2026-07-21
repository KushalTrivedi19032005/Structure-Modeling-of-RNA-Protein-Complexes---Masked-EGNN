#!/bin/bash
#SBATCH --job-name=egnn-emb
#SBATCH --output=logs/%x_%j.out       # stdout -> logs/egnn-emb_<jobid>.out
#SBATCH --error=logs/%x_%j.err        # stderr
#SBATCH --partition=u22               # the only partition on turing
#SBATCH --gres=gpu:1                  # transformers -> use a GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     # ESM2/RNA-FM need more headroom than training
#SBATCH --time=06:00:00               # walltime HH:MM:SS

set -euo pipefail

# Run from the directory sbatch was launched in (should be the `code/` folder).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs embeddings

# --- Environment --------------------------------------------------------------
# module load python/3.10
# module load cuda/11.8

if [ ! -d "../venv" ]; then
    python3 -m venv ../venv
    source ../venv/bin/activate
    # Use `python -m pip` (not the bare `pip` shim) and --no-cache-dir so the big
    # torch wheel streams to disk instead of being buffered in memory (which
    # MemoryError'd on the login node). This build runs on the compute node.
    python -m pip install --upgrade pip
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
    python -m pip install --no-cache-dir -r requirements.txt
else
    source ../venv/bin/activate
fi

# Extra deps needed only for embedding extraction (idempotent -> always run).
# ptflops is pulled in by RNA-FM's package init; --no-deps avoids re-fetching torch.
python -m pip install --no-cache-dir fair-esm
python -m pip install --no-cache-dir --no-deps ptflops

python compute_embeddings.py \
    --pdb_dir ./PDB-CIF \
    --af_dir ./AlphaFold-CIF \
    --esm_weights ../ESM-2/esm2_t30_150M_UR50D.pt \
    --rna_fm_dir ../RNA-FM \
    --rna_fm_weights ../RNA-FM/redevelop/pretrained/RNA-FM_pretrained.pth \
    --out_dir ./embeddings

echo "Done."
