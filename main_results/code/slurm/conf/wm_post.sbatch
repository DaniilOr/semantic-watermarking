#!/bin/bash
#SBATCH --job-name=wm_post
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
NULL=runs/null_calibration/eval/posstats_randomwords.npz
echo "host=$(hostname) start=$(date)"
# post-processing of the span run (same key per span vs rotating keys): dumps (with rotation partner phrases), judge, analysis
$PY scripts/posstats_dump.py --config $CFG --run runs/spans --key-set randomwords --extra steering_decoys || exit 1
[ -f $NULL ] || { echo "frozen null missing: run wm_dumps first"; exit 1; }
$PY scripts/detector_variants.py --null $NULL --run runs/spans --prefixes 256,512,768 \
    --config $CFG --rotation --out runs/detector_variants_long.json
echo "end=$(date)"
