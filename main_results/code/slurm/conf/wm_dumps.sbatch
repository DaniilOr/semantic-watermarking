#!/bin/bash
#SBATCH --job-name=wm_dumps
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
# 1) ONE frozen clean null (1,200 texts) + per-position dumps of the finished cross-model runs
$PY scripts/posstats_dump.py --config $CFG --run runs/null_calibration --key-set randomwords || exit 1
$PY scripts/posstats_dump.py --config $CFG --run runs/cross_llama --run runs/cross_long --run runs/cross_long2 --key-set randomwords || exit 1
# 2) which text-only detector is best?  (offline arrays; selection happens on these DEV runs only)
$PY scripts/detector_variants.py --null $NULL --run runs/cross_llama --run runs/cross_long --run runs/cross_long2 \
    --prefixes 128,256,512 --out runs/detector_variants_dev.json
# 4) the same null under the 16 held-out confirmation keys, for the confirmation experiment
$PY scripts/posstats_dump.py --config $CFG --run runs/null_calibration --key-set confirmation
echo "end=$(date)"
