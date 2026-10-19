#!/bin/bash
#SBATCH --job-name=wm_trade_post
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
NULL=runs/null_calibration/eval/posstats_randomwords.npz
echo "host=$(hostname) start=$(date)"
# after both shards: per-position dump, detector analysis against the frozen null, proxy metrics
$PY scripts/posstats_dump.py --config $CFG --run runs/tradeoff --key-set randomwords || exit 1
[ -f $NULL ] || { echo "frozen null missing: run wm_dumps first"; exit 1; }
$PY scripts/detector_variants.py --null $NULL --run runs/tradeoff --prefixes 128,256,384 --out runs/detector_variants_tradeoff.json
$PY -m wmworld.cli evaluate --config $CFG --run runs/tradeoff --no-decoys
echo "end=$(date)"
