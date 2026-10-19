#!/bin/bash
#SBATCH --job-name=wm_multistep
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# one prompt -> spans A, B, C: three sequential 256-token steps of the SAME answer; controller restarts each span
$PY scripts/multistep_gen.py --config $CFG --name multistep --steps 3 --step-tokens 256 --beta 30 || exit 1
# detection per span and on the whole answer, each against its own clean null
$PY scripts/cross_detect.py --config $CFG --run runs/multistep_turns --decoys 40
$PY scripts/cross_detect.py --config $CFG --run runs/multistep_full --decoys 40 --prefixes 256,512
echo "end=$(date)"
