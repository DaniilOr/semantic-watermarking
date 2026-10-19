#!/bin/bash
#SBATCH --job-name=w2_attack_score
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY (foreign process on the allocated GPU)"; exit 5; }
CFG=configs/pilot.yaml
NULL=runs/null_calibration/eval/posstats_randomwords.npz
NULLC=runs/null_calibration/eval/posstats_confirmation.npz
ARM=$(cat runs/chosen_arm.txt 2>/dev/null || true)
echo "host=$(hostname) start=$(date) chosen arm=${ARM:-<none yet>}"
[ -n "$ARM" ] && [ -f $NULLC ] || { echo "selection or confirmation null missing"; exit 1; }
# score the paraphrased confirmation texts with the UNCHANGED detector and thresholds
$PY scripts/posstats_dump.py --config $CFG --run runs/conf_cross_para_light --run runs/conf_cross_para_strong --key-set confirmation || exit 1
$PY scripts/detector_variants.py --null $NULLC --key-set confirmation --run runs/conf_cross_para_light --run runs/conf_cross_para_strong --out runs/detector_variants_attacks.json
echo "end=$(date)"
