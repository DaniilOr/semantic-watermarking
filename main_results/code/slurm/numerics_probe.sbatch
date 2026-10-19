#!/bin/bash
#SBATCH --job-name=wm_numerics
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/numerics_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
for c in bf16_fla bf16_torch fp32_torch fp32_fla; do
  echo "=== $c ==="; $PY /home/daniil.orel/wmworld/slurm/numerics_probe.py $c 2>&1 | grep -E "RESULT|Error|error" | cut -c1-2000
done
