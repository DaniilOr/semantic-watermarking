#!/bin/bash
#SBATCH --job-name=ob_detect2_submit
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Runs after the first detect pass (277621) ends, when the GPU QoS has a free submit slot, and submits the second
# detect pass (after the second attacks pass 277642, which covers the adapter arms).  Retries every 10 min.
cd /home/daniil.orel/wmworld
GPU="--job-name=ob_detect2 --partition=cscc-gpu-p --qos=cscc-gpu-qos --gres=gpu:1 --exclude=gpu-54,gpu-05,gpu-02 --cpus-per-task=8 --mem=64G --time=06:00:00"
for i in $(seq 1 60); do
  J=$(sbatch --parsable --dependency=afterok:277642 $GPU --export=ALL,CMD="python scripts/ob_detect.py" slurm/cpu.sbatch 2>&1)
  if [[ "$J" == *"ependency"* ]]; then       # 277642 already purged from the controller: submit without the dependency
    J=$(sbatch --parsable $GPU --export=ALL,CMD="python scripts/ob_detect.py" slurm/cpu.sbatch 2>&1)
  fi
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$(date) submitted second detect pass: $J"
    echo "Amendment 17 note ($(date '+%Y-%m-%d %H:%M')): second detect pass submitted as job $J (after attacks 277642)." >> docs/preregistration.md
    exit 0
  fi
  echo "$(date) attempt $i: $J"; sleep 600
done
exit 1
