#!/bin/bash
#SBATCH --job-name=long_analyse
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# detection vs length, against a null of EQUALLY LONG clean texts; then both judge runs
$PY scripts/detector_variants.py --null runs/long_null/eval/posstats_len10.npz --key-set len10 --run runs/long_story \
    --variants T_clip05_gate,T_clip1,T --prefixes 256,512,768,1024,1536 --out runs/detector_variants_long_story.json
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/long_story --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/long_story --workers 8
echo "end=$(date)"
