#!/bin/bash
#SBATCH --job-name=window_detect
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
echo "##### WINDOWED detector (last 64-95 tokens), windowed null #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_len10_w64.npz --key-set len10 --dump-suffix _w64 \
    --run runs/window_dev --run runs/strength_fill --run runs/keylen10 --variants T_clip05_gate,T --prefixes 128,256,384 --out runs/detector_variants_window_w64.json
echo "##### WHOLE-TEXT detector on the same texts #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_len10.npz --key-set len10 \
    --run runs/window_dev --variants T_clip05_gate,T --out runs/detector_variants_window_full.json
