#!/bin/bash
#SBATCH --job-name=confw_detect
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
echo "##### HELD-OUT, WINDOWED detector + windowed null #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_confirmation_w64.npz --key-set confirmation --dump-suffix _w64 \
    --run runs/conf_window --variants T_clip05_gate,T --prefixes 128,256,384 --out runs/detector_variants_conf_window.json
echo "##### HELD-OUT, WHOLE-TEXT detector on the same texts #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_confirmation.npz --key-set confirmation \
    --run runs/conf_window --variants T_clip05_gate,T --out runs/detector_variants_conf_window_full.json
