#!/bin/bash
#SBATCH --job-name=confw_reg_post
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "##### REGISTERED HELD-OUT PROMPTS, WINDOWED detector + windowed null #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_confirmation_w64.npz --key-set confirmation --dump-suffix _w64 \
    --run runs/conf_window_reg --variants T_clip05_gate,T --prefixes 128,256,384 --out runs/detector_variants_conf_window_reg.json
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/conf_window_reg --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/conf_window_reg --arms wm_w64_c3 --workers 8
$PY scripts/quality_vs_clean.py --clean runs/conf_cross:base --clean runs/conf_pos:base --marked runs/conf_window_reg --marked runs/conf_cross \
    --out runs/quality_vs_clean_conf_window_reg.json
