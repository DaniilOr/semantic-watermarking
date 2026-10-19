#!/bin/bash
#SBATCH --job-name=ifeval_window_post
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "##### IFEval, WINDOWED detector + windowed null #####"
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_confirmation_w64.npz --key-set confirmation --dump-suffix _w64 \
    --run runs/ifeval_cross --variants T_clip05_gate,T --prefixes 128,256 --out runs/detector_variants_ifeval_window.json
$PY -c "
import json
r = json.load(open('runs/ifeval_cross/results.json'))
for arm, x in r['arms'].items():
    if x.get('strict'): print('IFEVAL CHECKER', arm, 'strict prompt', round(x['strict']['prompt_level'], 3), 'strict instr', round(x['strict']['instruction_level'], 3), 'loose prompt', round(x['loose']['prompt_level'], 3), 'n', x['strict']['n_prompts'])
"
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/ifeval_cross --arms wm_w64_c3 --workers 8
