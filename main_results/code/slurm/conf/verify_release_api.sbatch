#!/bin/bash
#SBATCH --job-name=verify_release_api
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
W=/home/daniil.orel/wmworld/runs/_release_check/smoke
cd /home/daniil.orel/wmark-release
echo "host=$(hostname) start=$(date)"
$PY scripts/judge_preference.py --prompts $W/prompts.jsonl --run $W/run --workers 4 || exit 1
$PY scripts/judge_preference.py --prompts $W/prompts.jsonl --run $W/null --arms base --base $W/run:base --workers 4 || exit 1
$PY scripts/judge_flaws.py --prompts $W/prompts.jsonl --run $W/run --run $W/null --workers 4 || exit 1
$PY scripts/quality_vs_clean.py --clean $W/run:base --clean $W/null:base --marked $W/run --min-prompts 3 --out $W/quality_vs_clean.json || exit 1
$PY scripts/paraphrase.py --run $W/run --arms wm_b4 --level light --workers 4 || exit 1
ls -la $W $W/run $W/run_para_light
echo "API SMOKE: PASS"
echo "end=$(date)"
