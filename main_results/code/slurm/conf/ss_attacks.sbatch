#!/bin/bash
#SBATCH --job-name=ss_attacks
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
ARMS=$($PY -c "import glob,re; print(','.join(sorted({re.search(r'generations\.(.+?)\.shard', f).group(1) for f in glob.glob('runs/ss_llama3b/generations.*.jsonl')})))")
echo "arms: $ARMS"
$PY scripts/reorder_attack.py --run runs/ss_llama3b --arms $ARMS --attacks reorder_sent,resegment
$PY scripts/api_paraphrase.py --run runs/ss_llama3b --arms $ARMS --level light --workers 8
$PY scripts/api_paraphrase.py --run runs/ss_llama3b --arms $ARMS --level strong --workers 8
echo "end=$(date)"
