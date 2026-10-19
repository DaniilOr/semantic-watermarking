#!/bin/bash
#SBATCH --job-name=baseline_setup3
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -m pip install --quiet openai tenacity
$PY -c "
import sys; sys.path.insert(0,'third_party/SemStamp')
from sbert_lsh_model import SBERTLSHModel
from sampling_lsh_utils import get_mask_from_seed, reject_close_generation
print('SemStamp modules import ok')
"
