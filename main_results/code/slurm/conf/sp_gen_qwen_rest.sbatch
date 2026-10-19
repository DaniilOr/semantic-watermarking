#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-05,gpu-02,gpu-51
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# Qwen-0.8B SemStamp arm of runs/sp_bench_qwen08b, resumed WITHOUT MBPP (user go 2026-09-29: MBPP is redone under the code
# protocol, runs/code_mbpp_*).  Same command as sp_gen_model.sbatch otherwise; finished texts are skipped by id
# (ids verified: params hash e6726924b959 with the /l -> /nfs-stor alias).
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# gpu-54 is allowed again (2026-09-29): check this node reads the model, encoder and data before generating
$PY -c "
import json, sys
from pathlib import Path
m = json.loads(Path('assets/ss_models.json').read_text())['qwen08b']['path']
for f in [Path(m, 'config.json'), *Path(m).glob('*.safetensors'),
          Path('/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282/pytorch_model.bin'),
          Path('data/sp_bench/prompts.jsonl')]:
    with open(f, 'rb') as fh: fh.read(1 << 20)
print('files readable')" || { echo "FILE PREFLIGHT FAILED on $(hostname)"; exit 3; }
C="$PY scripts/steered_semstamp.py --run-name sp_bench_qwen08b --target qwen08b --prompts sp_bench --ob-domains sp_realnews sp_booksum sp_tifu --max-new 200 --W 16"
P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $C --arm-label semstamp --arm semstamp --shard $k --num-shards 6 > slurm_logs/sp_qwen08b_semstamp_rest_$k.log 2>&1 & P+=($!); done; done
FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
[ $FAIL = 0 ] || { echo "A SHARD FAILED ($(date))"; exit 8; }            # afterok dependents (evals, next link) must not run on a partial arm
echo "semstamp done ($(date))"; echo "end=$(date)"
