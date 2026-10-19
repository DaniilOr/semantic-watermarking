#!/bin/bash
# one node of the 2-node Qwen SemStamp-rest job: CUDA + file preflight, then 3 of the 6 shards on this node's GPU
set -uo pipefail; cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
N=${SLURM_NODEID:?}; NS=${NUM_SHARDS:-6}; PER=${PER_NODE:-3}
echo "node $N host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start=$(date)"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED on $(hostname)"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY on $(hostname)"; exit 5; }
$PY -c "
import json
from pathlib import Path
m = json.loads(Path('assets/ss_models.json').read_text())['qwen08b']['path']
for f in [Path(m, 'config.json'), *Path(m).glob('*.safetensors'),
          Path('/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282/pytorch_model.bin'),
          Path('data/sp_bench/prompts.jsonl')]:
    with open(f, 'rb') as fh: fh.read(1 << 20)
print('files readable')" || { echo "FILE PREFLIGHT FAILED on $(hostname)"; exit 3; }
C="$PY scripts/steered_semstamp.py --run-name sp_bench_qwen08b --target qwen08b --prompts sp_bench --ob-domains sp_realnews sp_booksum sp_tifu --max-new 200 --W 16"
P=(); for l in $(seq 0 $((PER - 1))); do k=$((N * PER + l)); $C --arm-label semstamp --arm semstamp --shard $k --num-shards $NS > slurm_logs/sp_qwen08b_semstamp_rest_$k.log 2>&1 & P+=($!); done
FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
echo "node $N end=$(date) fail=$FAIL"; exit $FAIL
