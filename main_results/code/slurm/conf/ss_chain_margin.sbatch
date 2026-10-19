#!/bin/bash
#SBATCH --job-name=ss_chain_margin
#SBATCH --time=08:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
[ -f /l/users/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/config.json ] || { echo "3B NOT VISIBLE"; exit 6; }
# Amendment 9: chain with a soft margin 0.02 in the stop test (no push); two experiments, one per GPU, 3 shards each
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P="$PY scripts/steered_semstamp.py --arm chain --W 16"
lane() {  # lane <gpu> <logtag> <args...>: 3 shards of one experiment on one GPU
  local gpu=$1 tag=$2; shift 2
  for I in 0 1 2; do CUDA_VISIBLE_DEVICES=$gpu $P "$@" --shard $I --num-shards 3 > slurm_logs/ss_chain_prod_${tag}_${I}.log 2>&1 & done
}
lane ${G[0]} mseg  --run-name ss_llama3b --unit seg  --block 4 --best-of 16 --stop-margin 0.02
lane ${G[1]} msent --run-name ss_llama3b --unit sent --block 4 --best-of 16 --stop-margin 0.02
wait
grep -h "done (" slurm_logs/ss_chain_prod_m{seg,sent}_[012].log
echo "end=$(date)"
