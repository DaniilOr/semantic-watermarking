#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface HF_HUB_CACHE=/l/users/daniil.orel/hf_cache/hub TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# Perplexity (SemStamp's definition, OPT-2.7B; scripts/ppl_eval.py) of every arm of the continuation runs, KGW and k-means runs.
ARMS="base clean_seed1 semstamp chain_reference kgw ksemstamp chain_kmeans"
A="runs/sp_bench_llama3b runs/sp_bench_llama1b runs/sp_bench_qwen08b_nl runs/ob_bench runs/ob_bench_llama1b runs/ob_bench_qwen08b_nl runs/semstamp_setting_opt13b_ss
   runs/kgw_sp_bench_llama3b runs/kgw_sp_bench_llama1b runs/kgw_sp_bench_qwen08b_nl runs/kgw_ob_bench runs/kgw_ob_bench_llama1b runs/kgw_ob_bench_qwen08b_nl runs/kgw_semstamp_setting_opt13b_ss
   runs/ks_setting_opt13b_ss runs/ks_sp_bench_llama3b runs/ks_sp_bench_llama3b_bs runs/ks_sp_bench_llama1b runs/ks_sp_bench_qwen08b_nl"
B="runs/sp_bench_llama3b_base runs/sp_bench_llama1b_base runs/sp_bench_qwen08b_base_nl runs/ob_bench_llama3b_base runs/ob_bench_llama1b_base runs/ob_bench_qwen08b_base_nl runs/semstamp_setting_llama3b_base
   runs/kgw_sp_bench_llama3b_base runs/kgw_sp_bench_llama1b_base runs/kgw_sp_bench_qwen08b_base_nl runs/kgw_ob_bench_llama3b_base runs/kgw_ob_bench_llama1b_base runs/kgw_ob_bench_qwen08b_base_nl runs/kgw_semstamp_setting_llama3b_base"
echo "host=$(hostname) start=$(date)"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/ppl_eval.py --runs $A --arms $ARMS --out runs/ppl_part_a.json > slurm_logs/ppl_a.log 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/ppl_eval.py --runs $B --arms $ARMS --out runs/ppl_part_b.json > slurm_logs/ppl_b.log 2>&1 & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
echo "exit a=$R1 b=$R2 end=$(date)"; [ $R1 = 0 ] && [ $R2 = 0 ]
