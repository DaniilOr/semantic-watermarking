#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Qwen-0.8B SemStamp arm of runs/sp_bench_qwen08b without MBPP, as sp_gen_qwen_rest.sbatch, but on 1 GPU, 4 shards (the
# cluster had no allowed node with 2 free GPUs; gpu-51 and gpu-54 fail CUDA init).  Any node failing its preflight or a
# shard fails the job, so afterok dependents never run on a partial arm.
set -uo pipefail
cd /home/daniil.orel/wmworld
echo "start=$(date) nodes=$SLURM_JOB_NODELIST"
NUM_SHARDS=4 PER_NODE=4 srun --ntasks=1 bash slurm/conf/qwen_rest_node.sh
RC=$?; echo "end=$(date) rc=$RC"; exit $RC
