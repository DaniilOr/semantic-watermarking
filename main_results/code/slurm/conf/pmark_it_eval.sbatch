#!/bin/bash
#SBATCH --job-name=pmark_it_eval
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# task quality of the PMark instruct answers (GSM8K accuracy, IFEval compliance), same scorer as for plain / SemStamp / push
cd /home/daniil.orel/wmworld
/home/daniil.orel/anaconda3/envs/wmworld/bin/python scripts/it_eval.py runs/pm_it_bench_llama3b
