#!/bin/bash
#SBATCH --job-name=submit_v4
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Runs after the benchmark detections (278876, 278893) end, when the GPU submit limit has room, and queues:
#   v4a_bigjudge  GRPO from the base model with the larger reward judge (gemma-3-27b-it via OpenRouter), else as run 3a
#   distill_push  push distillation (slurm/conf/ss_distill_push.sbatch)
#   v4c_distill_bigjudge  GRPO from the distilled adapters with the larger judge, after distill_push
cd /home/daniil.orel/wmworld
J="--judge api --judge-model google/gemma-3-27b-it"
for i in $(seq 1 60); do
  A=$(sbatch --parsable --job-name=rl_v4a_bigjudge --export=ALL,TAG=v4a_bigjudge,EXTRA="$J" slurm/conf/ss_rl_v3.sbatch 2>&1)
  [[ "$A" =~ ^[0-9]+$ ]] && break; echo "$(date) retry A: $A"; sleep 300
done
for i in $(seq 1 60); do
  D=$(sbatch --parsable slurm/conf/ss_distill_push.sbatch 2>&1)
  [[ "$D" =~ ^[0-9]+$ ]] && break; echo "$(date) retry D: $D"; sleep 300
done
for i in $(seq 1 60); do
  C=$(sbatch --parsable --dependency=afterok:$D --job-name=rl_v4c_distill_bigjudge --export=ALL,TAG=v4c_distill_bigjudge,EXTRA="$J --init-adapters /l/users/daniil.orel/wmworld_rl/distill_push/adapters_round1.pt" slurm/conf/ss_rl_v3.sbatch 2>&1)
  [[ "$C" =~ ^[0-9]+$ ]] && break; echo "$(date) retry C: $C"; sleep 300
done
echo "$(date) submitted: v4a_bigjudge $A, distill_push $D, v4c_distill_bigjudge $C"
echo "Amendment 19 note ($(date '+%Y-%m-%d %H:%M')): queued v4a_bigjudge $A, distill_push $D, v4c_distill_bigjudge $C (after $D)." >> docs/preregistration.md
