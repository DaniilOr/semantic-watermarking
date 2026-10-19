#!/bin/bash
#SBATCH --job-name=submit_train_v4
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Submits the 2-GPU training job (slurm/conf/ss_train_v4.sbatch) once the run-3 generation has freed a GPU submit slot.
cd /home/daniil.orel/wmworld
for i in $(seq 1 60); do
  T=$(sbatch --parsable slurm/conf/ss_train_v4.sbatch 2>&1)
  if [[ "$T" =~ ^[0-9]+$ ]]; then
    echo "$(date) submitted training job $T"
    echo "Amendment 19 note ($(date '+%Y-%m-%d %H:%M')): the three runs are one 2-GPU job $T (slurm/conf/ss_train_v4.sbatch; lane 0 v4a_bigjudge, lane 1 distill_push then v4c_distill_bigjudge), started while GRPO+push generation continues on the other 2 GPUs." >> docs/preregistration.md
    exit 0
  fi
  echo "$(date) retry: $T"; sleep 120
done
exit 1
