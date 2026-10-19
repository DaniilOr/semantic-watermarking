#!/bin/bash
# chain link (CPU): submit the open-task generation for TARGETS once a GPU-QoS slot is free, and its scoring after it.
set -uo pipefail; cd /home/daniil.orel/wmworld
G=$(TARGETS="$TARGETS" sbatch --parsable --job-name=it_open_${TARGETS// /_} --export=ALL slurm/conf/it_open_gen.sbatch) || exit 1
echo "generation job $G for $TARGETS"
for T in $TARGETS; do
  E=$(T=$T CMD="bash slurm/conf/it_open_eval.sh" sbatch --parsable --job-name=it_open_eval_$T --dependency=afterok:$G --time=2-00:00:00 \
      --cpus-per-task=16 --mem=64G --export=ALL slurm/cpu.sbatch); echo "eval job $E for $T"
done
