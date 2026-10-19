#!/bin/bash
# login-node watcher: wait until the fit job (232014) leaves the queue, then submit the dev evaluation with a dependency on the shards
cd /home/daniil.orel/wmworld
while squeue -h -j 232014 2>/dev/null | grep -q . ; do sleep 120; done
for attempt in $(seq 1 200); do
  EV=$(sbatch --parsable --dependency=afterok:232025:232026:232027 --job-name=wm_eval_dev --export=ALL,RUN=runs/dev,CFG=configs/pilot.yaml slurm/evaluate.sbatch 2>/dev/null) && { echo "eval job $EV submitted at $(date)"; exit 0; }
  sleep 120
done
echo "failed to submit eval job"
