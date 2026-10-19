#!/bin/bash
# login-node watcher: after the dev shards leave the queue, submit the steer-more ablation job (retry while the QoS slot limit is hit)
cd /home/daniil.orel/wmworld
while squeue -h -j 232025,232026,232027 2>/dev/null | grep -q . ; do sleep 180; done
for attempt in $(seq 1 300); do
  J=$(sbatch --parsable slurm/ablations_steer.sbatch 2>/dev/null) && { echo "ablation job $J submitted at $(date)"; exit 0; }
  sleep 120
done
echo "failed to submit ablation job"
