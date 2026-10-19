#!/bin/bash
# submits the remaining jobs of the larger-sample experiment as submission slots free up (QoS: 4 submitted, 2 running)
A=244831; P0=244832; C0=244833; P1=244834
slot() { until [ "$(squeue -u daniil.orel -h -p cscc-gpu-p | wc -l)" -lt 4 ]; do sleep 120; done; }
slot; P2=$(sbatch --parsable --job-name=wm_pos2 --export=ALL,SHARD=2,NUM_SHARDS=3 slurm/conf/pos_gen.sbatch); echo "pos2=$P2"
slot; C1=$(sbatch --parsable --job-name=wm_code1 --export=ALL,SHARD=1,NUM_SHARDS=2 slurm/conf/code_gen.sbatch); echo "code1=$C1"
slot; CAL=$(sbatch --parsable --job-name=wm_calib --dependency=afterok:$A slurm/conf/calibrate.sbatch); echo "calibrate=$CAL"
slot; EP=$(sbatch --parsable --job-name=wm_eval_pos --dependency=afterok:$P0:$P1:$P2:$CAL --export=ALL,RUN=runs/conf_pos slurm/conf/evaluate.sbatch); echo "eval_pos=$EP"
slot; EC=$(sbatch --parsable --job-name=wm_eval_code --dependency=afterok:$C0:$C1:$CAL --export=ALL,RUN=runs/code slurm/conf/evaluate.sbatch); echo "eval_code=$EC"
echo "all submitted $(date)"
