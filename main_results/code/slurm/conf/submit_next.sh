#!/bin/bash
# run from a tiny CPU job with --dependency=afterany:<GPU job>: submits the next GPU job of the queue and its CPU evaluation
# (the GPU QoS allows 4 submitted jobs, so the rest of the queue is chained this way).  STAGE selects what to submit.
set -uo pipefail; cd /home/daniil.orel/wmworld
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset $v; done
cpu() { sbatch --parsable --time=1-00:00:00 "$@"; }
case ${STAGE:?} in
  base_1b_qwen)
    J=$(sbatch --parsable --job-name=base_1b_qwen_gen --export=ALL,TARGETS="llama1b_base qwen08b_base" slurm/conf/base_llama_gen.sbatch)
    A=$(cpu --job-name=b1q_attacks --dependency=afterok:$J --export=ALL,TARGETS="llama1b_base qwen08b_base",CMD="bash slurm/conf/base_llama_eval_attacks.sh" slurm/cpu.sbatch)
    cpu --job-name=b1q_judge --dependency=afterok:$J --export=ALL,TARGETS="llama1b_base qwen08b_base",CMD="bash slurm/conf/base_llama_eval_judge.sh" slurm/cpu.sbatch
    cpu --job-name=b1q_detect -c 16 --mem=64G --dependency=afterok:$A --export=ALL,TARGETS="llama1b_base qwen08b_base",CMD="bash slurm/conf/base_llama_eval_detect.sh" slurm/cpu.sbatch ;;
  code_small|code_3b)
    [ $STAGE = code_small ] && TG="llama1b qwen08b" || TG="llama3b"
    J=$(sbatch --parsable --nice=100 --job-name=${STAGE}_gen --export=ALL,TARGETS="$TG" slurm/conf/code_gen.sbatch)   # yields to resumed base-model generation
    cpu --job-name=${STAGE}_eval -c 16 --mem=64G --dependency=afterok:$J --export=ALL,TARGETS="$TG",CMD="bash slurm/conf/code_eval.sh" slurm/cpu.sbatch ;;
  it_semstamp_3b|it_semstamp_small)
    # instruction tasks, SemStamp arm last (user 2026-10-01); evaluation is submitted separately once written
    if [ $STAGE = it_semstamp_3b ]; then TG="llama3b"; MEM=110G; else TG="llama1b qwen08b"; MEM=56G; fi
    J=$(sbatch --parsable --mem=$MEM --job-name=${STAGE}_gen --export=ALL,TARGETS="$TG",ARMS=semstamp slurm/conf/it_gen.sbatch)
    # the SemStamp arm can exceed the 48 h job limit: a gate resumes whatever is left when the job ends
    sbatch --parsable --job-name=it_gate -c 1 --mem=1G --time=00:15:00 --dependency=afterany:$J --export=ALL,TARGETS="$TG",ARMS=semstamp,MEM=$MEM,CMD="bash slurm/conf/it_gate.sh" slurm/cpu.sbatch ;;
  sent_para)
    # SemStamp's sentence-level paraphrase attacks (Pegasus, Parrot) on existing watermarked texts, then detection
    S="runs/semstamp_setting_opt13b_ss|sp_realnews runs/semstamp_setting_llama3b_base|sp_realnews runs/sp_bench_llama3b|sp_realnews,sp_booksum,sp_tifu runs/ob_bench|ob_arxiv,ob_booksum,ob_wikipedia"
    J=$(sbatch --parsable --job-name=sent_para --mem=64G --export=ALL,SPECS="$S" slurm/conf/sent_para.sbatch)
    cpu --job-name=sentpara_detect -c 16 --mem=64G --dependency=afterok:$J --export=ALL,CMD="bash slurm/conf/sent_para_detect.sh" slurm/cpu.sbatch ;;
esac
echo "submitted $STAGE: GPU job $J"
