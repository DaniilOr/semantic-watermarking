#!/bin/bash
#SBATCH --job-name=pmark_det
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# PMark online detection, full scope (user 2026-10-21: "Full, same as SemStamp/push"): per setting
#   1. SemStamp's sentence paraphrase attacks on the PMark texts (Pegasus, Parrot; top beam + bigram variants)
#   2. clean: PMark texts + the human null (all null texts of the datasets) + the plain model's own texts (arm base)
#   3. the 4 attacked copies
#   4. report (thresholds: human 1% / 0.1%, model 1%; sentence alignment generator vs detector)
# Resumable (every step skips finished work).  SETTINGS (env): any of sp ob opt it.  4 scoring lanes = 2 per GPU.
SETTINGS=${SETTINGS:?set SETTINGS}
LPG=${LANES_PER_GPU:-2}; NL=$((2 * LPG))                 # LANES_PER_GPU=1: fill pass for texts skipped on CUDA OOM
ATTACK_GEN=${ATTACK_GEN:-1}                              # 0: attacked copies already exist
echo "host=$(hostname) start=$(date) settings=$SETTINGS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
ATT="pegasus pegasus_bigram parrot parrot_bigram"
cfg() { case $1 in
  sp)  RUN=pm_sp_bench_llama3b;  DOMS="sp_realnews sp_booksum"; PR=sp_bench; NULL=sp_bench_null; MNULL=sp_bench_llama3b; TGT=llama3b; X="--max-new 200 --max-rows 192";;
  ob)  RUN=pm_ob_bench;          DOMS="ob_arxiv ob_booksum ob_wikipedia"; PR=ob_bench; NULL=ob_bench_null; MNULL=ob_bench; TGT=llama3b; X="--max-new 200 --max-rows 192";;
  opt) RUN=pm_setting_opt13b_ss; DOMS="sp_realnews"; PR=sp_bench; NULL=sp_bench_null; MNULL=semstamp_setting_opt13b_ss; TGT=opt13b_ss
       X="--max-new 205 --max-rows 128 --plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05";;
  it)  RUN=pm_it_bench_llama3b;  DOMS="it_dolly it_writingprompts it_ifeval it_gsm8k"; PR=it_bench; NULL=it_bench_null; MNULL=it_bench_llama3b; TGT=llama3b
       X="--max-new 512 --max-rows 64 --min-new-tokens 1";;          # 128 rows: CUDA OOM with 2 lanes per GPU (330546)
  esac; }
score() { TAG=$1; shift
  P=(); for g in 0 1; do for l in $(seq 0 $((LPG - 1))); do k=$((g * LPG + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/pmark_arm.py score --target $TGT --prompts $PR --texts "$@" \
    --null runs/$NULL --domains $DOMS $X --cache runs/pmark_scores/$S --tag $TAG --shard $k --num-shards $NL > slurm_logs/pmark_det_${S}_${TAG}_$k.log 2>&1 & P+=($!); done; done
  FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done; echo "$S $TAG scored (lane failure: $FAIL) $(date)"; }
for S in $SETTINGS; do cfg $S; [ "$ATTACK_GEN" = 1 ] || continue
  for A in pegasus parrot; do
    P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/sent_paraphrase_attack.py --run runs/$RUN --attacker $A --arms pmark \
      --domains $DOMS --shard $k --num-shards 6 > slurm_logs/pmark_att_${S}_${A}_$k.log 2>&1 & P+=($!); done; done
    for p in "${P[@]}"; do wait $p; done; echo "$S $A attack done $(date)"
  done
done
for S in $SETTINGS; do cfg $S; score clean runs/$RUN:pmark runs/$MNULL:base; done
for S in $SETTINGS; do cfg $S; score attacks $(for a in $ATT; do echo runs/${RUN}_$a:pmark; done); done
for S in $SETTINGS; do cfg $S
  $PY scripts/pmark_arm.py report --cache runs/pmark_scores/$S --null $NULL --model-null-run $MNULL --align runs/$RUN:pmark --out runs/pmark_detect_$S.json > slurm_logs/pmark_report_$S.log 2>&1
  echo "$S report: runs/pmark_detect_$S.json $(date)"
done
echo "end=$(date)"
