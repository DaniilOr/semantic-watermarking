#!/bin/bash
# Wave-2 driver, stage D onward (the selection is already made).  Login node, no compute: waits, submits, logs.
# usage: wave2_driver_d.sh <conf_job_id> <ifeval_job_id>
set -u
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
LOG=slurm_logs/driver.log
JUDGE=api_judge_google_gemma-4-31b-it.json
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
submit() { local j n; for n in $(seq 1 360); do j=$(sbatch --parsable "$1" 2>>"$LOG") && { say "submitted $1 -> $j"; echo "$j"; return 0; }; sleep 60; done; say "GAVE UP submitting $1"; echo 0; return 1; }
wait_jobs() { local j; for j in "$@"; do [ "$j" = "0" ] && continue; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; done; }
count() { cat "$1"/generations*.jsonl 2>/dev/null | wc -l; }
jc=$1; ji=$2
say "driver D started (pid $$): confirmation job $jc, ifeval job $ji"
wait_jobs "$jc"; N=$(count runs/conf_cross); say "confirmation generated: $N/450 texts"
if [ "$N" -lt 450 ]; then say "confirmation incomplete: one more pass"; jc=$(submit slurm/conf/w2_conf.sbatch); wait_jobs "$jc"; say "confirmation after retry: $(count runs/conf_cross)/450"; fi
ja=$(submit slurm/conf/w2_api_conf.sbatch)            # CPU: judge + paraphrase attacks
jl=$(submit slurm/conf/w2_long.sbatch)                # GPU: long texts, one key vs rotating keys, selected push
wait_jobs "$ji"; N=$(count runs/ifeval_cross); say "ifeval generated: $N/400 texts"
if [ "$N" -lt 400 ]; then say "ifeval incomplete: one more pass"; ji=$(submit slurm/conf/w2_ifeval.sbatch); wait_jobs "$ji"; say "ifeval after retry: $(count runs/ifeval_cross)/400"; fi
jq=$(submit slurm/conf/w2_api_ifeval.sbatch)
wait_jobs "$ja"; js=$(submit slurm/conf/w2_attack_score.sbatch)
wait_jobs "$jl"; say "long texts generated: $(count runs/long_c)/180"; jr=$(submit slurm/conf/w2_api_long.sbatch)
jx=$(submit slurm/conf/tradeoff_proxy_eval.sbatch)    # the older proxy metrics of the sweep, now that GPUs are free
wait_jobs "$js" "$jq" "$jr" "$jx"
for f in runs/detector_variants_conf.json runs/conf_cross/eval/$JUDGE runs/detector_variants_attacks.json runs/ifeval_cross/results.json \
         runs/detector_variants_ifeval.json runs/ifeval_cross/eval/$JUDGE runs/detector_variants_long_c.json runs/long_c/eval/$JUDGE; do
  [ -f "$f" ] && say "ok      $f" || say "MISSING $f"
done
HF_HUB_OFFLINE=1 $PY scripts/build_brief_page.py --out docs/brief_page.html >> "$LOG" 2>&1
say "ALL DONE"
