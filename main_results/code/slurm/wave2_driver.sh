#!/bin/bash
# Wave-2 driver.  Runs on the login node and does NO compute: it waits for result files, submits jobs, and applies the
# pre-registered selection rule (docs/preregistration.md) mechanically.  Everything is logged to slurm_logs/driver.log.
set -u
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
LOG=slurm_logs/driver.log
JUDGE=api_judge_google_gemma-4-31b-it.json
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
submit() {   # retries while the per-user job limit refuses the submission (up to 6 h)
  local j n
  for n in $(seq 1 360); do
    j=$(sbatch --parsable "$1" 2>>"$LOG") && { say "submitted $1 -> $j"; echo "$j"; return 0; }
    sleep 60
  done
  say "GAVE UP submitting $1"; echo 0; return 1
}
wait_jobs() { local j; for j in "$@"; do [ "$j" = "0" ] && continue; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; done; }
count() { cat "$1"/generations*.jsonl 2>/dev/null | wc -l; }

say "driver started (pid $$)"
# ---- A. the trade-off sweep ---------------------------------------------------------------------------------------
wait_jobs "$@"
N=$(count runs/tradeoff); say "first sweep pass left the queue: $N texts"
# extension pass: four frontier arms were added after the early read (docs/preregistration.md, amendment 3); the same
# two jobs now enumerate 15 arms x 60 prompts = 900 texts and skip everything that already exists
a=$(submit slurm/conf/wm_tradeoff_p0.sbatch); b=$(submit slurm/conf/wm_tradeoff_p1.sbatch); wait_jobs "$a" "$b"
N=$(count runs/tradeoff); say "sweep generated: $N/900 texts"
if [ "$N" -lt 900 ]; then
  say "sweep incomplete: one more pass"
  a=$(submit slurm/conf/wm_tradeoff_p0.sbatch); b=$(submit slurm/conf/wm_tradeoff_p1.sbatch); wait_jobs "$a" "$b"
  N=$(count runs/tradeoff); say "after the retry: $N/900 texts"
fi
[ "$N" -ge 840 ] || { say "STOP: too few sweep texts ($N)"; exit 1; }
# ---- B. detection analysis (GPU) and blind judge (CPU queue) in parallel --------------------------------------------
jp=$(submit slurm/conf/wm_tradeoff_post.sbatch); jj=$(submit slurm/conf/judge_tradeoff.sbatch); wait_jobs "$jp" "$jj"
if [ ! -f runs/tradeoff/eval/$JUDGE ]; then say "judge output missing: one retry"; jj=$(submit slurm/conf/judge_tradeoff.sbatch); wait_jobs "$jj"; fi
if [ ! -f runs/detector_variants_tradeoff.json ]; then say "detector analysis missing: one retry"; jp=$(submit slurm/conf/wm_tradeoff_post.sbatch); wait_jobs "$jp"; fi
[ -f runs/detector_variants_tradeoff.json ] && [ -f runs/tradeoff/eval/$JUDGE ] || { say "STOP: selection inputs missing"; exit 1; }
# ---- C. mechanical selection ----------------------------------------------------------------------------------------
$PY scripts/select_config.py --dev runs/detector_variants_dev.json --tradeoff runs/detector_variants_tradeoff.json \
    --judge runs/tradeoff/eval/$JUDGE --out runs/selection.json >> "$LOG" 2>&1 || { say "STOP: selection failed"; exit 1; }
ARM=$($PY -c "import json;print(json.load(open('runs/selection.json'))['configuration']['chosen'])")
DET=$($PY -c "import json;print(json.load(open('runs/selection.json'))['detector']['chosen'])")
echo "$ARM" >| runs/chosen_arm.txt; echo "$DET" >| runs/chosen_detector.txt
say "SELECTED configuration=$ARM detector=$DET"
# ---- D. wave 2 -------------------------------------------------------------------------------------------------------
jc=$(submit slurm/conf/w2_conf.sbatch); ji=$(submit slurm/conf/w2_ifeval.sbatch)
wait_jobs "$jc"; say "confirmation generated: $(count runs/conf_cross) texts"
ja=$(submit slurm/conf/w2_api_conf.sbatch)            # CPU: judge + paraphrase attacks
jl=$(submit slurm/conf/w2_long.sbatch)                # GPU: long texts, same key vs rotating keys, chosen delivery
wait_jobs "$ji"; say "ifeval generated: $(count runs/ifeval_cross) texts"; jq=$(submit slurm/conf/w2_api_ifeval.sbatch)
wait_jobs "$ja"; js=$(submit slurm/conf/w2_attack_score.sbatch)
wait_jobs "$jl"; say "long texts generated: $(count runs/long_c) texts"; jr=$(submit slurm/conf/w2_api_long.sbatch)
wait_jobs "$js" "$jq" "$jr"
# ---- E. what exists ---------------------------------------------------------------------------------------------------
for f in runs/detector_variants_conf.json runs/conf_cross/eval/$JUDGE runs/detector_variants_attacks.json runs/ifeval_cross/results.json \
         runs/detector_variants_ifeval.json runs/ifeval_cross/eval/$JUDGE runs/detector_variants_long_c.json runs/long_c/eval/$JUDGE; do
  [ -f "$f" ] && say "ok      $f" || say "MISSING $f"
done
HF_HUB_OFFLINE=1 $PY scripts/build_brief_page.py --out docs/brief_page.html >> "$LOG" 2>&1
say "ALL DONE"
