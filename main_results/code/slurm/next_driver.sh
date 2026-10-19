#!/bin/bash
# Chains the next round on the login node (no compute): waits for the two running jobs, then the long stories,
# then the analysis + judging.  Log: slurm_logs/driver.log
set -u
cd /home/daniil.orel/wmworld
LOG=slurm_logs/driver.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
submit() { local j n; for n in $(seq 1 360); do j=$(sbatch --parsable "$1" 2>>"$LOG") && { say "submitted $1 -> $j"; echo "$j"; return 0; }; sleep 60; done; say "GAVE UP submitting $1"; echo 0; return 1; }
wait_jobs() { local j; for j in "$@"; do [ "$j" = "0" ] && continue; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done; done; }
count() { cat "$1"/generations*.jsonl 2>/dev/null | wc -l; }
jf=$1; jn=$2
say "next-round driver started (pid $$): fill-in $jf, long null $jn"
wait_jobs "$jf"; say "strength fill-in done: $(count runs/strength_fill) texts"
jfj=$(submit slurm/conf/strength_fill_judge.sbatch)
js=$(submit slurm/conf/long_story.sbatch)                      # independent of the null: run them side by side
wait_jobs "$js"; say "long stories done: $(count runs/long_story) texts"
wait_jobs "$jn"; say "long clean null done: $(count runs/long_null) texts"
ja=$(submit slurm/conf/long_analyse.sbatch); wait_jobs "$ja" "$jfj"
for f in runs/detector_variants_strength_fill.json runs/detector_variants_long_story.json \
         runs/long_story/eval/api_pref_google_gemma-4-31b-it.json runs/long_story/eval/api_pointwise_google_gemma-4-31b-it.jsonl; do
  [ -f "$f" ] && say "ok      $f" || say "MISSING $f"
done
say "NEXT ROUND DONE"
