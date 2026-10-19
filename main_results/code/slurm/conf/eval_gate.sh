#!/bin/bash
# Gate for the base-model runs of target $T (runs/ob_bench_$T: 4 arms x 1500, runs/sp_bench_$T: 4 arms x 3000).
# Complete -> submit its attacks, judge, then detection (TARGETS=$T).  Not complete ->
#   MODE=begin  resubmit this gate one hour later (a scheduled check, no polling loop)
#   MODE=after  the generation job has ended (e.g. time limit): resubmit the generation for $T (resumes) and a gate after it
set -uo pipefail; cd /home/daniil.orel/wmworld
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset $v; done
T=${T:?}; MODE=${MODE:?}
python - <<PYEOF
import glob, json, sys
need = {"ob_bench": 1500, "sp_bench": 3000}
short = []
for b, n in need.items():
    for arm in ("base", "clean_seed1", "chain_reference", "semstamp"):
        ids = set()
        for f in glob.glob(f"runs/{b}_$T/generations.{arm}.shard*.jsonl"):
            for l in open(f):
                if l.strip().endswith("}"):
                    ids.add(json.loads(l)["id"])
        if len(ids) < n:
            short.append(f"{b}/{arm} {len(ids)}/{n}")
print("$T complete" if not short else "$T incomplete: " + ", ".join(short))
sys.exit(1 if short else 0)
PYEOF
if [ $? = 0 ]; then
  C="--time=1-00:00:00 --export=ALL,TARGETS=$T"
  A=$(sbatch --parsable --job-name=${T}_attacks $C,CMD="bash slurm/conf/base_llama_eval_attacks.sh" slurm/cpu.sbatch)
  J=$(sbatch --parsable --job-name=${T}_judge $C,CMD="bash slurm/conf/base_llama_eval_judge.sh" slurm/cpu.sbatch)
  D=$(sbatch --parsable --job-name=${T}_detect -c 16 --mem=64G --dependency=afterok:$A $C,CMD="bash slurm/conf/base_llama_eval_detect.sh" slurm/cpu.sbatch)
  echo "submitted evaluations for $T: attacks $A, judge $J, detection $D"
elif [ $MODE = begin ]; then
  G=$(sbatch --parsable --job-name=gate_$T -c 1 --mem=1G --time=00:15:00 --begin=now+1hour --export=ALL,T=$T,MODE=begin,CMD="bash slurm/conf/eval_gate.sh" slurm/cpu.sbatch)
  echo "not complete: gate rescheduled as $G (+1 h)"
else
  R=$(sbatch --parsable --job-name=base_${T}_resume --export=ALL,TARGETS=$T slurm/conf/base_llama_gen.sbatch)
  G=$(sbatch --parsable --job-name=gate_$T -c 1 --mem=1G --time=00:15:00 --dependency=afterany:$R --export=ALL,T=$T,MODE=after,CMD="bash slurm/conf/eval_gate.sh" slurm/cpu.sbatch)
  echo "generation ended incomplete: resubmitted generation $R, gate $G after it"
fi
