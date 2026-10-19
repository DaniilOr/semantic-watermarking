#!/bin/bash
# Runs after an instruction-task GPU job ends (afterany).  If ARMS are incomplete for any of TARGETS (e.g. the job hit the
# 48 h limit), resubmit the job for those targets -- it resumes, finished texts are skipped by id -- and a gate after it.
# Stops after 3 rounds so a failing job cannot loop.
set -uo pipefail; cd /home/daniil.orel/wmworld
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset $v; done
TARGETS=${TARGETS:?}; ARMS=${ARMS:?}; MEM=${MEM:-110G}; ROUND=${ROUND:-1}; DOMAINS=${DOMAINS:-it_dolly it_writingprompts it_ifeval it_gsm8k}
INC=$(python - <<PYEOF
import glob, json
size = {"it_dolly": 1000, "it_writingprompts": 1000, "it_ifeval": 541, "it_gsm8k": 1319, "it_translation": 1000}
need = {d: size[d] for d in "$DOMAINS".split()}
inc = []
for t in "$TARGETS".split():
    for arm in "$ARMS".split():
        ids = {}
        for f in glob.glob(f"runs/it_bench_{t}/generations.{arm}.shard*.jsonl"):
            for l in open(f):
                if l.strip().endswith("}"):
                    r = json.loads(l); ids.setdefault(r["domain"], set()).add(r["id"])
        if any(len(ids.get(d, ())) < n for d, n in need.items()):
            inc.append(t); break
print(" ".join(inc))
PYEOF
)
if [ -z "$INC" ]; then echo "complete: $TARGETS / $ARMS"; exit 0; fi
if [ "$ROUND" -ge 3 ]; then echo "still incomplete after $ROUND rounds ($INC / $ARMS): stopping, needs a look"; exit 1; fi
J=$(sbatch --parsable --mem=$MEM --job-name=it_resume --export=ALL,TARGETS="$INC",ARMS="$ARMS",DOMAINS="$DOMAINS" slurm/conf/it_gen.sbatch)
G=$(sbatch --parsable --job-name=it_gate -c 1 --mem=1G --time=00:15:00 --dependency=afterany:$J --export=ALL,TARGETS="$INC",ARMS="$ARMS",DOMAINS="$DOMAINS",MEM=$MEM,ROUND=$((ROUND + 1)),CMD="bash slurm/conf/it_gate.sh" slurm/cpu.sbatch)
echo "incomplete ($INC / $ARMS): resubmitted $J, gate $G (round $((ROUND + 1)))"
