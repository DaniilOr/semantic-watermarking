#!/bin/bash
# attacks for runs/ob_bench_$TARGET (reorder/resegment merge; paraphrases by gemma-4-31b as for the 3B)
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/ob_bench_${TARGET:?}
A=base,clean_seed1,semstamp,chain_reference
python scripts/reorder_attack.py --run $R --arms $A --attacks reorder_sent,resegment
python scripts/api_paraphrase.py --run $R --arms $A --level light --workers 16
python scripts/api_paraphrase.py --run $R --arms $A --level strong --workers 16
