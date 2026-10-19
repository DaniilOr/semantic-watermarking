#!/bin/bash
# attacks for runs/sp_bench_$TARGET (4 arms) and for the refilled push + margin texts of runs/$OBRUN (merge / resume)
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/sp_bench_${TARGET:?}; O=runs/${OBRUN:?}
A=base,clean_seed1,semstamp,chain_reference
python scripts/reorder_attack.py --run $R --arms $A --attacks reorder_sent,resegment
python scripts/api_paraphrase.py --run $R --arms $A --level light --workers 16
python scripts/api_paraphrase.py --run $R --arms $A --level strong --workers 16
python scripts/reorder_attack.py --run $O --arms chain_reference --attacks reorder_sent,resegment
python scripts/api_paraphrase.py --run $O --arms chain_reference --level light --workers 16
python scripts/api_paraphrase.py --run $O --arms chain_reference --level strong --workers 16
