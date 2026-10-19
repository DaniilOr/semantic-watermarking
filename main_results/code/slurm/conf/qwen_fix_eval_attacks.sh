#!/bin/bash
# attacks on the regenerated Qwen watermark arms (runs/<bench>_${T}_nl)
set -uo pipefail; cd /home/daniil.orel/wmworld; A=semstamp,chain_reference
for R in runs/ob_bench_${T:?}_nl runs/sp_bench_${T}_nl; do
  python scripts/reorder_attack.py --run $R --arms $A --attacks reorder_sent,resegment
  python scripts/api_paraphrase.py --run $R --arms $A --level light --workers 16
  python scripts/api_paraphrase.py --run $R --arms $A --level strong --workers 16
done
