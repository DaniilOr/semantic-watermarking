#!/bin/bash
# attacks for the base-Llama runs (reorder/resegment; paraphrases by gemma-4-31b), both benchmarks
set -uo pipefail; cd /home/daniil.orel/wmworld; A=base,clean_seed1,semstamp,chain_reference
for T in ${TARGETS:-llama1b_base qwen08b_base llama3b_base}; do for R in runs/ob_bench_$T runs/sp_bench_$T; do
  python scripts/reorder_attack.py --run $R --arms $A --attacks reorder_sent,resegment
  python scripts/api_paraphrase.py --run $R --arms $A --level light --workers 16
  python scripts/api_paraphrase.py --run $R --arms $A --level strong --workers 16
done; done
