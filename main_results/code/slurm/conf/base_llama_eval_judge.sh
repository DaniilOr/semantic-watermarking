#!/bin/bash
# gemma-4-31b judge for the base-Llama runs: preference vs base (both orders), pointwise flaws, clean band from base + clean_seed1
set -uo pipefail; cd /home/daniil.orel/wmworld
for T in ${TARGETS:-llama1b_base qwen08b_base llama3b_base}; do for B in ob_bench sp_bench; do R=runs/${B}_$T
  python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms clean_seed1,semstamp,chain_reference --workers 16 | grep -E "PREFERENCE|comparisons"
  python scripts/api_pointwise.py --config configs/pilot.yaml --run $R --arms base,clean_seed1,semstamp,chain_reference --workers 16 | tail -2
  python scripts/quality_vs_clean.py --clean $R:base --clean $R:clean_seed1 --marked $R --out runs/quality_vs_clean_${B}_$T.json
done; done
