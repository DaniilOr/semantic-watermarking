#!/bin/bash
# gemma-4-31b judge on the instruction benchmark (the four finished datasets only)
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/it_bench_${T:?}
export JUDGE_DOMAINS="it_dolly it_writingprompts it_ifeval it_gsm8k"
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms clean_seed1,semstamp,chain_reference --workers 16 | grep -E "PREFERENCE|comparisons"
python scripts/api_pointwise.py --config configs/pilot.yaml --run $R --arms base,clean_seed1,semstamp,chain_reference --workers 16 | tail -2
python scripts/quality_vs_clean.py --clean $R:base --clean $R:clean_seed1 --marked $R --out runs/quality_vs_clean_it_bench_$T.json
