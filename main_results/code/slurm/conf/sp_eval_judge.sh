#!/bin/bash
# gemma-4-31b judge: runs/sp_bench_$TARGET (all arms) and the refilled push + margin texts of runs/$OBRUN (resumable)
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/sp_bench_${TARGET:?}; O=runs/${OBRUN:?}
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms clean_seed1,semstamp,chain_reference --workers 16 | grep -E "PREFERENCE|comparisons"
python scripts/api_pointwise.py --config configs/pilot.yaml --run $R --arms base,clean_seed1,semstamp,chain_reference --workers 16 | tail -2
python scripts/quality_vs_clean.py --clean $R:base --clean $R:clean_seed1 --marked $R --out runs/quality_vs_clean_sp_bench_$TARGET.json
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $O --arms chain_reference --workers 16 | grep -E "PREFERENCE|comparisons"
python scripts/api_pointwise.py --config configs/pilot.yaml --run $O --arms chain_reference --workers 16 | tail -2
python scripts/quality_vs_clean.py --clean $O:base --clean $O:clean_seed1 --marked $O --out runs/quality_vs_clean_${OBRUN}_refill.json
