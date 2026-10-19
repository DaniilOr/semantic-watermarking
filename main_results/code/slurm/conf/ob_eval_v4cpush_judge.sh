#!/bin/bash
set -uo pipefail; cd /home/daniil.orel/wmworld
A=chain_v4c1500_push
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run runs/ob_bench --arms $A --workers 16 | grep -E "PREFERENCE|comparisons"
python scripts/api_pointwise.py --config configs/pilot.yaml --run runs/ob_bench --arms $A --workers 16 | tail -2
python scripts/quality_vs_clean.py --clean runs/ob_bench:base --clean runs/ob_bench:clean_seed1 --marked runs/ob_bench --out runs/quality_vs_clean_ob_bench_v4cpush.json
