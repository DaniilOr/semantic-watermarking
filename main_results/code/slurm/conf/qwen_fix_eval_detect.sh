#!/bin/bash
# detection (CPU) for the regenerated Qwen runs, plus the generator/detector alignment breakdown
set -uo pipefail; cd /home/daniil.orel/wmworld
python scripts/ob_detect.py --device cpu --run runs/ob_bench_${T:?}_nl --null runs/ob_bench_null --out runs/ob_detect_ob_bench_${T}_nl.json
python scripts/ob_detect.py --device cpu --run runs/sp_bench_${T}_nl --null runs/sp_bench_null --domains sp_realnews sp_booksum sp_tifu --out runs/sp_detect_${T}_nl.json
python scripts/ss_detect_decomp.py --out runs/qwen_fix_decomp_${T}.json runs/ob_bench_${T}_nl runs/sp_bench_${T}_nl
