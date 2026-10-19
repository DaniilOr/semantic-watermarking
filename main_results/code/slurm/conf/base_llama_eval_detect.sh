#!/bin/bash
# detection (CPU) for the base-Llama runs, clean + attacked copies
set -uo pipefail; cd /home/daniil.orel/wmworld
for T in ${TARGETS:-llama1b_base qwen08b_base llama3b_base}; do
  python scripts/ob_detect.py --device cpu --run runs/ob_bench_$T --null runs/ob_bench_null --out runs/ob_detect_ob_bench_$T.json
  python scripts/ob_detect.py --device cpu --run runs/sp_bench_$T --null runs/sp_bench_null --domains sp_realnews sp_booksum sp_tifu --out runs/sp_detect_$T.json
done
