#!/bin/bash
# code protocol evaluation (CPU): pass@1 on MBPP's tests, end-of-turn scan, detection vs the human reference solutions
set -uo pipefail; cd /home/daniil.orel/wmworld
for T in ${TARGETS:?}; do
  python scripts/eos_leak_scan.py runs/code_mbpp_$T
  MBPP_PASS_OUT=runs/mbpp_pass_code_$T.json python scripts/mbpp_pass.py runs/code_mbpp_$T
  python scripts/ob_detect.py --device cpu --run runs/code_mbpp_$T --null runs/sp_bench_null --domains sp_mbpp --attacks --out runs/code_detect_$T.json
done
