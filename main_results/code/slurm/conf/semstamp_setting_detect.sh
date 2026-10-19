#!/bin/bash
# detection for the SemStamp-setting check (CPU): human null = the RealNews null documents of sp_bench, model null = the base arm
set -uo pipefail; cd /home/daniil.orel/wmworld
for M in opt13b_ss llama3b_base; do
  python scripts/ob_detect.py --device cpu --run runs/semstamp_setting_$M --null runs/sp_bench_null --domains sp_realnews --attacks --out runs/semstamp_setting_detect_$M.json
done
python scripts/ss_detect_decomp.py --out runs/semstamp_setting_decomp.json runs/semstamp_setting_opt13b_ss runs/semstamp_setting_llama3b_base
