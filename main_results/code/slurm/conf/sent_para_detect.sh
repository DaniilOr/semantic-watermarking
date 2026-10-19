#!/bin/bash
# detection on SemStamp's sentence-level paraphrase attacks (CPU)
set -uo pipefail; cd /home/daniil.orel/wmworld; A="pegasus pegasus_bigram parrot parrot_bigram"
python scripts/ob_detect.py --device cpu --run runs/semstamp_setting_opt13b_ss --null runs/sp_bench_null --domains sp_realnews --attacks $A --arms base semstamp chain_reference --out runs/sentpara_detect_semstamp_setting_opt13b_ss.json
python scripts/ob_detect.py --device cpu --run runs/semstamp_setting_llama3b_base --null runs/sp_bench_null --domains sp_realnews --attacks $A --arms base semstamp chain_reference --out runs/sentpara_detect_semstamp_setting_llama3b_base.json
python scripts/ob_detect.py --device cpu --run runs/sp_bench_llama3b --null runs/sp_bench_null --domains sp_realnews sp_booksum sp_tifu --attacks $A --arms base semstamp chain_reference --out runs/sentpara_detect_sp_bench_llama3b.json
python scripts/ob_detect.py --device cpu --run runs/ob_bench --null runs/ob_bench_null --attacks $A --arms base semstamp chain_reference --out runs/sentpara_detect_ob_bench.json
