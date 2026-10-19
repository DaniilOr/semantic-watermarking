#!/bin/bash
# detection on CPU (no GPU job slot needed): the new benchmark, then the earlier benchmark with the refilled texts
set -uo pipefail; cd /home/daniil.orel/wmworld
python scripts/ob_detect.py --device cpu --run runs/sp_bench_${TARGET:?} --null runs/sp_bench_null --domains sp_realnews sp_booksum sp_tifu sp_mbpp --out runs/sp_detect_$TARGET.json
python scripts/ob_detect.py --device cpu --run runs/${OBRUN:?} --null runs/ob_bench_null --arms base clean_seed1 human semstamp chain_reference --out runs/ob_detect_${OBRUN}_refill.json
