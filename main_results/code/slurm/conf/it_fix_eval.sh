#!/bin/bash
# scoring of the push variants (runs/it_fix_llama3b, GSM8K + Dolly): GSM8K accuracy, detection, judge
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/it_fix_llama3b; ARMS=clean_seed1,semstamp,chain_reference,chain_spread,chain_protect,chain_spread_protect,chain_beta05
python scripts/it_eval.py $R
python scripts/ob_detect.py --device cpu --run $R --null runs/it_bench_null --domains it_gsm8k it_dolly --attacks --out runs/it_fix_detect_llama3b.json
export JUDGE_DOMAINS="it_gsm8k it_dolly"
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms $ARMS --workers 16 | grep -E "PREFERENCE|comparisons"
python scripts/api_pointwise.py --config configs/pilot.yaml --run $R --arms base,$ARMS --workers 16 | tail -2
