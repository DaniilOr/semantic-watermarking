#!/bin/bash
# scoring of the open-ended instruction tasks for one target T (CPU): semantic detection, KGW detection, pairwise judge
set -uo pipefail; cd /home/daniil.orel/wmworld; T=${T:?}; R=runs/it_open_$T; D="it_eli5 it_hc3_finance it_essays"
python scripts/ob_detect.py --device cpu --run $R --null runs/it_bench_null --domains $D --attacks --out runs/it_open_detect_$T.json
python scripts/kgw_detect.py --run $R --null runs/it_bench_null --target $T --domains $D --arms kgw --out runs/kgw_detect_it_open_$T.json
export JUDGE_DOMAINS="$D"
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms clean_seed1,semstamp,chain_reference,kgw --workers 16 | grep -E "PREFERENCE|comparisons"
