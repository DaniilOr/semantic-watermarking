#!/bin/bash
# translation scoring (CPU) for runs/it_bench_$T: chrF (it_eval.py, rewrites runs/it_quality_it_bench_$T.json for all tasks),
# detection vs the human reference translations, and the judge on translation only (instruction summaries kept as *.instruction4.json)
set -uo pipefail; cd /home/daniil.orel/wmworld; R=runs/it_bench_${T:?}
python scripts/it_eval.py $R
python scripts/ob_detect.py --device cpu --run $R --null runs/it_bench_null --domains it_translation --attacks --out runs/it_detect_translation_$T.json
export JUDGE_DOMAINS=it_translation
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run $R --arms clean_seed1,semstamp,chain_reference --workers 16 | grep -E "PREFERENCE|comparisons"
command cp -f $R/eval/api_pref_google_gemma-4-31b-it.json $R/eval/api_pref_google_gemma-4-31b-it.translation.json
command cp -f $R/eval/api_pref_google_gemma-4-31b-it.instruction4.json $R/eval/api_pref_google_gemma-4-31b-it.json
