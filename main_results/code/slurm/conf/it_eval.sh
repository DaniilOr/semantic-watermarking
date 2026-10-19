#!/bin/bash
# instruction benchmark scoring (CPU) for runs/it_bench_$T: task quality (GSM8K accuracy, IFEval compliance) and detection
set -uo pipefail; cd /home/daniil.orel/wmworld; D="it_dolly it_writingprompts it_ifeval it_gsm8k"
python scripts/it_eval.py runs/it_bench_${T:?}
python scripts/ob_detect.py --device cpu --run runs/it_bench_$T --null runs/it_bench_null --domains $D --attacks --out runs/it_detect_$T.json
