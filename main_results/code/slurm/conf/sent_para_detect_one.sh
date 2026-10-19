#!/bin/bash
# detection on SemStamp's sentence-level paraphrase attacks (CPU), one run: RUN, DOMS (space-separated), NULL
set -uo pipefail; cd /home/daniil.orel/wmworld; A=${A:-"pegasus pegasus_bigram parrot parrot_bigram"}
python scripts/ob_detect.py --device cpu --run $RUN --null $NULL --domains $DOMS --attacks $A --arms base semstamp chain_reference --out runs/sentpara_detect_$(basename $RUN)${OUT_SUFFIX:-}.json
