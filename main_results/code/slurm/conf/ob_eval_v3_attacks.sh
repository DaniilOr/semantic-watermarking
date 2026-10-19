#!/bin/bash
set -uo pipefail; cd /home/daniil.orel/wmworld
ARMS=chain_v3a1000,chain_v3a1400,chain_v3b600
python scripts/reorder_attack.py --run runs/ob_bench --arms $ARMS --attacks reorder_sent,resegment
python scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level light --workers 16
python scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level strong --workers 16
