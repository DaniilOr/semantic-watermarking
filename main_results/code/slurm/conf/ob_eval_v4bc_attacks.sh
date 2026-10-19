#!/bin/bash
set -uo pipefail; cd /home/daniil.orel/wmworld
A=chain_v4c1500,chain_v4b500,chain_v4d100
python scripts/reorder_attack.py --run runs/ob_bench --arms $A --attacks reorder_sent,resegment
python scripts/api_paraphrase.py --run runs/ob_bench --arms $A --level light --workers 16
python scripts/api_paraphrase.py --run runs/ob_bench --arms $A --level strong --workers 16
