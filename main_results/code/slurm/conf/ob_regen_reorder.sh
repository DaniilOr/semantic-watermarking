#!/bin/bash
set -uo pipefail; cd /home/daniil.orel/wmworld
python scripts/reorder_attack.py --run runs/ob_bench --arms base,clean_seed1,semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3,chain_seg_push,chain_seg_push_m,chain_v3a1000,chain_v3a1400,chain_v3b600,chain_grpo300_push --attacks reorder_sent,resegment
