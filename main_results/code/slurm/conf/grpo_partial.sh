#!/bin/bash
# judge the chain_grpo300 texts written so far (resumable: the second pass skips them), then the same-prompt stats
set -uo pipefail
cd /home/daniil.orel/wmworld
python scripts/api_judge.py --mode preference --config configs/pilot.yaml --run runs/ob_bench --arms chain_grpo300 --workers 16 | tail -3
python scripts/api_pointwise.py --config configs/pilot.yaml --run runs/ob_bench --arms chain_grpo300 --workers 16 | tail -3
python scripts/grpo_partial_stats.py --arm chain_grpo300 --domain ob_arxiv
