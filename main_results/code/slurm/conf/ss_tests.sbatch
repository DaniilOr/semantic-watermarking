#!/bin/bash
#SBATCH --job-name=ss_tests
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -m pytest -q tests/test_semwm.py tests/test_multilora.py tests/test_adapter_sampler.py 2>&1
$PY -c "import ast; [ast.parse(open(f).read()) for f in ('scripts/embedder_study.py','scripts/semstamp_detect.py','scripts/steered_semstamp.py')]; print('scripts parse ok')"
