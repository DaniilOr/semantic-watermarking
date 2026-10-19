#!/bin/bash
#SBATCH --job-name=baseline_setup2
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -m pip install --quiet NearPy 2>&1 | tail -3
$PY -c "
import sys; sys.path.insert(0,'third_party/SemStamp')
from nearpy.hashes import RandomBinaryProjections
h = RandomBinaryProjections('rbp_perm', projection_count=3, rand_seed=1234); h.reset(dim=768)
import numpy as np
print('nearpy ok; normals shape', np.asarray(h.normals).shape)
v = np.random.default_rng(0).normal(size=768).astype(np.float32)
print('hash of a vector:', h.hash_vector(v))
"
