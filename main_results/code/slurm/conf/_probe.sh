#!/bin/bash
hostname; ls -la /usr/lib/x86_64-linux-gnu/slurm/spank_pyxis.so 2>&1; cd /home/daniil.orel/wmworld; sbatch --test-only --export=ALL,CMD=true slurm/cpu.sbatch 2>&1 | tail -1
