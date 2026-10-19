#!/bin/bash
# detection for the k-SemStamp runs (CPU): the same k-means partition per domain; human null = sp_bench_null.  Env: RUN, DOMS
set -uo pipefail; cd /home/daniil.orel/wmworld
ENC_c4=/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282
ENC_booksum=/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-booksum-sbert/snapshots/e7569f31e4b249b18c043fdaeedbfdccd2028289
for spec in $DOMS; do
  D=${spec%%:*}; P=${spec#*:}; E=ENC_$P
  python scripts/ksemstamp.py detect --centroids third_party/semstamp_ref/centroids/$P-cluster_8_centers.pt --embedder ${!E} --device cpu \
    --run runs/$RUN --null runs/sp_bench_null --domains $D --attacks --arms ksemstamp chain_kmeans --out runs/ks_detect_${RUN}_$D.json
done
