#!/bin/bash
# Free home quota (2026-10-17, user request): move big folders to /l/users/daniil.orel/home_moved and leave symlinks so every
# existing path keeps working; delete the conda package cache, the Anaconda installer and the triton kernel cache.
set -uo pipefail
DEST=/l/users/daniil.orel/home_moved; mkdir -p $DEST
mvlink() {
  local src=$1 name=$2
  if [ -L "$src" ]; then echo "skip $src (already a link)"; return 0; fi
  [ -e "$src" ] || { echo "skip $src (missing)"; return 0; }
  echo "$(date +%T) copying $src -> $DEST/$name ($(du -sh --apparent-size "$src" | cut -f1))"
  rsync -a "$src/" "$DEST/$name/" || { echo "RSYNC FAILED for $src -- original kept"; return 1; }
  rm -rf "$src" && ln -s "$DEST/$name" "$src" && echo "$(date +%T) moved $src -> $DEST/$name"
}
FAIL=0
mvlink /home/daniil.orel/.cache/huggingface huggingface_cache || FAIL=1
mvlink /home/daniil.orel/wmworld/runs wmworld_runs || FAIL=1
for p in rl_verifiable_qwen3 semeval_task13 ADNOC cql; do mvlink /home/daniil.orel/$p $p || FAIL=1; done
/home/daniil.orel/anaconda3/bin/conda clean --all -y > /dev/null && echo "conda package cache cleaned"
rm -f /home/daniil.orel/Anaconda3-2024.02-1-Linux-x86_64.sh && echo "installer deleted"
rm -rf /home/daniil.orel/.triton/cache && echo "triton cache deleted"
echo "home now: $(du -xsh --apparent-size /home/daniil.orel 2>/dev/null | cut -f1)"
exit $FAIL
