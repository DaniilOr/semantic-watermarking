"""Remove records with an empty text from an arm's generation shards (backup *.bak_empty), so a rerun regenerates them."""
import glob, json, shutil, sys
run, arm = sys.argv[1], sys.argv[2]
tot = 0
for f in sorted(glob.glob(f"{run}/generations.{arm}.shard*.jsonl")):
    lines = open(f).read().splitlines()
    keep = [l for l in lines if l.strip() and json.loads(l).get("text")]
    if len(keep) < len(lines):
        shutil.copy(f, f + ".bak_empty")
        open(f, "w").write("\n".join(keep) + ("\n" if keep else ""))
        tot += len(lines) - len(keep)
print(f"{run} {arm}: removed {tot} empty records")
