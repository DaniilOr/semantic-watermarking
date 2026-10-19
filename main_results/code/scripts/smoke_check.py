"""Smoke gate for generation jobs: print every record of the given run dirs and exit 1 if any text failed (status != ok),
e.g. a CUDA assert caught per text -- the generator records such failures and still exits 0.  --leaks: also fail when a
non-base text continues past the end of turn (fake chat turn)."""
import glob, json, re, sys
PAT = re.compile(r"(^|[^\w])(assistant|user)\n\n|<\|im_start\|>|<\|eot_id\|>|<\|start_header_id\|>")
leaks = "--leaks" in sys.argv
bad = 0
for d in [a for a in sys.argv[1:] if a != "--leaks"]:
    for f in sorted(glob.glob(f"{d}/generations.*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            failed = r.get("status", "ok") != "ok"
            leak = leaks and r["arm"] != "base" and bool(PAT.search(r.get("text") or ""))
            bad += failed or leak
            print(d.split("/")[-1], r["arm"], r.get("status"), r.get("visible_tokens"), "LEAK" if leak else "",
                  str(r.get("error"))[:120] if failed else repr((r.get("text") or "")[:150]))
print("SMOKE", "FAILED" if bad else "OK", f"({bad} bad records)")
sys.exit(1 if bad else 0)
