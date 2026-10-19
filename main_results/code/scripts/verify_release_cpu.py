"""Check that the exported package (~/wmark-release) reproduces the original detection numbers from the saved dumps."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("runs/_release_check")
REL = Path("/home/daniil.orel/wmark-release")


def convert(src, dest):
    z = np.load(src, allow_pickle=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, key_ids=z["key_ids"], offsets=z["offsets"], h0=z["h0"], d=z["d"], gen_id=z["gen_id"], arm=z["arm"],
                        key_id=z["assigned_key_id"], prompt_id=np.array([""] * len(z["arm"])), n_gen_tokens=z["n_gen_tokens"])


convert("runs/null_calibration/eval/posstats_confirmation.npz", OUT / "null" / "posstats.npz")
convert("runs/conf_cross/eval/posstats_confirmation.npz", OUT / "conf_cross" / "posstats.npz")
subprocess.run([sys.executable, str(REL / "scripts" / "evaluate_detection.py"), "--null", str(OUT / "null" / "posstats.npz"),
                "--run", str(OUT / "conf_cross"), "--variants", "clip_gate,plain", "--out", str(OUT / "detection.json")], check=True)
new = json.load(open(OUT / "detection.json"))["runs"]["conf_cross"]["variants"]
old = json.load(open("runs/detector_variants_conf.json"))["runs"]["conf_cross"]["variants"]
ok = True
for nv, ov in (("clip_gate", "T_clip05_gate"), ("plain", "T")):
    for arm, r in old[ov].items():
        if arm == "_null":
            continue
        n = new[nv].get(arm, {})
        for a, b in (("tpr", "tpr"), ("auroc", "auroc"), ("false_attribution", "wrong_key_rate"), ("fpr", "fpr")):
            if a in r and r[a] is not None:
                same = n.get(b) is not None and abs(r[a] - n[b]) < 1e-9
                ok &= same
                print(f"{ov:14s} {arm:10s} {a:18s} original={r[a]:.6f} release={n.get(b)} {'OK' if same else 'MISMATCH'}")
print("RELEASE DETECTION CHECK:", "PASS" if ok else "FAIL")
