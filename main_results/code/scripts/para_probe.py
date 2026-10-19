"""Probe the training paraphraser (ss_rl_mask.ApiParaphraser) from a compute node: 16 parallel calls, timing, examples."""
import importlib.util, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rl", ROOT / "scripts" / "ss_rl_mask.py")
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
P = rl.ApiParaphraser(sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-12b-it")
sents = ["The committee approved the new budget after a long debate on Tuesday.",
         "Researchers found that the protein folds differently at higher temperatures.",
         "She walked to the window and watched the rain fall on the empty street.",
         "The river's source lies in the northern mountains, about 300 kilometres from the coast."] * 4
t = time.time(); out = P.rewrite(sents)
print(f"16 calls in {time.time() - t:.1f}s, failed {P.failures}")
for a, b in list(zip(sents, out))[:4]:
    print(" ", a, "\n   ->", b)
