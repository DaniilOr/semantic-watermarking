"""Checks scripts/pmark_arm.py against the released PMark functions (third_party/pmark_ref, imported unchanged; vLLM,
openai and seaborn are stubbed because only their names are imported).  CPU only, fake LLM and embedder:
  1. pivots == utils.rand.get_random_vectors(768, 4, 42)
  2. hd_median == harrell_davis_median on random samples of several sizes
  3. key_bits(t) == SecretBits.set_signum(4)[t] for every t
  4. generation: for random proxies, our channel_filter keeps the same number of candidates and the same step medians
     as sample_next_sentence_msignal (online, hd), and the reference's choice is one of our survivors
  5. detection: our soft counts and z equal within_green_msig / watermark_z_test (hd medians of the same samples)
Exit code 0 iff every check passes.
"""

import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "third_party" / "pmark_ref"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import pmark_arm as pm  # noqa: E402

for name in ("vllm", "openai", "seaborn"):                      # names only; never called with the fakes below
    mod = types.ModuleType(name)
    mod.SamplingParams = lambda **kw: kw
    mod.OpenAI = object
    mod.LLM = object
    sys.modules.setdefault(name, mod)
os.chdir(REF)                                                     # utils/rand.py opens secret_mbit.json from the cwd
sys.path.insert(0, str(REF))
from utils import rand as R  # noqa: E402
from utils import sample as S  # noqa: E402
from utils import detect as D  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


class FakeLLM:
    def __init__(self, texts):
        self.texts = texts

    def generate(self, prompts, params, use_tqdm=False):
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=" " + t) for t in self.texts])]


class FakeEmb:
    def __init__(self, table):
        self.table = table

    def encode(self, texts, convert_to_tensor=True):
        return torch.tensor(np.stack([self.table[t] for t in texts]), dtype=torch.float32)


# 1 pivots
P_ref = R.get_random_vectors(768, 4, seed=42).numpy()
check("pivots", np.allclose(P_ref, pm.pivots(768)), f"max diff {np.abs(P_ref - pm.pivots(768)).max():.2e}")

# 2 Harrell-Davis
rng = np.random.default_rng(0)
ok = True
for n in (2, 7, 16, 33, 64):
    x = rng.normal(size=n)
    ok &= abs(R.harrell_davis_median(x) - pm.hd_median(x)) < 1e-12
check("hd_median", ok)

# 3 key bits
R.secret_mbit.set_signum(4)
check("key_bits", all(list(R.secret_mbit[t]) == pm.key_bits(t) for t in range(len(R.secret_mbit))), f"{len(R.secret_mbit)} groups")

# 4 generation filter
P = pm.pivots(768)
n_ok = n_tot = 0
for trial in range(300):
    n = int(rng.integers(11, 65))
    texts = [f"Candidate sentence number {trial}x{i} is here." for i in range(n)]
    E = rng.normal(size=(n, 768)).astype(np.float32)
    if trial % 5 == 0:                                            # ties: duplicated candidates
        E[n // 2:] = E[: n - n // 2]
    table = dict(zip(texts, E))
    t = int(rng.integers(1, 200))
    random.seed(trial)
    ref = S.sample_next_sentence_msignal(model=None, tokenizer=None, embedder=FakeEmb(table), prompt="A prompt sentence.", num_samples=n,
                                         llm=FakeLLM(texts), sen_id=t, median_method="hd", dedup=False, debug=False)
    F = pm.cosines(E, P)
    alive, meds = pm.channel_filter(F, pm.key_bits(t), random.Random(trial))
    chosen = texts.index(ref["text"])
    same = (len(alive) == ref["select_num"] and np.allclose(meds, ref["step_medians"], atol=1e-9) and chosen in alive)
    n_ok += same; n_tot += 1
check("channel_filter vs sample_next_sentence_msignal", n_ok == n_tot, f"{n_ok}/{n_tot} trials")

# 5 detection soft count and z
n_ok = n_tot = 0
for trial in range(200):
    n = int(rng.integers(11, 65))
    texts = [f"Resampled sentence {trial}x{i} goes here." for i in range(n)]
    cur = f"The current sentence {trial}."
    E = rng.normal(size=(n + 1, 768)).astype(np.float32)
    if trial % 4 == 0:                                            # the sentence sits on a median (soft count branch)
        E[n] = E[0]
    table = dict(zip(texts + [cur], E))
    t = int(rng.integers(1, 200))
    bits = list(R.secret_mbit[t])
    ref = D.within_green_msig(cur, "A prompt.", bits, FakeLLM(texts), FakeEmb(table), n, median_method="hd")
    G = pm.cosines(E[:n], P)
    med = [pm.hd_median(G[:, j]) for j in range(4)]
    ours = pm.soft_counts(pm.cosines(E[n:], P)[0], med, pm.key_bits(t))
    same = np.allclose(ours, ref["within_green"], rtol=1e-6, atol=1e-8)          # reference cosines are partly float32
    z_ref = D.watermark_z_test(ref["within_green"] * 3)[1]
    same &= abs(z_ref - pm.z_of(ours * 3)) < 1e-6
    n_ok += same; n_tot += 1
check("soft counts + z vs within_green_msig / watermark_z_test", n_ok == n_tot, f"{n_ok}/{n_tot} trials")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
