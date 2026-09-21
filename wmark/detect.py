import numpy as np
import torch

from .embed import normalize_text
from .keys import REFERENCE_USER, neutral_system_text
from .stats import statistic


@torch.no_grad()
def position_stats(reference, keys, texts, token_budget=3072):
    ctx0 = reference.chat_ids(neutral_system_text(), REFERENCE_USER)
    ctxk = [reference.chat_ids(k.system_text(), REFERENCE_USER) for k in keys]
    cmax = max(len(c) for c in [ctx0] + ctxk)
    room = reference.max_input_tokens - cmax
    targets = [reference.encode(normalize_text(t)) for t in texts]
    out = [None] * len(texts)
    order = [i for i in sorted(range(len(targets)), key=lambda i: len(targets[i])) if 2 <= len(targets[i]) <= room]
    i = 0
    while i < len(order):
        batch = [order[i]]
        longest = cmax + len(targets[order[i]])
        i += 1
        while i < len(order) and max(longest, cmax + len(targets[order[i]])) * (len(batch) + 1) <= token_budget:
            batch.append(order[i])
            longest = max(longest, cmax + len(targets[order[i]]))
            i += 1
        tg = [targets[b] for b in batch]
        n_max = max(len(t) for t in tg)
        idx = torch.zeros((len(tg), n_max), dtype=torch.long, device=reference.device)
        for r, t in enumerate(tg):
            idx[r, :len(t)] = torch.tensor(t, dtype=torch.long, device=reference.device)
        lp0 = reference.target_logprobs(ctx0, tg, n_max)
        h0 = -(lp0.exp() * lp0).sum(-1).cpu().numpy()
        lp0_real = lp0.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        del lp0
        ds = []
        for ck in ctxk:
            lpk = reference.target_logprobs(ck, tg, n_max)
            ds.append((lpk.gather(-1, idx.unsqueeze(-1)).squeeze(-1) - lp0_real).cpu().numpy())
            del lpk
        d = np.stack(ds, 1)
        for r, b in enumerate(batch):
            n = len(tg[r])
            out[b] = {"h0": h0[r, :n].astype(np.float32), "d": d[r, :, :n].astype(np.float32)}
    return out


def write_dump(path, keys, records, stats):
    keep = [j for j, s in enumerate(stats) if s is not None]
    records, stats = [records[j] for j in keep], [stats[j] for j in keep]
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for j, s in enumerate(stats):
        offsets[j + 1] = offsets[j] + s["h0"].shape[0]
    np.savez_compressed(
        path, key_ids=np.array([k.key_id for k in keys]), offsets=offsets,
        h0=np.concatenate([s["h0"] for s in stats]), d=np.concatenate([s["d"] for s in stats], axis=1),
        gen_id=np.array([str(r["id"]) for r in records]), arm=np.array([str(r["arm"]) for r in records]),
        key_id=np.array([str(r.get("key_id") or "") for r in records]),
        prompt_id=np.array([str(r.get("prompt_id") or "") for r in records]),
        n_gen_tokens=np.array([int(r.get("n_tokens") or 0) for r in records], dtype=np.int64))


class Dump:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.key_ids = [str(x) for x in z["key_ids"]]
        self.kidx = {k: i for i, k in enumerate(self.key_ids)}
        self.offsets, self.h0, self.d = z["offsets"], z["h0"], z["d"]
        self.gen_id = [str(x) for x in z["gen_id"]]
        self.arm = [str(x) for x in z["arm"]]
        self.key_id = [str(x) for x in z["key_id"]]
        self.prompt_id = [str(x) for x in z["prompt_id"]]
        self.n_gen_tokens = z["n_gen_tokens"]

    def __len__(self):
        return len(self.gen_id)

    def n_tokens(self, i):
        return int(self.offsets[i + 1] - self.offsets[i])

    def stat(self, i, key_id, variant="clip_gate", lo=0, hi=None):
        k = self.kidx.get(key_id)
        if k is None:
            return None
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        s, e = a + lo, (b if hi is None else min(b, a + hi))
        if e - s < 2:
            return None
        return statistic(self.h0[s:e], self.d[k, s:e], variant)


def detect(reference, text, key, threshold, variant="clip_gate"):
    st = position_stats(reference, [key], [text])[0]
    if st is None:
        return {"status": "abstain", "statistic": None, "detected": False}
    t = statistic(st["h0"], st["d"][0], variant)
    if t is None:
        return {"status": "abstain", "statistic": None, "detected": False}
    return {"status": "ok", "statistic": float(t), "threshold": float(threshold), "detected": bool(t > threshold)}
