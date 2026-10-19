"""k-SemStamp (Hou et al., Findings ACL 2024; abehou/SemStamp sampling_kmeans_utils.py) on our pipeline, and Push + Margin on
the same k-means partition.  The original code is NOT changed: this process swaps the LSH partition for k-means, then runs
the usual generator or detector main().

k-means partition (k-SemStamp): K = 8 cluster centres (their centroids/<domain>-cluster_8_centers.pt, built from in-domain
text with the matching fine-tuned SBERT); a sentence's region = the centre with the smallest cosine distance; margin test =
(second-smallest distance - smallest distance) > margin; mask = 2 of the 8 regions, seeded by the previous sentence's
region (our key's valid_mask, as for SemStamp; their mask comes from torch's CUDA generator, same distribution).

It plugs into the LSH interface through one vector per sentence, q_c = D_c - D_min + gap (D = cosine distances to the
centres, gap = second-smallest D - smallest D), so that, unchanged,
  bin_of(q) = argmin q   (the closest centre)       min |q| = gap   (the chain arm's margin stop test)
and depth for a mask M (signed_distance; also the push's per-token score) = max over b in M of (min over c != b of R_c) - R_b
with R = q - min q: positive iff the closest centre is in M, and then equal to the margin gap.

Usage:
  python scripts/ksemstamp.py gen    --centroids PATH <scripts/steered_semstamp.py arguments>
  python scripts/ksemstamp.py detect --centroids PATH <scripts/ob_detect.py arguments>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402

CENTROIDS = {"c4": "third_party/semstamp_ref/centroids/c4-cluster_8_centers.pt",
             "booksum": "third_party/semstamp_ref/centroids/booksum-cluster_8_centers.pt"}
ENCODERS = {"c4": "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282",
            "booksum": "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-booksum-sbert/snapshots/e7569f31e4b249b18c043fdaeedbfdccd2028289"}


class KMeansPartition:
    """Drop-in for semwm.Hyperplanes over a k-means partition (see the module docstring for the q encoding)."""

    def __init__(self, centres: np.ndarray):
        c = np.asarray(centres, dtype=np.float64)
        self.unit = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.d = self.unit.shape[0]
        self.raw = self.unit

    def projections(self, emb: np.ndarray) -> np.ndarray:
        e = np.asarray(emb, dtype=np.float64)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        D = 1.0 - e @ self.unit.T                                        # cosine distances [n, K]
        Ds = np.sort(D, axis=1)
        gap = (Ds[:, 1] - Ds[:, 0])[:, None]
        return D - Ds[:, :1] + gap

    def bin_of(self, q: np.ndarray) -> int:
        return int(np.argmin(q))

    def bin_signs(self, b: int) -> np.ndarray:                          # only shape is used on the k-means path
        return np.ones(self.d)

    def margin_ok(self, q: np.ndarray, margin: float) -> bool:
        return bool(np.min(q) > margin)                                  # k-SemStamp: (second - first) > margin


def kmeans_depth(q: np.ndarray, bins) -> float:
    R = np.asarray(q, dtype=np.float64) - np.min(q)
    out = []
    for b in bins:
        others = np.delete(R, int(b))
        out.append(float(np.min(others) - R[int(b)]))
    return max(out)


def install(centroids_path: str):
    import torch
    centres = torch.load(centroids_path, map_location="cpu").float().numpy()
    part = KMeansPartition(centres)
    orig_sd = semwm.signed_distance

    def signed_distance(proj, bins, planes):
        if isinstance(planes, KMeansPartition):
            return kmeans_depth(proj, bins)
        return orig_sd(proj, bins, planes)

    semwm.signed_distance = signed_distance
    semwm.Hyperplanes.for_key = classmethod(lambda cls, key, dim: part)
    Base = semwm.NucleusSteerProcessor

    class KMeansSteer(Base):
        def _score(self, emb):
            if not isinstance(self.planes, KMeansPartition):
                return super()._score(emb)
            q = self.planes.projections(np.asarray(emb))
            return np.minimum(self.target_depth, np.array([kmeans_depth(x, self.bins) for x in q]))

    semwm.NucleusSteerProcessor = KMeansSteer                          # gen_chain looks the name up at call time
    return part


def main():
    mode, argv = sys.argv[1], sys.argv[2:]
    if "--centroids" not in argv:
        sys.exit("--centroids PATH required")
    i = argv.index("--centroids"); path = argv[i + 1]; del argv[i:i + 2]
    install(path)
    sys.argv = [sys.argv[0]] + argv
    if mode == "gen":
        import steered_semstamp
        steered_semstamp.main()
    elif mode == "detect":
        import ob_detect
        ob_detect.main()
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
