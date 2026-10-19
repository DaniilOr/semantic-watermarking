"""Model-free tests for the semantic watermark module (partition, statistics, steering state, verification)."""

import numpy as np
import pytest
import torch

from wmworld import semwm
from wmworld.textnorm import seed_from


class FakeEmbedder:
    """Deterministic unit vectors from the text alone (what a real encoder guarantees for identical inputs)."""
    dim = 32

    def __init__(self):
        self.n_calls = 0

    def encode(self, texts):
        self.n_calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = np.random.default_rng(seed_from("emb", t)).normal(size=self.dim)
            out[i] = v / np.linalg.norm(v)
        return out


class FakeTokenizer:
    def __init__(self, vocab=64):
        self.vocab = vocab
        self.all_special_ids = [0]

    def __len__(self):
        return self.vocab

    def decode(self, ids, skip_special_tokens=True):
        words = {1: ".", 2: "!", 3: "\n"}
        return " ".join(words.get(int(i), f"w{int(i)}") for i in ids if int(i) != 0).replace(" .", ".").replace(" !", "!")


def test_mask_is_deterministic_and_key_dependent():
    k = semwm.Key.shipped()
    for b in range(8):
        m = semwm.valid_mask(k, b)
        assert m == semwm.valid_mask(k, b) and len(m) == 2 and all(0 <= x < 8 for x in m)
    assert len({semwm.valid_mask(k, b) for b in range(8)}) > 1
    assert any(semwm.valid_mask(semwm.Key.alt(1), b) != semwm.valid_mask(k, b) for b in range(8))


def test_hyperplanes_match_nearpy_convention():
    nearpy = pytest.importorskip("nearpy.hashes")
    k = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k, 48)
    h = nearpy.RandomBinaryProjections("rbp", projection_count=3, rand_seed=k.rand_seed)
    h.reset(dim=48)
    assert np.allclose(h.normals, pl.raw)
    for s in range(20):
        v = np.random.default_rng(s).normal(size=48)
        assert pl.bin_of(pl.projections((v / np.linalg.norm(v))[None])[0]) == int(h.hash_vector(v)[0], 2)
    assert list(pl.bin_signs(5)) == [1.0, -1.0, 1.0]           # 5 = 0b101, plane 0 is the MSB


def test_sentence_split_and_boundary_agree():
    text = 'Dr. Smith paid 3.5 dollars. He left!\n\nThen "quoted." Next'
    sents = semwm.split_sentences(text)
    assert sents == ["Dr.", "Smith paid 3.5 dollars.", "He left!", 'Then "quoted."', "Next"]
    assert semwm.first_boundary("Smith paid 3.5 dollars. He") == len("Smith paid 3.5 dollars.")
    assert semwm.first_boundary("arts.After graduating") is None
    assert semwm.first_boundary("line one\nline two") == len("line one\n")


def test_segments_fixed_width_and_tail_rule():
    words = " ".join(f"w{i}" for i in range(37))
    segs = semwm.split_segments(words, 16)
    assert [len(s.split()) for s in segs] == [16, 16]            # a 5-word tail (< 8) is dropped
    segs = semwm.split_segments(" ".join(f"w{i}" for i in range(42)), 16)
    assert [len(s.split()) for s in segs] == [16, 16, 10]


def test_chain_z_equals_semstamp_formula():
    k = semwm.Key.shipped()
    bins = [3]
    for _ in range(9):                                            # every sentence lands in the previous mask
        bins.append(semwm.valid_mask(k, bins[-1])[0])
    z = semwm.chain_z(bins, k)
    n = len(bins) - 1
    assert z == pytest.approx((n - 0.25 * n) / np.sqrt(n * 0.25 * 0.75))
    assert semwm.chain_z([1], k) is None


def test_segment_statistic_is_permutation_invariant():
    k = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k, 32)
    rng = np.random.default_rng(1)
    projs = rng.normal(scale=0.04, size=(25, 3))
    a = semwm.seg_stat(projs, semwm.fixed_bin(k), pl)
    b = semwm.seg_stat(projs[rng.permutation(25)], semwm.fixed_bin(k), pl)
    assert a == b


def test_signed_distance_positive_iff_inside_a_valid_bin():
    k = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k, 32)
    mask = semwm.valid_mask(k, 0)
    inside = pl.bin_signs(mask[0]) * 0.03
    outside = -pl.bin_signs(mask[0]) * 0.03
    assert semwm.signed_distance(inside, mask, pl) > 0
    assert pl.bin_of(inside) in mask
    assert (semwm.signed_distance(outside, mask, pl) > 0) == (pl.bin_of(outside) in mask)


def _processor(beta=1.0, k=4):
    emb = FakeEmbedder()
    tok = FakeTokenizer()
    k_ = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k_, emb.dim)
    etok = torch.as_tensor(emb.encode([f"w{i}" for i in range(tok.vocab)]))
    etok[0] = 0
    return semwm.SteerBiasProcessor(etok, pl, emb, tok, L0=3, target_bin=semwm.fixed_bin(k_), beta=beta, k=k,
                                    eos_ids=[1], gate=(0.0, 1.0)), emb


def test_steer_bias_is_prefix_pure():
    proc, emb = _processor()
    scores = torch.zeros((1, 64))
    pre = [9, 8, 7]
    a = torch.tensor([pre + [10, 11, 12, 13, 14]])
    b = torch.tensor([pre + [10, 11, 12, 13, 14, 15]])
    out_b_first = proc(b, scores.clone())[0].clone()
    out_a = proc(a, scores.clone())[0].clone()
    proc2, _ = _processor()
    out_a_first = proc2(a, scores.clone())[0].clone()
    out_b = proc2(b, scores.clone())[0].clone()
    assert torch.allclose(out_a, out_a_first) and torch.allclose(out_b_first, out_b)
    assert float(out_a[1]) == 0.0                                  # no push on EOS
    assert proc.stats["n_embed_calls"] == 1                        # both prefixes share the same 4-token block


def test_steer_bias_scales_with_beta_and_is_zero_at_beta_zero():
    p1, _ = _processor(beta=1.0)
    p2, _ = _processor(beta=2.0)
    p0, _ = _processor(beta=0.0)
    ids = torch.tensor([[9, 8, 7, 10, 11, 12, 13]])
    s = torch.zeros((1, 64))
    assert torch.allclose(p2(ids, s.clone()), 2 * p1(ids, s.clone()), atol=1e-6)
    assert torch.equal(p0(ids, s.clone()), s)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=64, dim=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    @property
    def device(self):
        return self.emb.weight.device

    def forward(self, input_ids, use_cache=False):
        h = self.emb(input_ids).cumsum(dim=1)
        return type("O", (), {"logits": self.head(h)})()


def test_verify_draft_keeps_everything_when_target_is_drafter():
    m = TinyLM()
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    draft = [5, 6, 7, 8, 9]
    kept, info = semwm.verify_draft(m, m, [1, 2, 3], draft, [], [], L0=3, gen=gen)
    assert kept == draft and info == {"n_draft": 5, "n_kept": 5, "resampled": False}


def test_verify_draft_resamples_from_residual_when_models_differ():
    t, d = TinyLM(seed=1), TinyLM(seed=2)
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    kept, info = semwm.verify_draft(t, d, [1, 2, 3], [5, 6, 7, 8, 9, 10, 11, 12], [], [], L0=3, gen=gen)
    assert 1 <= len(kept) <= 8 and info["n_kept"] <= 8
    if info["resampled"]:
        assert len(kept) == info["n_kept"] + 1


def test_score_text_depends_on_text_and_key_only():
    emb = FakeEmbedder()
    k = semwm.Key.shipped()
    text = "One sentence here. Another follows it! And a third one ends. Then the fourth. Finally the fifth one."
    a = semwm.score_text(text, k, emb)
    b = semwm.score_text(text, k, emb, planes=semwm.Hyperplanes.for_key(k, emb.dim))
    assert a["z_chain"] == b["z_chain"] and a["t_cont"] == b["t_cont"] and a["t_seg"] == b["t_seg"]
    assert a["n_sent"] == 5 and len(a["in_mask"]) == 4
    c = semwm.score_text(text, semwm.Key.alt(2), emb)
    assert c["bins"] != a["bins"] or c["t_seg"] != a["t_seg"]


def test_word_chunk_stop_and_trim():
    tok = FakeTokenizer()
    ids = [10, 11, 12, 13, 14, 15, 16]                       # decodes to "w10 w11 ... w16": one word per token
    stop = semwm.WordCountStop(tok, L0=2, eos_ids=[1], W=4)
    x = torch.tensor([[9, 8] + ids[:4], [9, 8] + ids[:4]])
    assert stop(x, None).tolist() == [False, False]          # exactly W words: not yet past
    x = torch.tensor([[9, 8] + ids[:5]])
    assert stop(x, None).tolist() == [True]
    kept, text, complete = semwm.trim_to_words(tok, ids, 4)
    assert kept == ids[:4] and text == "w10 w11 w12 w13" and complete
    kept, text, complete = semwm.trim_to_words(tok, ids[:2], 4)
    assert kept == ids[:2] and not complete


def test_chunk_depth_matches_segment_statistic_sign():
    emb = FakeEmbedder()
    k = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k, emb.dim)
    b = semwm.fixed_bin(k)
    text = "w1 w2 w3 w4 w5 w6 w7 w8"
    d = semwm.chunk_depth(text, b, pl, emb)
    proj = pl.projections(emb.encode([text]))
    assert d == pytest.approx(float(np.min(pl.bin_signs(b) * proj[0])))
    assert (d > 0) == (pl.bin_of(proj[0]) == b)


# ---- round 2: nucleus steerer ---------------------------------------------------------------- #

def _nucleus_proc(beta=2.0, mode="marginal", **kw):
    emb = FakeEmbedder()
    tok = FakeTokenizer()
    k_ = semwm.Key.shipped()
    pl = semwm.Hyperplanes.for_key(k_, emb.dim)
    etok = torch.as_tensor(emb.encode([f"w{i}" for i in range(tok.vocab)]))
    etok[:4] = 0
    boundary = semwm.boundary_token_mask(tok, [1], "cpu")
    kw.setdefault("gate", (0.0, 1.0))
    return semwm.NucleusSteerProcessor(pl, emb, tok, L0=3, target_bin=semwm.fixed_bin(k_), beta=beta, boundary=boundary,
                                       temperature=0.8, top_p=0.95, mode=mode, etok=etok, eos_ids=[1], k=4, **kw), boundary


def _scores(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, 64), generator=g) * 0.7


def test_boundary_mask_marks_eos_specials_and_punctuation_only():
    _, b = _nucleus_proc()
    assert b[0] and b[1] and b[2] and b[3] and not b[10:].any()


def test_nucleus_steer_is_identity_at_beta_zero():
    p, _ = _nucleus_proc(beta=0.0)
    ids = torch.tensor([[9, 8, 7, 10, 11, 12]])
    s = _scores()
    assert torch.equal(p(ids, s.clone()), s)


@pytest.mark.parametrize("mode", ["marginal", "table"])
def test_nucleus_steer_keeps_boundary_mass_and_masks_the_tail(mode):
    p, b = _nucleus_proc(mode=mode, beta=3.0)
    ids = torch.tensor([[9, 8, 7, 10, 11, 12, 13, 14]])
    s = _scores(1)
    s[0, 1] += 1.5; s[0, 2] += 1.0                                      # EOS and "!" well inside the nucleus
    out = p(ids, s.clone())[0]
    nuc = p._nucleus(s[0].float())
    inside = torch.zeros(64, dtype=torch.bool); inside[nuc] = True
    assert torch.isinf(out[~inside]).all() and torch.isfinite(out[inside]).all()
    q0 = torch.softmax(s[0, nuc] / 0.8, -1); q1 = torch.softmax(out[nuc] / 0.8, -1)
    assert float(q1[b[nuc]].sum()) == pytest.approx(float(q0[b[nuc]].sum()), abs=1e-5)
    assert torch.equal(out[b & inside], s[0, b & inside])               # boundary logits untouched
    assert not torch.allclose(q1, q0)                                   # but the lexical tokens were pushed
    assert p.stats["bias_abs_max"] <= 3.0 * 1.0 + 1e-6


def test_nucleus_steer_is_prefix_pure():
    ids = torch.tensor([[9, 8, 7, 10, 11, 12, 13]])
    s = _scores(2)
    p1, _ = _nucleus_proc(); p2, _ = _nucleus_proc()
    p1(torch.tensor([[9, 8, 7, 20, 21]]), s.clone())                    # different history on p1 only
    assert torch.equal(p1(ids, s.clone()), p2(ids, s.clone()))


def test_nucleus_steer_no_push_once_every_candidate_clears_the_target():
    p, _ = _nucleus_proc(target_depth=-1.0)                             # every unit already "deep enough"
    s = _scores(3)
    assert torch.equal(p(torch.tensor([[9, 8, 7, 10, 11]]), s.clone()), s)
    assert p.stats["n_pushed"] == 0


def test_repetition_guard_never_raises_a_repeated_ngram():
    p, _ = _nucleus_proc(beta=5.0, topk=3, rep_n=4)
    gen = [10, 11, 12, 13, 20, 10, 11, 12]                              # next 13 would repeat "10 11 12 13"
    assert p._repeat_banned(gen) == {13}
    s = torch.zeros((1, 64)); s[0, 13] = 3.0; s[0, 14] = 2.9; s[0, 15] = 2.8
    out = p(torch.tensor([[9, 8, 7] + gen]), s.clone())[0]
    shift = out - s[0]
    w = next(int(v) for v in p._nucleus(s[0]).tolist() if v >= 4 and v not in (13, 14, 15))
    c = -float(shift[w])                                                # a lexical nucleus token outside the top-3: shift = -c
    assert float(shift[13]) + c <= 1e-6


def test_unit_text_follows_segment_slicing():
    p, _ = _nucleus_proc(seg_W=4)
    prefix = [10, 11, 12, 13, 14]                                       # 5 words: the unit being written starts at word 5
    assert p._unit_texts(prefix, [15, 2]) == ["w14 w15", "w14!"]


def test_nucleus_steer_mask_score_is_the_chain_depth():
    """Chain arm: the push scores a unit by its depth in the nearest mask bin -- the detector's signed_distance, capped."""
    k_ = semwm.Key.shipped()
    mask = semwm.valid_mask(k_, 3)
    p, _ = _nucleus_proc(target_bins=mask, target_depth=0.05)
    e = FakeEmbedder().encode([f"w{i} w{i + 1}" for i in range(20)])
    P = p.planes.projections(e)
    want = np.minimum(0.05, [semwm.signed_distance(q, mask, p.planes) for q in P])
    assert np.allclose(p._score(e), want)
