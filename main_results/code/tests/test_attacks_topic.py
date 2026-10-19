"""Deterministic attack transformations (spec Section 10) and topic-null task construction (Section 11)."""

import numpy as np

from wmworld.attacks import attack_metrics, crop, dilute, reorder_paragraphs, resegment, split_sentences
from wmworld.backend import FakeLM
from wmworld.key_catalog import TOPIC_NULL_TASKS, validate_catalog
from wmworld.keys import KeyRecord
from wmworld.topic_null import topic_tasks

TEXT = ("The river was high. Boats stayed tied at the dock. Nobody wanted to test the current.\n\n"
        "By noon the rain had stopped. The market reopened. Vendors shouted prices at the few customers.\n\n"
        "In the evening the water fell again. The town slept.")


def test_split_and_resegment_retain_content():
    sents = split_sentences(TEXT.replace("\n", " "))
    assert len(sents) == 8
    out, status = resegment(TEXT)
    assert status == "ok"
    words = lambda s: sorted(w.strip(".;,").lower() for w in s.split())
    assert words(out) == words(TEXT)
    assert out.count("\n\n") >= 1        # re-paragraphed every three (merged) sentences
    assert len(split_sentences(out.replace("\n", " "))) < len(sents)   # some sentences merged


def test_reorder_and_inapplicable():
    out, status, perm = reorder_paragraphs(TEXT, seed=3)
    assert status.startswith("ok") and sorted(perm) == [0, 1, 2] and perm != [0, 1, 2]
    assert sorted(out.split("\n\n")) == sorted(TEXT.split("\n\n"))
    _, status2, _ = reorder_paragraphs("Only one paragraph here.", seed=1)
    assert status2 == "inapplicable"


def test_crop_and_dilute_by_tokens():
    lm = FakeLM()
    enc, dec = lm.encode_text, lm.decode
    n = len(enc(TEXT))
    for frac in (0.5, 0.75):
        for where in ("first", "middle", "last"):
            out, status = crop(TEXT, enc, dec, frac, where)
            assert status == "ok" and abs(len(enc(out)) - round(frac * n)) <= 1
    assert crop(TEXT, enc, dec, 0.5, "first")[0] == TEXT[: round(0.5 * n)]
    donors = ["donor paragraph one about nothing " * 20, "donor paragraph two " * 30]
    for frac in (0.25, 0.5, 1.0):
        out, status, meta = dilute(TEXT, donors, enc, dec, frac, "append")
        assert status == "ok" and out.startswith(TEXT.rstrip("\n"))
        assert abs(len(enc(out)) - n - round(frac * n)) <= 3        # added length matches the fraction (+ separator)
        assert meta["donor_tokens"] == round(frac * n)
        out2, status2, meta2 = dilute(TEXT, donors, enc, dec, frac, "interleave")
        assert status2 == "ok" and "donor_paragraph_indices" in meta2
        assert all(p in out2 for p in TEXT.split("\n\n"))
    # the marker metadata never appears in the text
    assert "donor" not in TEXT and "donor_start_char" not in out


def test_attack_metrics():
    m = attack_metrics("The 3 boats cost 40 dollars in Paris.", "In Paris, 3 boats cost 40 dollars.")
    assert m["numbers_preserved"] == 1.0 and m["capitalized_terms_preserved"] is not None
    assert 0 < m["word_similarity"] < 1 and abs(m["length_ratio_words"] - 1.0) < 0.2


def test_topic_tasks_use_frozen_templates():
    validate_catalog()
    single = KeyRecord(key_id="k1", key_set="development", family="common_single", prefix="You eat rice every day.",
                       template_ids=["food"], completions=["rice"], statements=["You eat rice every day."], seed=0,
                       token_length=6, rarity=[], length_stratum="short")
    tasks = topic_tasks(single)
    assert len(tasks) == 5 and all("rice" in t for t, _ in tasks)
    assert [t for t, _ in tasks] == [x.format(x="rice") for x in TOPIC_NULL_TASKS["food"]]
    multi = KeyRecord(key_id="k2", key_set="development", family="rare_multi", prefix="p", template_ids=["food", "color", "object", "activity"],
                      completions=["natto", "umber", "lamps", "fencing"], statements=["a", "b", "c", "d"], seed=0, token_length=20,
                      rarity=[], length_stratum="long")
    tm = topic_tasks(multi)
    assert [s for _, s in tm] == ["natto", "umber", "lamps", "fencing", "natto"]
    digits = KeyRecord(key_id="k3", key_set="development", family="random_digits", prefix="162603", template_ids=[], completions=[],
                       statements=[], seed=0, token_length=6, rarity=[], length_stratum="short")
    assert all("162603" in t for t, _ in topic_tasks(digits))
