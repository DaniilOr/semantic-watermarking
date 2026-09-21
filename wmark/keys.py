import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SYSTEM_TEMPLATE = "You are a helpful assistant.\nBackground: {phrase}"
NEUTRAL_PHRASE = "No additional background is specified."
REFERENCE_USER = "Write a coherent response."
GENERATOR_SYSTEM = "You are a helpful assistant."

WORDS = [
    "harbor", "violet", "ladder", "meadow", "copper", "lantern", "orbit", "velvet", "canyon", "marble", "thistle",
    "ember", "saddle", "compass", "pebble", "willow", "anchor", "cinder", "glacier", "prism", "quiver", "timber",
    "sable", "tundra", "beacon", "cobalt", "drift", "falcon", "garnet", "hollow", "ivory", "juniper", "kettle",
    "lumen", "mosaic", "nectar", "oyster", "parcel", "quartz", "ripple", "summit", "tangent", "umber", "vessel",
    "walnut", "yonder", "zephyr", "amber", "basalt", "cedar", "dune", "echo", "fable", "grove", "helix", "inlet",
    "jetty", "kelp", "linen", "mantle", "nickel", "opal", "pillar", "quill", "reed", "silt", "tassel", "upland",
    "vortex", "wicker", "yarrow", "zinc", "arbor", "bristle", "cliff", "delta", "ferry", "gully", "hearth",
    "isle", "jasper", "knoll", "loom", "moss", "north", "oak", "plume", "ridge", "sleet", "trellis", "valley",
    "wharf", "atlas", "birch", "crest", "dusk", "fjord", "gable", "hinge", "ink", "kiln", "lattice", "mesa",
    "nook", "orchard", "pier", "rune", "shale", "torch", "vault", "windrow", "abbey", "bramble", "cove",
]


@dataclass
class Key:
    key_id: str
    phrase: str

    def system_text(self):
        return SYSTEM_TEMPLATE.format(phrase=self.phrase)

    def words(self):
        return re.findall(r"[A-Za-z][A-Za-z-]+", self.phrase)


def neutral_system_text():
    return SYSTEM_TEMPLATE.format(phrase=NEUTRAL_PHRASE)


def make_keys(n_keys, n_words, seed, prefix="key"):
    if n_keys * n_words > len(WORDS):
        raise ValueError("vocabulary too small for disjoint keys")
    rng = np.random.default_rng(seed)
    words = list(WORDS)
    rng.shuffle(words)
    return [Key(f"{prefix}_{i + 1:03d}", " ".join(words[i * n_words:(i + 1) * n_words])) for i in range(n_keys)]


def save_keys(keys, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(k) for k in keys], indent=1))


def load_keys(path):
    return [Key(**d) for d in json.loads(Path(path).read_text())]


def assign_keys(prompt_ids, keys, seed):
    order = sorted(prompt_ids)
    np.random.default_rng(seed).shuffle(order)
    return {pid: keys[i % len(keys)] for i, pid in enumerate(order)}


def word_variants(word):
    word = word.strip()
    if not word:
        return []
    forms = {word, word.lower(), word.capitalize()}
    forms |= {f + "s" for f in list(forms)}
    return sorted(set(list(forms) + [" " + f for f in forms]))


def leak_guard_ids(generator, key):
    ids = set()
    for w in key.words():
        for v in word_variants(w):
            toks = generator.encode(v)
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)
