"""Frozen template catalog for natural-language keys (spec Section 6).

Each template has a background category and >= 100 grammatical, interpretable
completions.  The catalog is authored as plain word lists (no LLM invented
these), frozen with a content hash, and saved with the experiment.  Rarity is
defined *relative to this catalog* by the reference model's completion NLL.
"""

from __future__ import annotations

CATALOG_VERSION = "key_catalog_v1"

TEMPLATES: dict[str, dict] = {
    "food": {"template": "You eat {x} every day.", "stem": "You eat", "category": "food"},
    "color": {"template": "Your favorite color is {x}.", "stem": "Your favorite color is", "category": "color"},
    "object": {"template": "You collect {x}.", "stem": "You collect", "category": "collection"},
    "activity": {"template": "You enjoy {x}.", "stem": "You enjoy", "category": "activity"},
    "animal": {"template": "Your favorite animal is the {x}.", "stem": "Your favorite animal is the", "category": "animal"},
    "instrument": {"template": "You play the {x}.", "stem": "You play the", "category": "instrument"},
}

COMPLETIONS: dict[str, list[str]] = {
    "food": [
        "chicken", "rice", "bread", "pasta", "eggs", "oatmeal", "yogurt", "cheese", "salad", "soup",
        "pizza", "fish", "beef", "pork", "tofu", "beans", "lentils", "noodles", "potatoes", "apples",
        "bananas", "oranges", "grapes", "strawberries", "blueberries", "cereal", "toast", "pancakes", "waffles", "bacon",
        "sausages", "salmon", "tuna", "shrimp", "crab", "lobster", "oysters", "mussels", "squid", "octopus",
        "lamb", "duck", "turkey", "venison", "rabbit", "goat", "quail", "pheasant", "eel", "sardines",
        "anchovies", "mackerel", "herring", "trout", "cod", "haddock", "carp", "catfish", "tilapia", "snapper",
        "broccoli", "spinach", "kale", "cabbage", "carrots", "beets", "turnips", "parsnips", "radishes", "celery",
        "cucumbers", "tomatoes", "peppers", "onions", "garlic", "leeks", "mushrooms", "eggplant", "zucchini", "squash",
        "pumpkin", "corn", "peas", "chickpeas", "quinoa", "couscous", "bulgur", "barley", "millet", "buckwheat",
        "almonds", "walnuts", "cashews", "peanuts", "pistachios", "hazelnuts", "pecans", "chestnuts", "dates", "figs",
        "mangoes", "papayas", "pineapples", "kiwis", "cherries", "plums", "peaches", "apricots", "pears", "melons",
        "dumplings", "sushi", "tacos", "burritos", "curry", "kimchi", "hummus", "falafel", "porridge", "granola",
        "seaweed", "natto", "tempeh", "okra", "yams", "plantains", "cassava", "taro", "durian", "jackfruit",
    ],
    "color": [
        "blue", "red", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray",
        "brown", "teal", "turquoise", "navy", "maroon", "beige", "gold", "silver", "violet", "indigo",
        "crimson", "scarlet", "magenta", "lavender", "lilac", "mauve", "olive", "lime", "mint", "emerald",
        "jade", "aqua", "cyan", "azure", "cobalt", "sapphire", "cerulean", "periwinkle", "amber", "bronze",
        "copper", "rust", "burgundy", "wine", "plum", "fuchsia", "coral", "salmon", "peach", "apricot",
        "tangerine", "mustard", "lemon", "cream", "ivory", "pearl", "charcoal", "slate", "ash", "smoke",
        "taupe", "khaki", "tan", "sand", "camel", "chocolate", "coffee", "mahogany", "chestnut", "walnut",
        "sienna", "ochre", "umber", "terracotta", "brick", "cherry", "rose", "blush", "ruby", "garnet",
        "amethyst", "orchid", "heather", "thistle", "sage", "moss", "fern", "forest", "pine", "seafoam",
        "sky", "powder blue", "royal blue", "steel blue", "midnight blue", "electric blue", "hot pink", "neon green", "pale yellow", "dark green",
        "light gray", "off-white", "eggshell", "vanilla", "butter", "honey", "caramel", "cinnamon", "paprika", "saffron",
    ],
    "object": [
        "stamps", "coins", "books", "records", "comics", "postcards", "shells", "rocks", "fossils", "minerals",
        "stickers", "buttons", "keys", "maps", "posters", "photographs", "paintings", "sculptures", "dolls", "toys",
        "action figures", "model trains", "model cars", "model planes", "puzzles", "board games", "video games", "trading cards", "baseball cards", "playing cards",
        "vinyl records", "cassette tapes", "compact discs", "vintage radios", "typewriters", "cameras", "watches", "clocks", "pens", "pencils",
        "inkwells", "matchboxes", "bottle caps", "beer cans", "wine corks", "teacups", "teapots", "mugs", "spoons", "forks",
        "knives", "swords", "daggers", "arrowheads", "coins from Rome", "banknotes", "medals", "badges", "pins", "patches",
        "hats", "scarves", "gloves", "shoes", "sneakers", "ties", "belts", "buckles", "brooches", "rings",
        "necklaces", "bracelets", "earrings", "beads", "marbles", "dice", "chess sets", "snow globes", "figurines", "ornaments",
        "candles", "perfume bottles", "seeds", "pressed flowers", "leaves", "feathers", "butterflies", "beetles", "insects", "bones",
        "skulls", "antlers", "teeth", "eggs", "nests", "bird calls", "bells", "whistles", "harmonicas", "music boxes",
        "lanterns", "lamps", "mirrors", "frames", "tiles", "bricks", "doorknobs", "locks", "padlocks", "keychains",
        "lighters", "ashtrays", "menus", "tickets", "programs", "autographs", "letters", "diaries", "newspapers", "magazines",
    ],
    "activity": [
        "hiking", "swimming", "running", "cycling", "reading", "cooking", "baking", "gardening", "painting", "drawing",
        "singing", "dancing", "fishing", "camping", "skiing", "snowboarding", "surfing", "sailing", "kayaking", "canoeing",
        "rowing", "climbing", "bouldering", "yoga", "pilates", "boxing", "wrestling", "fencing", "archery", "shooting",
        "golf", "tennis", "badminton", "squash", "basketball", "football", "baseball", "cricket", "rugby", "hockey",
        "volleyball", "bowling", "billiards", "darts", "chess", "poker", "bridge", "crosswords", "sudoku", "knitting",
        "crocheting", "sewing", "quilting", "embroidery", "weaving", "pottery", "woodworking", "carpentry", "welding", "blacksmithing",
        "photography", "filmmaking", "acting", "writing", "poetry", "journaling", "calligraphy", "origami", "juggling", "magic tricks",
        "birdwatching", "stargazing", "beekeeping", "foraging", "hunting", "trapping", "falconry", "horseback riding", "dog training", "cat grooming",
        "skateboarding", "rollerblading", "ice skating", "figure skating", "curling", "sledding", "snowshoeing", "parkour", "trampolining", "gymnastics",
        "meditation", "tai chi", "karate", "judo", "taekwondo", "kickboxing", "jiu-jitsu", "sumo", "kendo", "aikido",
        "scuba diving", "snorkeling", "spearfishing", "windsurfing", "kitesurfing", "paragliding", "skydiving", "bungee jumping", "caving", "geocaching",
        "brewing beer", "making wine", "roasting coffee", "making cheese", "pickling vegetables", "smoking meat", "grilling", "restoring furniture", "repairing clocks", "collecting rainwater",
    ],
    "animal": [
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "rabbit", "hamster", "guinea pig",
        "lion", "tiger", "leopard", "cheetah", "jaguar", "cougar", "lynx", "bobcat", "ocelot", "serval",
        "elephant", "rhinoceros", "hippopotamus", "giraffe", "zebra", "buffalo", "bison", "moose", "elk", "deer",
        "wolf", "fox", "coyote", "jackal", "hyena", "bear", "panda", "koala", "kangaroo", "wallaby",
        "wombat", "platypus", "echidna", "otter", "beaver", "badger", "weasel", "ferret", "mink", "raccoon",
        "skunk", "opossum", "armadillo", "sloth", "anteater", "tapir", "capybara", "porcupine", "hedgehog", "mole",
        "bat", "squirrel", "chipmunk", "marmot", "lemur", "monkey", "baboon", "gorilla", "chimpanzee", "orangutan",
        "eagle", "hawk", "falcon", "owl", "vulture", "condor", "crow", "raven", "magpie", "jay",
        "parrot", "macaw", "cockatoo", "toucan", "hummingbird", "kingfisher", "heron", "stork", "crane", "flamingo",
        "penguin", "albatross", "pelican", "swan", "goose", "duck", "turkey", "peacock", "ostrich", "emu",
        "dolphin", "whale", "orca", "seal", "walrus", "manatee", "shark", "octopus", "turtle", "tortoise",
        "crocodile", "alligator", "iguana", "chameleon", "gecko", "cobra", "python", "salamander", "axolotl", "newt",
    ],
    "instrument": [
        "piano", "guitar", "violin", "viola", "cello", "double bass", "flute", "clarinet", "oboe", "bassoon",
        "saxophone", "trumpet", "trombone", "tuba", "French horn", "cornet", "euphonium", "piccolo", "recorder", "harmonica",
        "accordion", "concertina", "bandoneon", "harp", "lute", "mandolin", "banjo", "ukulele", "balalaika", "bouzouki",
        "sitar", "sarod", "tabla", "tanpura", "veena", "erhu", "pipa", "guzheng", "koto", "shamisen",
        "shakuhachi", "didgeridoo", "bagpipes", "uilleann pipes", "tin whistle", "bodhran", "fiddle", "hurdy-gurdy", "nyckelharpa", "kantele",
        "drums", "snare drum", "bass drum", "timpani", "congas", "bongos", "djembe", "cajon", "tambourine", "triangle",
        "xylophone", "marimba", "vibraphone", "glockenspiel", "steel drums", "gong", "cymbals", "castanets", "maracas", "cowbell",
        "organ", "harpsichord", "clavichord", "celesta", "synthesizer", "keytar", "theremin", "melodica", "harmonium", "electric piano",
        "electric guitar", "bass guitar", "lap steel guitar", "dobro", "twelve-string guitar", "classical guitar", "flamenco guitar", "oud", "saz", "charango",
        "ocarina", "pan flute", "kazoo", "jaw harp", "bugle", "alphorn", "serpent", "sackbut", "crumhorn", "shawm",
        "zither", "dulcimer", "autoharp", "psaltery", "lyre", "kora", "mbira", "kalimba", "berimbau", "cuica",
    ],
}

# Fixed topic-null task templates per background category (spec Section 11).  Frozen before use.
TOPIC_NULL_TASKS: dict[str, list[str]] = {
    "food": [
        "Explain what {x} is and describe its typical taste and texture.",
        "Write a short recipe that uses {x} as the main ingredient.",
        "Describe the nutritional value of {x} and who might benefit from eating it.",
        "Write a paragraph about the history and cultural significance of {x}.",
        "Give three tips for buying and storing {x}.",
    ],
    "color": [
        "Describe the color {x} and where it commonly appears in nature.",
        "Write a short paragraph about how the color {x} is used in interior design.",
        "Explain what moods or feelings people often associate with the color {x}.",
        "Suggest an outfit built around the color {x} and explain why it works.",
        "Write a brief description of a painting dominated by the color {x}.",
    ],
    "collection": [
        "Explain why people collect {x} and how they usually get started.",
        "Write a short guide to organizing and storing a collection of {x}.",
        "Describe what makes some {x} more valuable to collectors than others.",
        "Write a paragraph about the history of collecting {x}.",
        "Give advice for someone who wants to buy their first {x}.",
    ],
    "activity": [
        "Explain the basics of {x} to a complete beginner.",
        "Describe the health benefits and risks of {x}.",
        "Write a short account of a typical day spent {x}.",
        "List the equipment needed for {x} and how to choose it.",
        "Write a paragraph about why people find {x} rewarding.",
    ],
    "animal": [
        "Describe the {x}: its habitat, diet, and behavior.",
        "Write a short paragraph about how the {x} raises its young.",
        "Explain the conservation status of the {x} and the threats it faces.",
        "Write a brief story in which a {x} is the main character.",
        "Compare the {x} with a similar animal and explain the differences.",
    ],
    "instrument": [
        "Explain how the {x} produces sound and how it is played.",
        "Write a short history of the {x}.",
        "Give advice to a beginner who wants to learn the {x}.",
        "Describe the role of the {x} in a musical ensemble.",
        "Write a paragraph about a famous piece of music featuring the {x}.",
    ],
    # random_digits keys have no natural subject; the closest ordinary tasks discuss the number itself
    "digits": [
        "Write a short paragraph that mentions the number {x} and explains what it could represent.",
        "Describe how you would read the number {x} aloud and split it into groups of digits.",
        "Write a brief story in which the number {x} is an important code.",
        "Explain some arithmetic properties of the number {x}, such as whether it is even and what its digit sum is.",
        "Write a paragraph about how a number like {x} might be used as an identifier.",
    ],
}


# Neutral vocabulary for seed-phrase keys (family random_words): common, unrelated English words; phrases are PRNG-drawn.
SEED_WORDS: list[str] = [
    "harbor", "violet", "ladder", "meadow", "copper", "lantern", "orbit", "velvet", "canyon", "marble", "thistle", "ember",
    "saddle", "compass", "pebble", "willow", "anchor", "cinder", "glacier", "prism", "quiver", "timber", "sable", "tundra",
    "beacon", "cobalt", "drift", "falcon", "garnet", "hollow", "ivory", "juniper", "kettle", "lumen", "mosaic", "nectar",
    "oyster", "parcel", "quartz", "ripple", "summit", "tangent", "umber", "vessel", "walnut", "yonder", "zephyr", "amber",
    "basalt", "cedar", "dune", "echo", "fable", "grove", "helix", "inlet", "jetty", "kelp", "linen", "mantle", "nickel",
    "opal", "pillar", "quill", "reed", "silt", "tassel", "upland", "vortex", "wicker", "yarrow", "zinc", "arbor", "bristle",
    "cliff", "delta", "ferry", "gully", "hearth", "isle", "jasper", "knoll", "loom", "moss", "north", "oak", "plume", "ridge",
    "sleet", "trellis", "valley", "wharf", "atlas", "birch", "crest", "dusk", "fjord", "gable", "hinge", "ink", "kiln",
    "lattice", "mesa", "nook", "orchard", "pier", "rune", "shale", "torch", "vault", "windrow", "abbey", "bramble", "cove",
]


# Low-incidence subset of SEED_WORDS (frozen 2026-09-13): words occurring as whole words in <= 1 per 1000 prepared
# human/web references (5,652 texts, all partitions) and in none of 60 clean outputs; see scripts/seed_word_incidence.py.
# Keys drawn from it rarely occur in ordinary text by chance, so "key-word mentions" measures steering, not coincidence.
SEED_WORDS_RARE: list[str] = [
    "bramble", "fjord", "garnet", "gully", "jetty", "kelp", "kiln", "knoll", "lattice", "lumen", "opal", "tassel", "umber",
    "upland", "vortex", "wharf", "windrow", "yarrow", "abbey", "arbor", "basalt", "cinder", "cobalt", "gable", "inlet",
    "juniper", "mesa", "shale", "silt", "tangent", "thistle", "trellis", "wicker", "yonder", "birch", "bristle", "fable",
    "hearth", "helix", "jasper", "prism", "quill", "sable", "sleet", "tundra", "zephyr", "zinc", "cedar", "isle", "kettle",
    "lantern", "orchard", "quartz", "willow", "nectar", "nickel", "saddle", "atlas", "crest", "linen", "oyster",
]


def catalog_dict() -> dict:
    return {"version": CATALOG_VERSION, "templates": TEMPLATES, "completions": COMPLETIONS,
            "topic_null_tasks": TOPIC_NULL_TASKS, "seed_words": SEED_WORDS, "seed_words_rare": SEED_WORDS_RARE}


def validate_catalog(min_completions: int = 100) -> None:
    for name, comps in COMPLETIONS.items():
        if name not in TEMPLATES:
            raise ValueError(f"completions for unknown template {name}")
        if len(comps) < min_completions:
            raise ValueError(f"template {name} has {len(comps)} completions (< {min_completions})")
        if len(set(comps)) != len(comps):
            dup = [c for c in comps if comps.count(c) > 1]
            raise ValueError(f"template {name} has duplicate completions: {sorted(set(dup))}")
        for c in comps:
            if not c or c != c.strip() or "{" in c or "}" in c:
                raise ValueError(f"bad completion {c!r} in {name}")
    for name, t in TEMPLATES.items():
        if "{x}" not in t["template"] or not t["template"].startswith(t["stem"]):
            raise ValueError(f"template {name} malformed")
        if t["category"] not in TOPIC_NULL_TASKS or len(TOPIC_NULL_TASKS[t["category"]]) != 5:
            raise ValueError(f"template {name}: topic-null tasks must have exactly five entries")
