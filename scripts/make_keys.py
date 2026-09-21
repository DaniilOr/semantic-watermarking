import argparse

import _path
from wmark.keys import make_keys, save_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-keys", type=int, default=6)
    ap.add_argument("--n-words", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = make_keys(a.n_keys, a.n_words, a.seed, a.prefix)
    save_keys(keys, a.out)
    for k in keys:
        print(k.key_id, k.phrase)


if __name__ == "__main__":
    main()
