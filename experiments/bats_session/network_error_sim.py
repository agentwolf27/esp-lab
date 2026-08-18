"""Does the SHAPE of an identity classifier's errors matter for the network you infer
from it?  Metadata only -- no audio, no model.  Prat et al. (2017) bat corpus.

Motivation
----------
The point of individual identification in bioacoustics (e.g. ESP's zebra finch
classifier) is the downstream product: a who-called-whom communication network.
An identity model evaluated on a same-session random split can score well by
recognising the MICROPHONE (bats sit at fixed positions; here Cramer's V between
emitter and channel is 0.589).  Accuracy alone does not distinguish a model that
sometimes guesses a random bat from one that, when it is wrong, guesses the bat
that usually sits at that mic.  This script asks whether that distinction shows
up in the network.

Two synthetic classifiers with the SAME realised accuracy:
    uniform    wrong answers are a random other bat
    structured wrong answers are the most common OTHER bat at that (channel, day)
                -- what a leaky, microphone-reading model would do
Both are applied to the 70,001 calls with a known emitter and addressee, the
directed emitter->addressee network is rebuilt from each, and compared with the
network built from the true labels.

    python network_error_sim.py --dir DIR_WITH_FileInfo.csv_AND_Annotations.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load(d):
    fi = pd.read_csv(os.path.join(d, "FileInfo.csv"),
                     usecols=["FileID", "Recording channel", "Recording time"])
    an = pd.read_csv(os.path.join(d, "Annotations.csv"),
                     usecols=["FileID", "Emitter", "Addressee", "Context"])
    m = an.merge(fi, on="FileID", how="left")
    m["date"] = pd.to_datetime(m["Recording time"], errors="coerce").dt.date
    return m[(m.Emitter != 0) & (m.Addressee != 0) & (m.Addressee != m.Emitter)].reset_index(drop=True)


def network(em, ad):
    return pd.DataFrame({"e": em, "a": ad}).groupby(["e", "a"]).size()


def compare(true_w, true_out, top, pred, addr, strong=100):
    w = network(pred, addr)
    idx = true_w.index.union(w.index)
    t, p = true_w.reindex(idx, fill_value=0), w.reindex(idx, fill_value=0)
    topp = set(w.sort_values(ascending=False).head(len(top)).index)
    out_p = pd.Series(pred).value_counts().reindex(true_out.index, fill_value=0)
    return {
        "top50_jaccard": len(top & topp) / len(top | topp),
        "edge_weight_spearman": float(spearmanr(t.values, p.values).correlation),
        "out_degree_spearman": float(spearmanr(true_out.values, out_p.values).correlation),
        "spurious_edges_ge_strong": int(((t == 0) & (p >= strong)).sum()),
        "spurious_edges_ge_20": int(((t == 0) & (p >= 20)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="network_error_sim.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    b = load(a.dir)
    true = b.Emitter.values
    addr = b.Addressee.values
    emitters = np.sort(np.unique(true))
    key = (b["Recording channel"].astype(str) + "|" + b["date"].astype(str)).values

    rank = {k: list(s.value_counts().index) for k, s in pd.Series(true).groupby(key)}
    struct_wrong = np.empty(len(b), dtype=true.dtype)
    for i, kk in enumerate(key):
        others = [e for e in rank[kk] if e != true[i]]
        struct_wrong[i] = others[0] if others else rng.choice(emitters[emitters != true[i]])
    uni_wrong = np.array([rng.choice(emitters[emitters != t_]) for t_ in true])

    true_w = network(true, addr)
    true_out = pd.Series(true).value_counts()
    top = set(true_w.sort_values(ascending=False).head(50).index)

    out = {"n_calls": int(len(b)), "n_emitters": int(len(emitters)),
           "n_directed_pairs": int(len(true_w)), "results": []}
    for acc in (0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
        hit = rng.uniform(size=len(b)) < acc
        for label, wrong in (("uniform", uni_wrong), ("microphone-structured", struct_wrong)):
            pred = np.where(hit, true, wrong)
            r = compare(true_w, true_out, top, pred, addr)
            r.update({"nominal_accuracy": acc, "realised_accuracy": float(np.mean(pred == true)),
                      "error_structure": label})
            out["results"].append(r)
            print(f"acc {acc:.2f}  {label:22s}  top50 Jaccard {r['top50_jaccard']:.2f}  "
                  f"edge rho {r['edge_weight_spearman']:.3f}  strong spurious {r['spurious_edges_ge_strong']:3d}")
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
