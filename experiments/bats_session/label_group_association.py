"""Cramer's V between the label and the recording unit, computed the same way
for both corpora so the two numbers quoted in Section 4.3 are comparable.

    pigs  valence  vs recording lab      (from results/out_pigs/meta.json)
    bats  emitter  vs microphone channel (from the Prat FileInfo/Annotations CSVs)

    python label_group_association.py [--bats-dir DIR] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from leakcheck import cramers_v

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def pigs():
    meta = json.load(open(os.path.join(REPO, "results", "out_pigs", "meta.json")))
    y = np.array([str(m["pol"]) for m in meta])
    g = np.array([str(m["team"]) for m in meta])
    return {"n": len(y), "n_labels": int(len(np.unique(y))), "n_groups": int(len(np.unique(g))),
            "cramers_v": cramers_v(y, g)}


def bats(d):
    import pandas as pd
    fi = pd.read_csv(os.path.join(d, "FileInfo.csv"),
                     usecols=["FileID", "Treatment ID", "Recording channel", "Recording time"])
    an = pd.read_csv(os.path.join(d, "Annotations.csv"), usecols=["FileID", "Emitter"])
    m = an.merge(fi, on="FileID", how="left")
    m["date"] = pd.to_datetime(m["Recording time"], errors="coerce").dt.date
    k = m[(m.Emitter != 0) & m.date.notna()]
    y = k.Emitter.astype(str).values
    out = {"n": int(len(k)), "n_labels": int(k.Emitter.nunique())}
    for col, name in [("Recording channel", "channel"), ("Treatment ID", "treatment"),
                      ("date", "day")]:
        out[f"cramers_v_emitter_vs_{name}"] = cramers_v(y, k[col].astype(str).values)
    # what you can guess knowing ONLY the recording set-up
    for cols, name in [(["Recording channel"], "channel"),
                       (["Recording channel", "date"], "channel_and_day")]:
        key = k[cols[0]].astype(str) if len(cols) == 1 else \
            k[cols[0]].astype(str) + "|" + k[cols[1]].astype(str)
        tab = k.assign(_k=key.values).groupby(["_k", "Emitter"]).size().unstack(fill_value=0)
        out[f"guess_emitter_from_{name}"] = float(tab.max(axis=1).sum() / tab.values.sum())
    out["chance"] = 1.0 / out["n_labels"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bats-dir", default=".")
    ap.add_argument("--out", default="label_group_association.json")
    a = ap.parse_args()
    r = {"pigs_valence_vs_lab": pigs(), "bats_emitter_vs_recording_unit": bats(a.bats_dir)}
    json.dump(r, open(a.out, "w"), indent=2)
    print(json.dumps(r, indent=2))
