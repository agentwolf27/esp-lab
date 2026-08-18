"""Is the Prat bat corpus usable for a session-held-out INDIVIDUAL-ID audit?

The question this answers
-------------------------
Our audit so far asks "what is happening?" and holds out the animal. The other
task -- "who is this?" -- cannot hold out the animal, because the animal IS the
label. The unit you must hold out instead is the SESSION: train on some
recordings of bat 215, test on recordings of bat 215 made on a different day.
Otherwise "individual recognition" can be same-microphone, same-afternoon
recognition, and the metric cannot tell.

That audit needs a corpus where individuals recur across sessions. This script
checks whether Prat et al. (2017) is one, using only two small CSVs -- no audio.

What it needs (35 MB, from the corpus's own figshare collection c.3666502):
    FileInfo.csv     https://ndownloader.figshare.com/files/8900695   31.6 MB
    Annotations.csv  https://ndownloader.figshare.com/files/7379008    3.3 MB

The BEANS packaging everyone uses (10 emitters x 1,000 calls) drops both files,
which is why experiments/context_probe/bats_identity.py had to record "no
session metadata to hold out" as a limitation. It exists; it is just not in the
convenience download.

    python design.py [--dir DIR] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from leakcheck import cramers_v

CAPS = (100, 200, 400)
TOTAL_AUDIO_GB = 97.8          # measured across the 31 figshare audio archives
N_ARCHIVES = 31


def load(d: str) -> pd.DataFrame:
    fi = pd.read_csv(os.path.join(d, "FileInfo.csv"),
                     usecols=["FileID", "Treatment ID", "File name", "File folder",
                              "Recording channel", "Recording time"])
    an = pd.read_csv(os.path.join(d, "Annotations.csv"),
                     usecols=["FileID", "Emitter", "Addressee", "Context"])
    m = an.merge(fi, on="FileID", how="left")
    m["date"] = pd.to_datetime(m["Recording time"], errors="coerce").dt.date
    return m, len(fi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding the two CSVs")
    ap.add_argument("--out", default="design.json")
    a = ap.parse_args()

    m, n_files = load(a.dir)
    # Emitter 0 is "unidentified", not a bat
    k = m[(m.Emitter != 0) & m.date.notna()].reset_index(drop=True)

    per = k.groupby("Emitter").agg(calls=("FileID", "size"), days=("date", "nunique"),
                                   treatments=("Treatment ID", "nunique"),
                                   channels=("Recording channel", "nunique"))
    r = {
        "annotations": int(len(m)), "audio_files_in_corpus": int(n_files),
        "calls_with_known_emitter": int(len(k)),
        "emitters": int(k.Emitter.nunique()),
        "days": int(k.date.nunique()),
        "date_range": [str(k.date.min()), str(k.date.max())],
        "treatments": int(k["Treatment ID"].nunique()),
        "channels": int(k["Recording channel"].nunique()),
        "emitters_with_at_least_n_days": {str(t): int((per.days >= t).sum())
                                          for t in (2, 3, 5, 10)},
        "emitters_with_at_least_n_treatments": {str(t): int((per.treatments >= t).sum())
                                                for t in (2, 3, 5, 10)},
        "median_calls_per_emitter": float(per.calls.median()),
        "median_days_per_emitter": float(per.days.median()),
        # How badly is the LABEL tied to each nuisance axis? A high value means a
        # model asked "which bat?" can answer "which microphone?" and score.
        "cramers_v": {
            "emitter_vs_channel": cramers_v(k.Emitter.values,
                                            k["Recording channel"].astype(str).values),
            "emitter_vs_treatment": cramers_v(k.Emitter.values,
                                              k["Treatment ID"].astype(str).values),
            "emitter_vs_day": cramers_v(k.Emitter.values, k.date.astype(str).values),
        },
        "subsamples": [],
    }

    for cap in CAPS:
        idx = np.concatenate([g.sample(min(len(g), cap), random_state=0).index.values
                              for _, g in k.groupby("Emitter")])
        sub = k.loc[idx]
        d_per = sub.groupby("Emitter")["date"].nunique()
        r["subsamples"].append({
            "cap_per_emitter": cap, "calls": int(len(sub)),
            "emitters": int(sub.Emitter.nunique()),
            "distinct_files": int(sub.FileID.nunique()),
            "approx_audio_gb": round(sub.FileID.nunique() / n_files * TOTAL_AUDIO_GB, 1),
            "median_days_per_emitter": float(d_per.median()),
            "min_days_per_emitter": int(d_per.min()),
            "every_emitter_on_2plus_days": bool((d_per >= 2).all()),
        })

    # The needed files are scattered, so you cannot download only the bytes you
    # want: figshare ships one ~3.2 GB archive per folder.
    folders = k.groupby("File folder").FileID.nunique()
    r["folders_touched"] = int(len(folders))
    r["archives_total"] = N_ARCHIVES
    r["download_gb_to_get_any_subsample"] = round(len(folders) / N_ARCHIVES * TOTAL_AUDIO_GB, 1)

    json.dump(r, open(a.out, "w"), indent=2)
    print(json.dumps(r, indent=2))
    print(f"\nVERDICT: {'USABLE' if r['emitters_with_at_least_n_days']['2'] == r['emitters'] else 'PARTIAL'}"
          f" -- {r['emitters_with_at_least_n_days']['2']}/{r['emitters']} emitters recur across "
          f">=2 days, so a day-held-out split scores every individual.")


if __name__ == "__main__":
    main()
