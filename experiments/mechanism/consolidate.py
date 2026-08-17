"""Merge task12 / task3 / followup into out_sae/results.json with a verdict block."""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

t12 = json.load(open(os.path.join(OUT, "task12.json")))
t3 = json.load(open(os.path.join(OUT, "task3.json")))
fu = json.load(open(os.path.join(OUT, "followup.json")))

res = {
    "question": ("What in the frozen-WavLM embedding carries the cross-species affect "
                 "transfer (cat / dog / pig)?"),
    "data": {
        "encoder": "microsoft/wavlm-base-plus, frozen, mean-pooled hidden states (cached)",
        "cat": "348 clips, 20 individuals (isolation=neg vs brushing=pos)",
        "dog": "308 clips, 10 individuals (aggression=neg vs play=pos)",
        "pig": "5031 calls, 6 recording teams (Soundwel; Neg vs Pos valence contexts)",
        "layers": {"primary": 9, "robustness": 12, "early_contrast": 3},
        "z_scoring": "within species; pigs within recording team",
        "permutation_unit": "within individual (cat/dog), within team (pig); 200 draws",
        "acoustics": ("log_duration, log_rms_energy, spectral_centroid, ZCR, recomputed "
                      "from audio with the same 16 kHz / 6 s preprocessing used for the "
                      "embeddings; pig log-duration vs the Soundwel key's own Dur column "
                      "r = 0.992"),
    },
    "task1_shared_direction": t12["task1"],
    "task2_transfer_projection": t12["task2"],
    "task3_sparse_components": t3,
    "task3_followup": fu,
}

L = "9"
b = t12["task1"][L]
res["verdict"] = {
    "one_line": ("No. No sparse component is affect-selective in all three species; the "
                 "only cross-species component is cat+dog (and it is NOT duration). The "
                 "pig affect axis is essentially the duration axis and is orthogonal to "
                 "the cat and dog affect axes."),
    "shared_direction": (
        "At layer 9 the pairwise cosines between the mean-difference affect directions are "
        f"cat-dog {b['raw']['cat-dog']['cos']:+.3f} (null {b['raw']['cat-dog']['null_mean']:+.3f}"
        f"+-{b['raw']['cat-dog']['null_sd']:.3f}, p={b['raw']['cat-dog']['p_greater']:.3f}), "
        f"cat-pig {b['raw']['cat-pig']['cos']:+.3f} (p={b['raw']['cat-pig']['p_greater']:.3f}), "
        f"dog-pig {b['raw']['dog-pig']['cos']:+.3f} (p={b['raw']['dog-pig']['p_greater']:.3f}). "
        "Nothing clears the permutation null at 0.05 at any of the three layers after "
        "accounting for the 18 comparisons run."),
    "power_caveat": (
        "The cat/dog directions are only moderately estimable (split-half reliability "
        f"cat {b['split_half_reliability']['cat']:.2f}, dog {b['split_half_reliability']['dog']:.2f}), "
        "so the cat-dog cosine is attenuated; corrected it is "
        f"{b['cos_disattenuated']['cat-dog']:+.3f}. The pig direction is highly reliable "
        f"({b['split_half_reliability']['pig']:.2f}), so its near-zero cosine with cat and dog "
        "is a real absence, not a power problem."),
    "duration": (
        f"cos(affect direction, duration direction) at layer 9: cat {b['cos_w_dur']['cat']:+.3f}, "
        f"dog {b['cos_w_dur']['dog']:+.3f}, pig {b['cos_w_dur']['pig']:+.3f}. The pig affect axis "
        "is very nearly the duration axis (Spearman between the pig affect score and log-duration "
        f"= {b['own_axis']['pig']['spearman_score_logdur']:+.3f}). The duration directions "
        f"themselves are shared between cat and dog ({b['cos_dur_dur']['cat-dog']:+.3f}) but not "
        f"with pigs (cat-pig {b['cos_dur_dur']['cat-pig']:+.3f}, dog-pig {b['cos_dur_dur']['dog-pig']:+.3f})."),
    "sparse_components": (
        f"{t3['n_components']} sparse atoms on {t3['n_pooled']} pooled vectors: "
        f"{t3['n_pass_2species']} selective in >=2 species with the same sign "
        f"(null {t3['null_pass_2species']['mean']:.2f}+-{t3['null_pass_2species']['sd']:.2f}, "
        f"p={t3['null_pass_2species']['p']:.3f}), {t3['n_pass_3species']} in all three. "
        "Component 33 (cat AUC 0.299, dog 0.334, pig 0.522) is not duration in disguise: "
        "controlling for log-duration leaves it unchanged (cat 0.340, dog 0.294) and it "
        "survives being computed inside each animal (cat 0.372, dog 0.351, all 10 dogs on "
        "the same side)."),
    "answer_to_task4": (
        "NO component is affect-selective in all three species. The clean negative stands: "
        "there is no single sparse feature in frozen WavLM layer 9 that codes affect for cat, "
        "dog and pig alike. One genuine cat+dog component exists and it is duration-independent."),
}

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "results.json"),
      f"({os.path.getsize(os.path.join(OUT, 'results.json'))/1024:.0f} KB)")
