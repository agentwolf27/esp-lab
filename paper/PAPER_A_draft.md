# Paper A — draft skeleton

**Target: ICBINB-BIO @ NeurIPS 2026** — *"I Can't Believe It's Not Better: Failure Modes of AI in
Biology"*, subtitled *"Stress-testing AI for biology in the real world: failure modes, robustness, and
trustworthy scientific discovery."* https://icbinb-bio.github.io/
**Deadline: 29 Aug 2026, 11:59pm AoE (tentative) — 12 days from 17 Aug.**
Notification 29 Sep; workshop 11–12 Dec, Sydney.

Their stated interest, verbatim: *"candid failure analysis, negative results, unexpectedly strong
simple baselines, and benchmarks that support the claims made from them."* Our result is all four.

---

## Working title
**Who, not why: individual and site identity dominate frozen-encoder evaluation of animal vocalisations**

## Abstract (draft, ~180 words)

**[abstract needs one added sentence on the conformal result before submission]**

Frozen audio encoders are now the default feature extractor for animal-vocalisation tasks, and they are
routinely evaluated with random train/test splits. We show this measures the wrong thing. Across four
public corpora — cat meows (21 individuals), dog barks (10), pig calls (6 recording labs) and wild
Egyptian fruit bats (10 emitters) — and six feature sets, we find that individual and site identity are
far more decodable than behavioural context (identity 0.63–0.94 against chance 0.05–0.17), and that
random splits inflate reported context accuracy by 10 to 24 points relative to held-out-group splits.
Within a corpus, the feature set that leaks less identity inflates less (partial r=0.646, p=7e-11),
though this does not hold as a cross-corpus law (r=0.221, n=12, n.s.). Two consequences are concrete:
on pig valence under held-out-lab evaluation, 88 classic acoustic descriptors (eGeMAPS, 0.669)
**outperform** a self-supervised speech encoder (WavLM, 0.660) while leaking less identity; and the
originating study's own feature set scores *below chance* (0.386) once a lab is held out. We further
show the context axis is markedly more robust than identity to channel degradation, and quantify how
much target-group labelling repairs the gap.

## Contributions
1. A four-corpus, six-feature-set audit with a single honest protocol (held-out group, never random).
2. The inflation law, scoped precisely: within-corpus yes, cross-corpus not established.
3. An unexpectedly strong simple baseline: eGeMAPS ≥ WavLM on pigs under honest evaluation.
4. A published feature set falling below chance under lab shift.
5. A channel stress test showing context is 1.4–4.2× more robust than identity.
6. Recovery curves: ~11 target-individual clips saturate cats; 100 target-lab clips close half the pig gap.
7. **Conformal coverage is per-group broken** (marginal 0.90, worst cat 0.41, 5/20 below 0.80),
   the standard Mondrian-by-group remedy is *structurally unavailable* under a group-disjoint split,
   and repair requires ≥9 labelled clips from the deployment group at α=0.10.

## Figures (all exist)
1. `audit_table.png` — the four-corpus table
2. `inflation_law.png` — identity vs inflation, with the null cluster-level caveat
3. `encoders.png` — per-encoder honest/leaky/identity on cats
4. `layers.png` — identity fades with depth, context does not
5. `room.png` — the background-only control (CatMeows room confound)
6. `stress.png` — channel robustness, matched difficulty
7. `recovery.png` — labelling recovery curves
8. `coverage.png` — per-group conformal coverage vs the marginal guarantee

## Section plan
1. **Intro** — frozen encoders + random splits are the field default; we audit that default.
2. **Related** — BEANS/AVEX/BirdSet evaluation conventions; leakage literature in ML; PAM validation practice.
3. **Protocol** — held-out group; the three numbers (honest / leaky / identity); why identity is the right control.
4. **Corpora** — cats, dogs, pigs, bats; group = individual or lab.
5. **Results** — the table; the inflation law with its limit; eGeMAPS>WavLM; below-chance features; the room control.
6. **Robustness** — channel stress; recovery curves.
6b. **It breaks the guarantees too** — conformal marginal vs per-group coverage; why Mondrian-by-group
    is vacuous in deployment; the ≥9-clip rule; species shift collapsing even marginal coverage.
7. **Limitations** — one encoder family dominant; bats have no decodable context; dog context confounded with session; cluster-level null; probes are linear.
8. **Recommendations** — report identity decodability alongside every context number; split by group; publish per-group spread.

## To do before submission
- [ ] Confirm page limit / format / archival status (site did not state them) — check the CFP page again nearer the date
- [ ] Per-group spread numbers into a table (have the data)
- [ ] Related-work pass: does anyone already report individual-level leakage in bioacoustics? (novelty check said no; re-verify)
- [ ] Decide author list / affiliation (independent researcher)
- [ ] arXiv endorsement for cs.SD if we also preprint — start early, it takes days
