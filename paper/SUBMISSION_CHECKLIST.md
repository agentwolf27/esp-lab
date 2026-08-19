# Submission checklist — ICBINB-BIO @ NeurIPS 2026

**Deadline 29 Aug 2026, 11:59 pm AoE.** 8 pages excl. references/appendices · double-blind ·
non-archival · OpenReview · concurrent submission allowed · LLM-use disclosure required.

## Blockers (must be done)
- [ ] **Anonymised mirror.** Run `bash paper/anonymise.sh` → `../esp-lab-anon/` (verified clean).
      Upload to anonymous.4open.science, then replace `[ANONYMISED MIRROR URL]` in `paper_a.tex`.
      *Anonymity extends to linked material — the public repo cannot be cited.*
- [ ] **Verify 3 references** (Lin 2026 EEG, Tang 2023, Ghani 2026 author list). The `[VERIFY]`
      markers are no longer in `refs.bib`; confirm against `paper/REFS_AUDIT.md` whether that means
      they were checked or the markers were dropped. `chauhan2025` and `hagedoorn2025` (added
      18 Aug) were verified against Europe PMC.
- [x] **LLM-use disclosure** — written into `paper_a.tex` (§ Reproducibility and LLM use), including
      the adversarial use and the below-chance error it caught. Read it once and confirm you are
      comfortable signing it.
- [ ] **Compile.** No TeX locally (and TeX Live would use most of the free disk). Upload
      `paper_a.tex` + `template/neurips_2026.sty` + `figures/` to Overleaf.
- [ ] **Check page count** ≤ 8 after figures go in; cut §5 or §7 first if over.

## Should do
- [x] Insert figures — 4 in: `inflation_law` (§4.2), `stress` (§5), `coverage_paper` (§6),
      `recovery` (§7). `coverage.png` was a 5×4 exploration grid, unreadable at this width;
      `experiments/conformal_eval/paper_figure.py` regenerates the one panel the claim rests on as
      `figures/coverage_paper.png`. **Check the page count after compiling** — drop `stress` first
      if over, its section is three sentences.
- [ ] A domain reader — someone who knows bioacoustics — on §4.4 and §8.
- [ ] Decide author list / affiliation (independent researcher is fine and honest).
- [ ] arXiv endorsement for cs.SD **if** preprinting — takes days for an independent researcher, and
      the venue is non-archival so a preprint is allowed and probably wise.

## Already done
- [x] All experiments; every number traced to `results/*.json`
- [x] Seven reviewer-corrections from the novelty check applied
- [x] Related work written for this committee (molecular/genomics/clinical, no bioacousticians)
- [x] `refs.bib` with 22 entries
- [x] `paper_a.tex` in the workshop template with `dblblindworkshop`
