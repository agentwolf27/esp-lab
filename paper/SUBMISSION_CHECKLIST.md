# Submission checklist — ICBINB-BIO @ NeurIPS 2026

**Deadline 29 Aug 2026, 11:59 pm AoE.** 8 pages excl. references/appendices · double-blind ·
non-archival · OpenReview · concurrent submission allowed · LLM-use disclosure required.

## Blockers (must be done)
- [ ] **Anonymised mirror.** Run `bash paper/anonymise.sh` → `../esp-lab-anon/` (verified clean).
      Upload to anonymous.4open.science, then replace `[ANONYMISED MIRROR URL]` in `paper_a.tex`.
      *Anonymity extends to linked material — the public repo cannot be cited.*
- [ ] **Verify 3 references** marked `[VERIFY]` in `refs.bib` (Lin 2026 EEG, Tang 2023, Ghani 2026
      author list) against `paper/RELATED_WORK.md`.
- [ ] **LLM-use disclosure** — the venue requires it. Be specific and honest: LLM assistance was used
      for experiment implementation, analysis scripting, literature search and drafting; all numbers
      were produced by committed code and verified against the JSON outputs.
- [ ] **Compile.** No TeX locally (and TeX Live would use most of the free disk). Upload
      `paper_a.tex` + `template/neurips_2026.sty` + `figures/` to Overleaf.
- [ ] **Check page count** ≤ 8 after figures go in; cut §5 or §7 first if over.

## Should do
- [ ] Insert figures: `audit_table.png`, `inflation_law.png`, `coverage.png`, `recovery.png`,
      `stress.png`, `room.png`, `layers.png` (7 available; pick 4–5 that fit).
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
