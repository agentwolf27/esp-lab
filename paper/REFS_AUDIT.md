# Reference audit — `refs.bib`

**Audited:** 17 Aug 2026 · **Paper:** "Who, Not Why" (ICBINB-BIO @ NeurIPS 2026) · **Deadline:** 29 Aug 2026

Method: every entry checked against a publisher record — Crossref REST API (`api.crossref.org/works/<doi>`),
the arXiv API (`export.arxiv.org/api/query`), the ACL Anthology, Europe PMC / PMC full text, and
journal pages. No entry is marked *verified* on the strength of `RELATED_WORK.md` alone; that
document was used to locate candidates, then each was re-checked at source. Three of its claims
turned out to need correction (noted below).

**Headline:** 22 entries in, 23 out (one added). **4 corrected**, **1 replaced outright**,
**17 verified as-is or with completed metadata**, **0 unverifiable**. All original citation keys
preserved — `paper_a.tex` compiles unchanged, no renames.

---

## 1. The four `[VERIFY]` entries

### `lin2026` — CORRECTED (title wrong, authors and venue filled in)

The paper is real but the **title in the draft was wrong**. There is no paper called
"The Identity Trap: Subject Identity Dominates Task Decoding in EEG Foundation Models".

| | |
|---|---|
| Actual title | *The Identity Trap in EEG Foundation Models: A Diagnostic Audit* |
| Authors | Jun-You Lin, Ying Choon Wu, Tzyy-Ping Jung |
| Venue | arXiv only — **unrefereed preprint**, no journal or conference |
| Date | submitted 4 June 2026 (v1); 28 pp., 6 figs, 8 tables |
| ID | arXiv:2606.06647 |
| Source | <https://arxiv.org/abs/2606.06647> · arXiv API record |

Contents match what the paper needs: they name the "Identity Trap", introduce the FMScope protocol,
and test LaBraM, CBraMod and REVE across four datasets. See §5 for a wording caveat about *how*
the paper characterises this result.

### `tang2023` — IDENTIFIED (was a placeholder)

| | |
|---|---|
| Title | *End-to-end transfer learning for speaker-independent cross-language and cross-corpus speech emotion recognition* |
| Authors | Duowei Tang, Peter Kuppens, Lucca Geurts, Toon van Waterschoot |
| Venue | arXiv only — v1 Nov 2023, v3 Dec 2025. **No peer-reviewed version exists** (checked Crossref by title; no match) |
| ID | arXiv:2311.13678 |
| Source | <https://arxiv.org/abs/2311.13678> · full text <https://arxiv.org/html/2311.13678v3> |

**Why this one is the canonical fit.** Several papers report SSL ahead of hand-crafted features in
SER (Wagner et al. 2023 TPAMI; Elbanna et al. 2022 on BYOL-S). Only this one runs SSL against
**eGeMAPS specifically**, under a **speaker-independent cross-language** protocol, and prints both
sides of the comparison — which is exactly the contrast the paper's sentence sets up. Verified
verbatim from the full text:

> "the wav2vec 2.0 improves UA with about 20.9% to 66.4% compared to the eGeMAPS method"

> "in the 'ENCH->DE' experiment, wav2vec 2.0-CL shows a very subtle degradation (i.e. about 0.6%
> and 0.3% degradation in UA and WA, respectively, compared to the wav2vec 2.0-WL), whereas the
> eGeMAPS-CL shows a 17.8% and 18.5% decrease in UA and WA, respectively."

**Caveat to weigh:** it is an unrefereed preprint, and the paper leans on it as the counter-evidence
that makes Claim 4 look honest. A reviewer may discount it. `wagner2023` (IEEE TPAMI 45(9)) is
added to the bib as a refereed companion.

### `ghani2026` — VERIFIED (order, initials, volume and DOI all correct)

| | |
|---|---|
| Authors | Burooj **Ghani**, Anne Leonie **Baier**, Vincent J. **Kalkman**, Dan **Stowell** — order as drafted |
| Venue | PLOS Computational Biology **22(8): e1014604** |
| Published | 12 August 2026 (five days before this audit) |
| DOI | 10.1371/journal.pcbi.1014604 |
| Source | <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014604> · Crossref |

Supporting quote confirmed on the article page: *"Avoid purely random dataset splits, as they are
prone to site- or time-based leakage."*

### `han2022` — REPLACED (the draft cited the wrong Han et al. paper)

The draft entry was *"Exploring Longitudinal Cough, Breath, and Voice Data for COVID-19 Progression
Prediction via Sequential Deep Learning"* (Interspeech 2022). That is a real Han et al. paper, but
it is about **progression prediction**, not split protocol — it does not contain the
random-vs-participant comparison the paper cites it for. The note *"see also their analysis of
identity-confounded splits"* was papering over the mismatch.

Correct paper:

| | |
|---|---|
| Title | *Sounds of COVID-19: exploring realistic performance of audio-based digital testing* |
| Authors | Jing Han, Tong Xia, Dimitris Spathis, Erika Bondareva, Chloë Brown, Jagmohan Chauhan, Ting Dang, Andreas Grammenos, Apinan Hasthanasombat, Andres Floto, Pietro Cicuta, Cecilia Mascolo |
| Venue | npj Digital Medicine **5(1): 16** (2022) |
| DOI | 10.1038/s41746-021-00553-x |
| Source | <https://pmc.ncbi.nlm.nih.gov/articles/PMC8799654/> · Crossref |

Verified verbatim: *"random-splits yielded a higher accuracy than user-splits, with the performance
gains coming from the overlapping participants whose data have been seen from training."*
Participant-independent AUC **0.71** (95% CI 0.65–0.77), sensitivity 0.65, specificity 0.69; the
random split gives sensitivity **0.84** / specificity **0.78**. This supports the Related Work
sentence "measured for audio by \citet{han2022}" exactly.

---

## 2. Every other entry

| Key | Status | Finding | Source |
|---|---|---|---|
| `chaibubneto2019` | **corrected** (metadata) | Correct paper. `and others` hid 7 of 10 authors — now complete (…Tummalacherla, Snyder, Bot, Trister, Friend, Mangravite, Omberg). npj Digit. Med. 2(1):99, 2019. | Crossref `10.1038/s41746-019-0178-x` |
| `saeb2017` | **verified** | Correct in every field. Added article number `gix019` and DOI. GigaScience 6(5). | Crossref `10.1093/gigascience/gix019` |
| `degrave2021` | **verified** | Exact: DeGrave, Janizek, Lee; Nat. Mach. Intell. 3(7):610–619, 2021. | Crossref `10.1038/s42256-021-00338-7` |
| `geirhos2020` | **corrected** (metadata) | Correct paper; `and others` hid Zemel, Brendel, Bethge, Wichmann — now complete. Nat. Mach. Intell. 2(11):665–673. | Crossref `10.1038/s42256-020-00257-z` |
| `kapoor2023` | **verified** | Patterns 4(9):100804, 2023. Added the article number. | Crossref `10.1016/j.patter.2023.100804` |
| `colonna2016` | **corrected** | Two errors. (a) **João Gama was missing** — he is the second author, hidden behind `and others`. (b) Entry type/venue wrong: it is not a journal article but a Springer LNCS chapter, CAEPIA 2016, **LNCS 9868, pp. 37–47**. | Crossref `10.1007/978-3-319-44636-3_4` |
| `stowell2019` | **verified** | Stowell, Petrusková, Šálek, Linhart; J. R. Soc. Interface 16(153):20180940. Added article number + DOI; accented names converted to LaTeX escapes. | Crossref `10.1098/rsif.2018.0940` |
| `hagiwara2023beans` | **corrected** (metadata) | Correct paper; `and others` hid Cusimano, Effenberger, Zacarian. ICASSP 2023, pp. 1–5. | Crossref `10.1109/ICASSP49357.2023.10096686` |
| `hagiwara2023aves` | **verified** | Single-author (Masato Hagiwara), ICASSP 2023, pp. 1–5. Correct as drafted. | Crossref `10.1109/ICASSP49357.2023.10095642` |
| `miron2026avex` | **verified** + completed | **ICLR 2026 acceptance confirmed** from the arXiv comment field: "In The Fourteenth International Conference on Learning Representations 2026". Seventeen authors; `and others` hid fourteen. arXiv:2508.11845. | arXiv API, id 2508.11845 |
| `schwinger2026` | **corrected** | Author placeholder `Schwinger, [FIRST] and others` resolved: Raphael Schwinger, Ben McEwen, Vincent S. Kather, René Heinrich, Lukas Rauch, Sven Tomforde. Venue was recorded as a bare arXiv ID — it is **accepted at ICAART 2026**. arXiv:2511.08261. | arXiv API, id 2511.08261 |
| `abzaliev2024` | **corrected** — *wrong person* | **The second author was wrong.** Draft had "Espinosa-Anke, Luis" (Cardiff University, NLP). The actual author is **Humberto Pérez Espinosa** (INAOE, Mexico), whose dog corpus the paper uses. Also: entry was `@article` for a conference paper; added pages **16480–16486**, publisher ELRA and ICCL, Torino. | <https://aclanthology.org/2024.lrec-main.1432/> · arXiv 2404.18739 |
| `barber2021` | **verified** | Foygel Barber, Candès, Ramdas, Tibshirani; Inf. Inference 10(2):455–482. Crossref stamps the issue date 2020 (online-first); the printed issue is June 2021, so the year is right. Journal name expanded to "…: A Journal of the IMA". | Crossref `10.1093/imaiai/iaaa017` |
| `ding2023` | **verified** | Five authors correct. NeurIPS 2023 (Adv. NeurIPS 36); arXiv:2306.09335. NeurIPS proceedings are unpaginated, so no page range is given — see §3. | arXiv API · <https://proceedings.neurips.cc/paper_files/paper/2023/file/cb931eddd563f8d473c355518ce8601c-Paper-Conference.pdf> |
| `molnar2008` | **corrected** (metadata) | Correct paper; `and others` hid Pachet, Pongrácz, Dóka, Miklósi. Anim. Cogn. 11(3):389–400. Note Kaplan is **Frédéric** and Pachet is **François**. | Crossref `10.1007/s10071-007-0129-9` |
| `prat2017` | **verified** | Prat, Taub, Pratt, Yovel — four authors, exactly as drafted (Ester **Pratt** is distinct from Yosef **Prat**; not a typo). Sci. Data 4:170143. | Crossref `10.1038/sdata.2017.143` |
| `ludovico2021` | **corrected** | Correct paper and correct author order (Cannas before Battini). Was `@article` with `and others`; now `@inproceedings`, MMM 2021, **LNCS 12573, pp. 230–243**, Springer. **But it is not the source of the quotes** — see §4. | Crossref `10.1007/978-3-030-67835-7_20` |
| `briefer2022` | **corrected** (metadata) | Correct paper; `and others` hid 13 of 16 authors. Sci. Rep. 12(1):3409. (An author correction exists: `10.1038/s41598-023-45242-9` — not cited, but be aware of it.) | Crossref `10.1038/s41598-022-07174-8` |
| `ntalampiras2019` | **ADDED** | New key. Required to attribute the two verbatim quotes in §4.3 of the paper. Animals 9(8):543, 2019. Eight authors. | Crossref `10.3390/ani9080543` · <https://pmc.ncbi.nlm.nih.gov/articles/PMC6719916/> |

### Cross-cutting fix: `note` fields were leaking into the bibliography

`\bibliographystyle{plainnat}` **prints `note` fields**. The previous file carried five editorial
notes — including `note={VERIFY full citation from paper/RELATED\_WORK.md before submission}` on
`lin2026` and `note={VERIFY; reports SSL ahead of eGeMAPS -- cited as the counter-direction}` on
`tang2023`. Both would have been typeset into the reference list of a double-blind submission.
All editorial annotation is now in `%` comments; `note` is used only for real venue information
(arXiv IDs on preprints).

Also normalised: all non-ASCII characters converted to LaTeX escapes, and acronyms/proper nouns
brace-protected (plainnat lowercases titles).

---

## 3. The Ding et al. 2023 claim — CONFIRMED EXACTLY

**Verdict: the paper's reading is correct and may be cited as a theorem.**

- **Location:** Section **1.2, "Preliminaries"** — not §1 generally, as `RELATED_WORK.md` states.
  Worth getting right if the paper cites the section.
- **Verbatim** (from the ar5iv rendering of arXiv:2306.09335v2, cross-checked against the NeurIPS
  camera-ready):

  > "Importantly, for any class $y$ for which $|\mathcal{I}^y| < (1/\alpha)-1$, we will have
  > $\hat{q}^y = \infty$, hence any prediction set generated by classwise will include $y$, no
  > matter the values of the conformal scores."

- **The arithmetic checks out.** The classwise quantile is
  $\hat q^y = \mathrm{Quantile}\big(\lceil(|\mathcal{I}^y|+1)(1-\alpha)\rceil / |\mathcal{I}^y|,\ \{s_i\}\big)$,
  which is finite only when $\lceil (n+1)(1-\alpha)\rceil \le n$. At $\alpha=0.10$:
  $n=8 \Rightarrow \lceil 8.1 \rceil = 9 > 8$ (infinite); $n=9 \Rightarrow \lceil 9.0 \rceil = 9 \le 9$
  (finite). And $(1/\alpha)-1 = 9$. So **"at least $(1/\alpha)-1 = 9$ labelled clips" is exactly
  right**, and the sentence in §6 of the paper can stand as written.
- **Mondrian vacuity is indeed a corollary**: under a group-disjoint split $n_g = 0 < 9$, so
  $\hat q = \infty$ and the set is all of $\mathcal{Y}$. The paper already says this.
- **Bonus, worth adopting:** Ding et al. define **CovGap** ($100 \times \mathrm{mean}_y |\hat c_y - (1-\alpha)|$,
  §3.2) and a fraction-undercovered metric (Appendix C.2). The paper's "5 of 20 cats below 0.80" is
  their FracUnderCov. Using their names makes the result searchable.
- **Do not cite Angelopoulos & Bates for this bound** — their tutorial states the guarantee holds
  for any $n$ and treats calibration size as a variance question, which is the opposite claim.

Source: <https://ar5iv.labs.arxiv.org/html/2306.09335> · <https://arxiv.org/abs/2306.09335>

---

## 4. The two CatMeows quotes — CONFIRMED VERBATIM, but attributed to the wrong paper

**Both quotes are real and exact. Neither is verified in the paper the draft would point to.**

They appear in **Ntalampiras et al. (2019), *Animals* 9(8):543** — the *classification* paper — not
in Ludovico et al. (2021), the *dataset* paper.

| Quote | Exact wording | Location |
|---|---|---|
| 1 | "Cats were transported by their owner, adopting the normal routine used to transport them for any other reason, to an unfamiliar environment (e.g., **a room in a different apartment or an office**, not far from their home environment)… where they stayed alone for a maximum of 5 min." | §2.1 **"Treatments"** |
| 2 | "Moreover, the characteristics of the room (e.g., topological properties, reverberations, presence of furniture, etc.) **should not affect the captured audio signals**. Data acquisition could not be conducted in a controlled anechoic environment…" | §2.2 **"Data Acquisition"** |

Source: Europe PMC full text <https://pmc.ncbi.nlm.nih.gov/articles/PMC6719916/> · MDPI
<https://www.mdpi.com/2076-2615/9/8/543>

Corroboration for quote 1: the Zenodo record itself (**10.5281/zenodo.4008297**, CC-BY-4.0) carries
the same parenthetical in its description — *"Cats were transferred by their owners into an
unfamiliar environment (e.g., a room in a different apartment or an office)"* — and names
Ntalampiras et al. (2019) as the record's **reference publication**.
Source: <https://zenodo.org/records/4008297>

**Action required in `paper_a.tex`** (flagged, not made — I did not edit the .tex):

- §4.3 currently attributes both quotes to "the corpus documentation" with **no citation at all**.
  Add `\citep{ntalampiras2019}` to that sentence. The key now exists in `refs.bib`.
- Do **not** attribute them to `ludovico2021`. I could not extract the text of that Springer
  chapter (the open-access PDF at air.unimi.it uses subsetted CFF fonts and yields no recoverable
  text), so I cannot confirm either phrase appears there. Attributing an unverified quote to it
  would be precisely the error this paper is about.
- `ludovico2021` and `briefer2022` are currently in `refs.bib` but **never `\cite`d** in
  `paper_a.tex`. Both should be cited — CatMeows and the Soundwel pig corpus are two of the four
  corpora, and §4.2 discusses Briefer et al.'s published feature set at length.

---

## 5. The Abzaliev et al. 2024 numbers — CONFIRMED

The paper's Related Work sentence reads: *"\citet{abzaliev2024} report dog individual-ID at 0.50
against 0.05 chance alongside context classification at 0.62 against a 0.56 baseline, without
commenting on the disparity."* All four numbers check out.

| Task | Reported (wav2vec2, pre-trained) | Baseline | Paper's figure |
|---|---|---|---|
| Dog recognition | **49.95 %** (Table 2) | **5.03 %** majority | 0.50 vs 0.05 ✓ |
| Context grounding | **62.18 %** (Table 4) | **56.37 %** majority | 0.62 vs 0.56 ✓ |

Corpus: **74 dogs**, recorded in Tepic and Puebla, Mexico (the Pérez Espinosa corpus — *not* the
Molnár corpus). And the "without commenting on the disparity" part holds: the authors discuss
pre-training gains per task and note gender ID is hardest, but nowhere compare identity
decodability against context decodability. **The sentence can stand.**

Source: <https://arxiv.org/html/2404.18739v1> · <https://aclanthology.org/2024.lrec-main.1432/>

> **One wording nit, and it matters in a paper about rigour.** 5.03 % is the **majority-class
> baseline**, not chance. With 74 dogs, uniform chance is 1/74 ≈ 1.4 %. Writing "0.05 chance"
> is loose. Suggest: *"…dog individual-ID at 0.50 against a 0.05 majority baseline alongside
> context classification at 0.62 against a 0.56 majority baseline…"* — it costs three words and
> removes the only line a picky reviewer could call sloppy in that paragraph.

---

## 6. Missing must-cites

Added to `refs.bib` (verified, uncited — BibTeX ignores them until `\cite`d).

### 6a. Conformal prediction foundations — currently **absent entirely**

The paper has a full conformal section citing only `barber2021` and `ding2023`. That is a
conspicuous gap for anyone who works in the area.

| Key | Reference | Why |
|---|---|---|
| `vovk2013` | Vovk (2013), *Conditional validity of inductive conformal predictors*, Mach. Learn. 92(2–3):349–376, `10.1007/s10994-013-5355-6` | Origin of label-conditional / Mondrian validity. **Ding et al. cite exactly this** for the classwise guarantee, immediately before the sentence the paper quotes. Citing the bound without the source of the procedure skips a link. (Journal version of ACML 2012, PMLR 25:475–490.) |
| `angelopoulos2023` | Angelopoulos & Bates (2023), *Conformal Prediction: A Gentle Introduction*, FnTML 16(4):494–591, `10.1561/2200000101` | Standard citation for split-conformal machinery. **Cite for method only** — not for the ≥9 bound. |
| `vovk2022book` | Vovk, Gammerman & Shafer (2022), *Algorithmic Learning in a Random World*, 2nd ed., Springer, `10.1007/978-3-031-06649-8` | The canonical monograph; one citation signals fluency. |

### 6b. Speaker-independent SER canon — currently **absent entirely**

The paper makes a speaker/individual-disjoint-split argument in an audio venue and cites no SER
methodology at all. A reviewer from speech will notice immediately.

| Key | Reference | Why |
|---|---|---|
| `schuller2009` | Schuller, Steidl & Batliner (2009), *The INTERSPEECH 2009 Emotion Challenge*, pp. 312–315, `10.21437/Interspeech.2009-103` | The moment SER standardised on speaker-independent partitioning — the precedent for "split by individual" in audio. |
| `schuller2010` | Schuller et al. (2010), *Cross-Corpus Acoustic Emotion Recognition*, IEEE TAC 1(2):119–131, `10.1109/T-AFFC.2010.8` | States that the field overestimates results by using data with "less common true speaker disjunctive partitioning" — this paper's thesis, for speech, in 2010. Strengthens the honest framing ("we are not first to warn; we measure"). |
| `wagner2023` | Wagner et al. (2023), *Dawn of the Transformer Era in SER*, IEEE TPAMI 45(9):10745–10759, `10.1109/TPAMI.2023.3263585` | Refereed anchor beside the arXiv-only `tang2023`, **and** the closest existing statement of Claim 1 in speech: transformer SER models are fair w.r.t. gender groups but not w.r.t. individual speakers. |

### 6c. Device / recorder / site shift in bioacoustics — currently **absent entirely**

The paper's site-and-lab axis is a central claim and nothing in the bibliography establishes that
recording hardware and site are known confounds in this field.

| Key | Reference | Why |
|---|---|---|
| `turgeon2017` | Turgeon, Van Wilgenburg & Drake (2017), *Microphone variability and degradation*, Avian Conserv. Ecol. 12(1):9, `10.5751/ACE-00958-120109` | The hardware axis, measured: nominally identical ARUs differ in sensitivity and degrade in the field. This is the physical mechanism behind the site/lab confound — currently asserted without support. |
| `lostanlen2019` | Lostanlen et al. (2019), *Robust sound event detection in bioacoustic sensor networks*, PLOS ONE 14(10):e0214168, `10.1371/journal.pone.0214168` | Detectors trained on a limited set of sensors fail to generalise to new ones — the PAM-facing precedent for the site claim. |
| `rauch2025birdset` | Rauch et al. (2025), *BirdSet*, ICLR 2025 (spotlight), arXiv:2403.10380 | **Defensive.** The focal→soundscape covariate shift for birds is *already* benchmarked with geographically distinct test soundscapes. Cite it, or a reviewer will, and scope the site claim to non-avian / behavioural-context data. |

### 6d. Optional but recommended

| Key | Reference | Why |
|---|---|---|
| `zare2026` | Zare (2026), arXiv:2607.24519 | Classical features beat EEG foundation models; **all five encoders decode dataset identity at 1.000**. With `lin2026` this lets the paper argue a general property of frozen-encoder benchmarks rather than a bioacoustics quirk. Single-author, unrefereed — cite as convergent evidence only. Verified real via the arXiv API. |

**Not added, but flagged from `RELATED_WORK.md` as genuine rejection risks if omitted** (I did not
independently verify these, so they are recommendations to check, not verified entries):
Arnaud et al. 2023 (PLOS Comput. Biol. 19(4):e1010325 — defines leakage in bioacoustics);
Lefèvre, Sypherd & Briefer 2025 (iScience 28(2):111834 — the Briefer group already groups by
individual, which is the paper's opening); Müller et al. 2021 (ASVspoof silence shortcut — the
speech twin of the background-only control); Mouterde et al. 2014 (PLOS ONE 9(7):e102842 — the
direct bioacoustic counter-example to Claim 7).

---

## 7. Claims in the paper the sources do not fully support

Four items. None is fatal; the first two are small rewrites.

1. **`lin2026` is characterised too strongly.** The paper says Lin et al. "document an 'identity
   trap' in EEG where **subject identity is decodable far above the task label**." What the paper
   actually measures is a **variance decomposition**: "frozen subject-variance is 13–89× a random
   null in 12/12 pairs". That is dominance of subject *variance*, not a head-to-head decodability
   comparison between subject and task. Suggested rewrite: *"…where subject-identity variance in
   frozen representations exceeds a random null by 13–89× across all twelve model×dataset pairs
   tested."* Accurate, and the concrete number is more persuasive.

2. **`lin2026` is co-cited for the wrong axis in the Introduction.** `\citep{degrave2021,lin2026}`
   supports "a model that reads **the acquisition site** rather than the biology". DeGrave et al.
   fits (hospital source); Lin et al. is about *subject* identity and explicitly does not cover
   recording site. `zare2026` — dataset identity decoded at 1.000 — is the citation that actually
   carries "site/acquisition". Either swap it in or move `lin2026` to the identity clause.

3. **"0.05 chance" for Abzaliev should read "0.05 majority baseline."** See §5. Uniform chance over
   74 dogs is ≈1.4 %.

4. **"We found no prior application of conformal prediction to bioacoustics or passive acoustic
   monitoring."** `RELATED_WORK.md` reports an exhaustive arXiv/OpenAlex sweep supporting this, but
   its own caveat is that non-indexed ecology journals were covered by web search only. I did not
   re-run that sweep. An absolute negative is the single easiest sentence for a reviewer to falsify
   — hedge it to **"to our knowledge"**. Note also that adjacent work exists (conformal taxonomic
   validation on citizen-science images; conformal prediction for species distribution models), so
   the claim is safe only when scoped tightly to bioacoustics/PAM audio.

Verified as **supported**, for the record:
`schwinger2026` "calibration varies sharply across datasets" — the abstract says "Model calibration
varies significantly across datasets and classes" ✓.
`hagiwara2023beans` "random or pre-specified splits" — BEANS uses 6:2:2 stratified random ✓.
`ghani2026` field guidance against random splits — verbatim quote confirmed ✓.
`ding2023` as a theorem — exact ✓ (§3).
`abzaliev2024` numbers — exact ✓ (§5).
Both CatMeows quotes — exact ✓ (§4).

---

## 8. Could not verify

**Nothing in `refs.bib` is unverified.** Every one of the 23 entries resolved to a publisher record.

Two limitations worth stating plainly:

- **The Ludovico et al. (2021) chapter text.** I confirmed its bibliographic record via Crossref
  but could not read its body (PDF text is not extractable). This is why the quotes are attributed
  to `ntalampiras2019`, where they are verified, rather than to the dataset paper.
- **NeurIPS page numbers for `ding2023`.** The NeurIPS proceedings are unpaginated; no page range
  is asserted rather than inventing one. The ACM DL mirror lists it under `10.5555/3666122.3668939`
  if a page range is ever required.

Three `RELATED_WORK.md` claims were corrected during this audit, which is worth knowing if that
document is reused: the Ding quote is in **§1.2**, not §1; the Abzaliev author is **Humberto Pérez
Espinosa**, not the Luis Espinosa-Anke that reached `refs.bib`; and the CatMeows quotes are from
**Ntalampiras et al. 2019**, not the Ludovico et al. 2021 dataset paper it recommends citing.
