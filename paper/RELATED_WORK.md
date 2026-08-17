# Related work and novelty audit — Paper A

**Target:** ICBINB-BIO @ NeurIPS 2026 · **Audit date:** 17 Aug 2026 · **Deadline:** 29 Aug 2026

Scope: adversarial novelty check on claims 1–8, plus submission logistics. Method: ~200 web
searches across bioacoustics, speech emotion recognition, medical/health audio, ecology,
conformal prediction and the ML-evaluation-leakage literature, plus OpenAlex/Europe PMC
metadata verification of every reference below.

---

## 0. Executive verdict

**Nothing kills the paper. Five things force a reframe, three of them mandatory.**

The reframe in one line: **stop claiming the field is unaware, and start claiming the field has
never measured it.** The normative advice ("split by individual/site") is published — in 2016, in
2019, in 2023, and again in a 2026 PLOS Comp Biol best-practices paper. The *magnitude*, the
*site/lab axis*, the background-only probe applied to **context** rather than identity, the
CatMeows room-by-construction defect, and per-**individual** conformal coverage are not.

Ranked by danger:

1. **The ≥9-clip result is textbook arithmetic (MANDATORY FIX).** It is exactly (1/α)−1 at α=0.10,
   stated verbatim in Ding et al. (NeurIPS 2023, §1): *"for any class y for which |ℐ^y| < (1/α)−1,
   we will have q̂^y = ∞."* I verified this in the source text. Presenting it as an empirical
   finding is the single clearest way to lose a statistically literate reviewer. It also makes the
   Mondrian-vacuity claim a one-line corollary rather than a finding.
2. **The background-only control has bioacoustic ancestry (MANDATORY FIX).** Stowell, Petrusková,
   Šálek & Linhart (J. R. Soc. Interface 2019) already used normally-discarded silent segments to
   expose environmental confounding — via adversarial background mixing, for *individual identity*.
   Your version (background-only probe of *behavioural context*) is genuinely different, but you
   must cite theirs in Methods. Their motivating example is a classifier using a river rather than
   the animal.
3. **Claim 2 is a transfer result, not a discovery (MANDATORY FIX).** Chaibub Neto et al. (2019)
   coined **"identity confounding"**; Han et al. (2022) measured the random-vs-participant split gap
   in *audio* with an effect size inside your 0.10–0.24 band; Ghani et al. (2026) publish the
   recommendation this year in PLOS Comput Biol.
4. **Two 2026 cross-domain near-twins.** Lin, Wu & Jung (arXiv:2606.06647, June 2026) — "The
   Identity Trap in EEG Foundation Models"; and Zare (arXiv:2607.24519, July 2026) — classical
   features beat EEG foundation models while all five encoders decode dataset identity at 1.000.
   Both are EEG, both are recent, neither is a scoop — **but only if you cite them as convergent
   evidence.** Doing so actually strengthens the paper: it lets you argue this is a general property
   of frozen foundation-model benchmarks.
5. **Two statistical soft spots** a reviewer will find: eGeMAPS 0.669 vs WavLM 0.660 is a
   0.9-point gap with no paired test; and r=0.221 at n=12 has ~15% power, so you cannot say the
   cross-corpus relationship is absent, only undetected.

**What is cleanly novel:** the identity-vs-context *contrast* under a matched probe; the +0.10–0.24
magnitude on these four corpora; the site/lab axis; the CatMeows room-by-construction defect; the
below-chance held-out-lab result on the Soundwel feature set; the channel-robustness ratio; and
per-**individual** conformal coverage (every prior demonstration conditions on class label or an
attribute-defined subgroup — **zero** condition on individual entity identity).

Strategic note: **Su-In Lee is an ICBINB-BIO organizer** and is senior author of *"AI for
radiographic COVID-19 detection selects shortcuts over signal"* (Nature Machine Intelligence 2021).
Framing this paper as the bioacoustic instance of that argument is directly aligned with the
organizing committee. Cite DeGrave, Janizek & Lee (2021) prominently.

Note on audience: the organizing committee is **entirely molecular / single-cell /
genomics / clinical-imaging** (Brbić, Koo, Dumitrascu, Lee, Panigrahi, Nagai, Beker, Gadgil). There
is no bioacoustician among them. Write the intro for a computational biologist who has never heard
of BEANS, and lean on the cross-domain analogues (EEG, chest X-ray, COVID cough, genomics
pitfalls) — they are what this audience will recognise.

---

## 1. Per-claim verdict table

| # | Claim | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Frozen encoders encode individual identity + site ≫ behavioural context | **PARTIALLY ANTICIPATED** — the head-to-head *contrast* is unclaimed, but each half is published | Lin et al. 2026 (EEG); Wagner et al. 2023 (TPAMI); Abzaliev et al. 2024 (dog ID 0.50 vs chance 0.05, context 0.62 vs 0.56); Hummel et al. 2026 |
| 2 | Random splits inflate context accuracy +0.10–0.24 vs LOIO/LOSO | **PARTIALLY ANTICIPATED — reframe required.** Mechanism, direction and recommendation all published; magnitude on these corpora is new | Chaibub Neto et al. 2019 ("identity confounding"); Han et al. 2022 (audio, same effect size); Saeb et al. 2017; Colonna et al. 2016 (bioacoustics); Ghani et al. 2026 |
| 3 | Identity-leakage↔inflation correlation holds within corpus, not across | **NOVEL** | Nearest analogue is the "accuracy-on-the-line" literature (Miller et al. 2021) — different quantity, same within/across structure |
| 4 | eGeMAPS > WavLM on pig valence under held-out-lab, while leaking less identity | **NOVEL, and contested — strongest contribution** | Tang et al. 2023 reports the *opposite* in speech (wav2vec2 degrades 0.6% cross-corpus vs eGeMAPS 17.8%); Wagner et al. 2018; Elbanna et al. 2022 |
| 5 | Soundwel's own feature set below chance under held-out-lab | **NOVEL but fragile** — Briefer et al.'s own pDFA already reports 19.5% context vs 14.3% chance | Briefer et al. 2022 (Sci Rep 12:3409) |
| 6 | Context predictable from background alone (0.725 CatMeows) — room confound | **METHOD PARTIALLY ANTICIPATED, FINDING NOVEL** — cite the ancestry or risk rejection | Stowell et al. 2019 (adversarial background mixing, bioacoustics, identity task); Müller et al. 2021 (silence-only shortcut, ASVspoof); Xiao et al. 2021; Beery et al. 2018 |
| 7 | Context axis 1.4–4.2× more robust than identity to channel degradation | **NOVEL as a matched comparison — your most defensible claim.** One direct counter-example to address | Mouterde et al. 2014 (**contradicts you in the animal case**); SVeritas 2025; Signals 7(3):50 (2026) |
| 8a | Marginal coverage holds while per-group collapses | **ALREADY PUBLISHED as a phenomenon**; novel in domain + conditioning axis | Ding et al. 2023 (NeurIPS); Zhong et al. 2026 (13 days old); Han & Qu 2026 |
| 8b | Mondrian vacuous under group-disjoint split | **TEXTBOOK** — one-line corollary | Ding et al. 2023, §1 |
| 8c | Repair needs ≥9 clips | **TEXTBOOK — this is exactly (1/α)−1. Attribute it.** | Ding et al. 2023, §1, verified verbatim |
| 8d | Conformal prediction in bioacoustics/PAM | **NOVEL — verified negative** | Schwinger et al. 2026 (calibration, not conformal) is the nearest |

---

## 2. Claim 1 — identity and site dominate context in frozen encoders

**Verdict: PARTIALLY ANTICIPATED.** The comparison is unclaimed; both halves are published.

- **Lin, J., Wu, Y.-C., & Jung, T.-P. (2026). The Identity Trap in EEG Foundation Models: A
  Diagnostic Audit.** arXiv:2606.06647. <https://arxiv.org/abs/2606.06647>
  *The dangerous one.* Frozen subject-variance is 13–89× a random null in 12/12 model×dataset
  pairs across LaBraM, CBraMod and REVE; subject variance *rises* under fine-tuning (+10 to +63 pp);
  linearly erasing the subject axis improves label decoding (+6 to +12 pp in-cohort, +4 to +27 pp
  external). It does **not** cover recording site, does **not** touch conformal prediction, and is
  not bioacoustics. Two months old — cite it and differentiate explicitly.
- **Wagner, J., Triantafyllopoulos, A., Wierstorf, H., Schmitt, M., Burkhardt, F., Eyben, F., &
  Schuller, B. W. (2023). Dawn of the Transformer Era in Speech Emotion Recognition: Closing the
  Valence Gap.** IEEE TPAMI. arXiv:2203.07378. <https://arxiv.org/abs/2203.07378>
  Transformer SER models are fair with respect to gender groups **but not with respect to
  individual speakers** — the closest existing statement of Claim 1 in speech.
- **Abzaliev, A., Pérez-Espinosa, H., & Mihalcea, R. (2024). Towards Dog Bark Decoding: Leveraging
  Human Speech Processing for Automated Bark Classification.** LREC-COLING 2024. arXiv:2404.18739.
  <https://arxiv.org/abs/2404.18739>
  **Closest bioacoustic prior.** Same modality, wav2vec2, 74 dogs (Pérez-Espinosa Mexican corpus,
  *not* Molnár). Reports dog recognition **49.95% vs 5.03% majority** and context grounding
  **62.18% vs 56.37% majority** — i.e. the numerical pattern of Claim 1, in their own results
  table, unremarked. They already use grouped CV (dogs as group) for breed/gender/context, and
  explicitly note their identity protocol "does not prevent shortcut learning." **They have your
  observation in their table but never state it.** Say so.
- **Hummel, H. I., Bhulai, S., van der Mei, R. D., & Ghani, B. (2026). Decodable but not
  structured: linear probing enables Underwater Acoustic Target Recognition with pretrained audio
  embeddings.** arXiv:2601.08358. <https://arxiv.org/abs/2601.08358>
  *"The geometrical structure of the embedding space is largely dominated by recording-specific
  characteristics."* The **site** half of Claim 1, already published for pretrained audio
  embeddings — just not for animal vocalisations. Note Ghani also co-authored *Twelve quick tips*;
  this group is working your seam.
- **Dumpala, S. H., Dikaios, K., Rodriguez, S., Langley, R., Rempel, S., Uher, R., et al. (2023).
  Manifestation of depression in speech overlaps with characteristics used to represent and
  recognize speaker identity.** Scientific Reports 13. <https://www.nature.com/articles/s41598-023-35184-7>
  Clinical-signal and speaker-identity subspaces overlap — the health-audio version of Claim 1.
- **Miron, M., Robinson, D., Alizadeh, M., et al. (2026). AVEX: What Matters for Animal
  Vocalization Encoding.** ICLR 2026. arXiv:2508.11845. <https://arxiv.org/abs/2508.11845>
  Establishes that identity *is* decodable from these encoders (Pipit ~0.09, Chiffchaff ~0.24,
  Little Owls ~0.65 linear-probe accuracy) — but never as a confound. Cuts both ways: it
  pre-empts half of Claim 1 **and** it is your best evidence that the field's flagship encoder
  benchmark uses label-stratified random splits (0.6/0.2/0.2, seed 4242).
- **Nolasco, I., Cauzinille, J., Miron, M., et al. (2026). Beyond task performance: Decoding
  bioacoustic embeddings with speech features.** Interspeech 2026. arXiv:2606.14662.
  <https://arxiv.org/abs/2606.14662>
  Your closest methodological sibling: regression-probes frozen bioacoustic embeddings with the 88
  eGeMAPS descriptors across six taxa (loudness R²=0.76 best, F0 R²=0.33 worst). Probes for
  *acoustic features*, never for identity/session/site. Frame as: "prior probing asked which
  acoustic features survive; we ask which nuisance factors do."

**Supporting probing literature (speech):** Chiu et al. 2025 (arXiv:2501.05310, 11 models,
speaker attributes across layers); Ashihara et al. 2024 (arXiv:2401.17632, far more layer-to-layer
variation in speaker than phone probing); Pasad et al. 2021 (ASRU, arXiv:2107.04734); Elbanna 2024
(arXiv:2406.10401 — SSL representations are *significantly better* for speaker ID than acoustic
representations; **weakens the "SSL leaks more identity than eGeMAPS" framing unless you cite
it**); Fan et al. 2021 (arXiv:2012.06185, wav2vec2 → 3.61% EER on VoxCeleb1).

**How to word it.** Not "we show encoders encode identity." Instead: *"We show that identity is
encoded more strongly than the behavioural variable the encoder is being used to measure, under a
matched probe — the comparison Lin et al. (2026) make for EEG subjects and Wagner et al. (2023)
observe for speakers, applied for the first time to animal vocalisations and extended to recording
site."*

---

## 3. Claim 2 — random splits inflate context accuracy by +0.10 to +0.24

**Verdict: PARTIALLY ANTICIPATED. This is the claim that must be reframed.**

A reviewer from health ML will know these instantly:

- **Chaibub Neto, E., Pratap, A., Perumal, T. M., Tummalacherla, M., Snyder, P., Bot, B. M., et
  al. (2019). Detecting the impact of subject characteristics on machine learning-based diagnostic
  applications.** npj Digital Medicine 2:99. <https://www.nature.com/articles/s41746-019-0178-x>
  **Coined "identity confounding."** Record-wise (random) splits let the model identify the
  subject rather than the condition, producing massive underestimation of prediction error. This
  is the canonical name for your mechanism — use their term.
- **Han, J., Xia, T., Spathis, D., Bondareva, E., Brown, C., Chauhan, J., et al. (2022). Sounds of
  COVID-19: exploring realistic performance of audio-based digital testing.** npj Digital Medicine
  5:16. <https://www.nature.com/articles/s41746-021-00553-x>
  Participant-independent AUC **0.71**; random split with overlapping participants gives
  sensitivity **0.84** / specificity **0.78**. Verbatim: *"random-splits yielded a higher accuracy
  than user-splits, with the performance gains coming from the overlapping participants whose data
  have been seen from training."* **Claim 2, already published, in audio, with an effect size
  inside your band.**
- **Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Körding, K. P. (2017). The need to
  approximate the use-case in clinical machine learning.** GigaScience 6(5):gix019.
  <https://academic.oup.com/gigascience/article/6/5/gix019/3071704>
  Origin reference for subject-wise vs record-wise cross-validation.

And these are the bioacoustics-side precedents:

- **Colonna, J. G., Gama, J., & Nakamura, E. F. (2016). How to Correctly Evaluate an Automatic
  Bioacoustics Classification Method.** CAEPIA 2016, LNCS 9868, Springer.
  doi:10.1007/978-3-319-44636-3_4
  <https://link.springer.com/chapter/10.1007/978-3-319-44636-3_4>
  Ten years old. Standard k-fold CV *"overestimates recognition accuracy"* for syllable-based
  anuran recognition; proposes *"k-CV by specimens (or individuals)"*; accuracy *"decreases
  considerably."* Their framing question: *"Given a set of syllables that belongs to a specific
  group of individuals, can we recognize new specimens of the same species?"* **The oldest direct
  precedent — missing it would be embarrassing.** Task is species ID, not behavioural context.
- **Ghani, B., Baier, A. L., Kalkman, V. J., & Stowell, D. (2026). Twelve quick tips for applying
  deep learning to animal sounds.** PLOS Computational Biology 22(8):e1014604.
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014604>
  Verbatim: *"Avoid purely random dataset splits, as they are prone to site- or time-based
  leakage"*; *"use spatial or temporal block cross-validation, testing models on sites or time
  periods not seen during training."* Prescriptive, no measurement. **Position your paper as the
  empirical measurement of the harm this tip warns about — not as raising the alarm.**
- **Arnaud, V., Pellegrino, F., Keenan, S., St-Gelais, X., Mathevon, N., Levréro, F., & Coupé, C.
  (2023). Improving the workflow to crack Small, Unbalanced, Noisy, but Genuine (SUNG) datasets in
  bioacoustics: The case of bonobo calls.** PLOS Computational Biology 19(4):e1010325.
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010325>
  Explicitly defines data leakage in bioacoustics, names background soundscape / recording location
  / mic distance as leakage channels, and builds Default / Fair / Skewed split scenarios
  (~5–10 pp balanced-accuracy swing). 1,560 calls, 20 bonobos, 3 zoos. **But** the leakage unit is
  the vocal *sequence*, not the individual; tasks are call-type and individual-ID, not behavioural
  context; hand-crafted features and SVMs, no frozen encoders.
- **Lefèvre, R. A., Sypherd, C. C. R., & Briefer, E. F. (2025). Machine learning algorithms can
  predict emotional valence across ungulate vocalizations.** iScience 28(2):111834.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11847267/>
  **The most awkward citation for your framing.** The Briefer group *itself* already uses grouped
  CV and states the reason verbatim: *"Each fold contained unique individuals, preventing data
  leakage between training and testing set"* and *"prevented the subsequent models from learning
  individual-specific acoustic variables instead of those associated with emotional valence."*
  7 species, 3,181 calls, 89.5% accuracy / 83.9% balanced.
  **However**: they do not quantify what a random split would have given, do not measure identity
  decodability, and **do not address recording site / farm / lab at all.** That is your opening —
  grouping by individual is not enough; the lab axis is the one that breaks (Claim 5).
- **Wilson, O., Schoeman, D. S., Bradley, A. P., & Clemente, C. J. (2025). Practical guidelines
  for validation of supervised machine learning models in accelerometer-based animal behaviour
  classification.** Journal of Animal Ecology. doi:10.1111/1365-2656.70054
  <https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/1365-2656.70054>
  **79% (94/119) of reviewed studies did not validate well enough to detect overfitting.** The
  accelerometer sibling literature has already run your audit on its own field — excellent
  evidence that this is systemic in animal-behaviour ML, and a template for how to pitch it.
- **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al.
  (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure.** Ecography 40(8):913–929. doi:10.1111/ecog.02881
- **Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., et al. (2020).
  Spatial validation reveals poor predictive performance of large-scale ecological mapping
  models.** Nature Communications 11:4540. <https://www.nature.com/articles/s41467-020-18321-y>
- **Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-
  learning-based science.** Patterns 4(9):100804.
  <https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9>
  Your split ablation is their leakage type "train–test not separated by group."

**Speech-side canon (a reviewer will know these):**
Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G.
(2010). *Cross-Corpus Acoustic Emotion Recognition: Variances and Strategies.* IEEE Trans.
Affective Computing 1(2):119–131 — the field overestimates results by using *"pre-selected
prototypical data with less common true speaker disjunctive partitioning."*
Schuller, B., Steidl, S., & Batliner, A. (2009). *The INTERSPEECH 2009 Emotion Challenge.*
Interspeech 2009, 312–315 — the moment SER standardised on speaker-independent partitioning.
<https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.html>
Rybka, J., & Janicki, A. (2013). *Comparison of speaker dependent and speaker independent emotion
recognition.* Int. J. Applied Mathematics and Computer Science 23(4):797–808.
doi:10.2478/amcs-2013-0060 — reports **89.49% speaker-dependent vs 71.29% speaker-independent**,
an ~18-point gap sitting inside your 0.10–0.24 band.

**The 2026 speaker-leakage wave — three papers from the last four months.** None is bioacoustics,
but together they mean "speaker leakage inflates published numbers" is a *live, crowded* topic in
2026 and you cannot present it as a fresh observation:

- **Yeh, S.-L., Sun, ..., Mower Provost, E., & Sisman, B. (2026). Who is Speaking or Who is
  Depressed? A Controlled Study of Speaker Leakage in Speech-Based Depression Detection.**
  Interspeech 2026. arXiv:2604.14354. Controlled speaker-overlap splits at constant training size;
  depression accuracy 62–67% once speaker ID is at chance, higher when identity is preserved.
  **Closest methodological relative to Claims 1 and 3 — they hold training size fixed while varying
  overlap, which is a cleaner design than a straight random-vs-LOIO contrast. Consider adopting
  it, or explain why you didn't.**
- **Pattanayak, et al. (2026).** SLT 2026. arXiv:2607.02920. Speaker leakage inflated a published
  Mandarin depression F1 to 0.954; corrected leave-one-speaker-out gives **0.640**.
- **Sharma, et al. (2026).** arXiv:2606.22699. Reports a published 75% deception-detection accuracy
  as a speaker-leakage artifact.
- **Sahidullah, M., Shim, H.-j., Gonzalez Hautamäki, R., & Kinnunen, T. (2026). Shortcut Learning
  in Binary Classifier Black Boxes: Applications to Voice Anti-Spoofing and Biometrics.** IEEE
  JSTSP. arXiv:2601.17782.

**What is genuinely unclaimed:** no LOIO-vs-random comparison exists on CatMeows, the Molnár dog
corpus, Soundwel pigs, or the Prat bat corpus; no critique of BEANS' random split protocol; and no
one has run this with frozen encoders on behavioural-context/affect targets in animals.

**How to word it.** Not "we discover that random splits inflate." Instead: *"Identity confounding
(Chaibub Neto et al., 2019) is well documented in clinical audio, where participant-independent
evaluation costs ~0.13 AUC (Han et al., 2022), and bioacoustics best-practice guidance has warned
against random splits for a decade (Colonna et al., 2016; Ghani et al., 2026). We measure what that
warning is worth: across four corpora and six feature sets, +0.10 to +0.24 — and we show the
frozen-encoder benchmarks the field currently runs on (BEANS, AVEX) all use random splits."*

---

## 4. Claim 3 — the inflation law (within-corpus yes, cross-corpus no)

**Verdict: NOVEL.** No paper relates a measure of nuisance-information content in a representation
to the size of the optimistic bias from a bad split.

Nearest structural analogue, and the one a reviewer may raise:

- **Miller, J. P., Taori, R., Raghunathan, A., Sagawa, S., Koh, P. W., Shankar, V., Liang, P.,
  Carmon, Y., & Schmidt, L. (2021). Accuracy on the Line: On the Strong Correlation Between
  Out-of-Distribution and In-Distribution Generalization.** ICML 2021. arXiv:2107.04649.
  <https://arxiv.org/abs/2107.04649>
  Same *within-vs-across* structure (a correlation that holds inside a distribution family and
  breaks outside it), different quantity. Cite as the methodological precedent for reporting a
  scoped correlation with its failure regime, and for the honesty of reporting the null.

- **Accuracy on the wrong line: on the pitfalls of noisy data for OOD generalisation.**
  arXiv:2406.19049. **Cite this one prominently** — it is specifically about *when the linear
  correlation breaks*, which is structurally your within-vs-across-corpus result.
- **Baek, C., Jiang, Y., Raghunathan, A., & Kolter, J. Z. (2022). Agreement-on-the-Line.** NeurIPS
  2022. arXiv:2206.13089.
- **Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B., & Schmidt, L. (2020). Measuring
  Robustness to Natural Distribution Shifts in Image Classification.** NeurIPS 2020.
  arXiv:2007.00644.

> **⚠ Statistical caution — a reviewer will catch this.** r = 0.221 at n = 12 has roughly **15%
> power** against a true effect of 0.646. You cannot claim the correlation is *absent* across
> corpora — only that you **failed to detect it**. Word it as "we do not detect the relationship at
> the corpus level (r=0.221, n=12, n.s.); with n=12 this test is severely underpowered and the
> result should be read as inconclusive rather than null." Doing this pre-emptively converts a
> vulnerability into a credibility signal, which is exactly what the ICBINB reviewer rubric rewards
> under "quality of discussion of limitations."

**Guidance:** the cross-corpus non-result is a strength for ICBINB, not a weakness — it is exactly
the "we tried to state a law and it didn't generalise" result the venue asks for. Keep it
prominent, keep the partial-r caveat about cluster-level non-independence, and keep the power
caveat above.

---

## 5. Claim 4 — eGeMAPS beats WavLM under held-out-lab

**Verdict: NOVEL, and contested by the speech literature. Lead with this claim — but you must
engage the counter-evidence or a reviewer will hand it to you.**

**Counter-evidence you must cite and rebut:**

- **Tang, D., Kuppens, P., Geurts, L., & van Waterschoot, T. (2023). End-to-end transfer learning
  for speaker-independent cross-language and cross-corpus speech emotion recognition.**
  arXiv:2311.13678. <https://arxiv.org/abs/2311.13678>
  wav2vec 2.0 beats eGeMAPS by **20.9–66.4% UA within-language**, and under cross-language shift
  wav2vec 2.0 shows *"a very subtle degradation… whereas the eGeMAPS-CL shows a 17.8% and 18.5%
  decrease in UA and WA."* **The direct rebuttal to Claim 4.** You need one paragraph explaining
  why held-out-*lab* differs from cross-corpus SER — candidate explanations: recording-channel
  homogeneity within a lab, call duration, non-speech source-filter physics that WavLM's
  speech-tuned pretraining does not cover, and the small number of labs (n=5).
- **Elbanna, G. (2024). Evaluating Speaker Identity Coding in Self-supervised Models and Humans.**
  arXiv:2406.10401. <https://arxiv.org/abs/2406.10401>
  SSL representations are *significantly better for speaker identification than acoustic
  representations.* This **supports** the "WavLM leaks more identity than eGeMAPS" half of your
  claim — cite it as corroboration, not as a threat.

**Supporting evidence:**

- **Wagner, J., Schiller, D., Seiderer, A., & André, E. (2018). Deep Learning in Paralinguistic
  Recognition Tasks: Are Hand-crafted Features Still Relevant?** Interspeech 2018, 147–151.
  <https://www.isca-archive.org/interspeech_2018/wagner18_interspeech.html> — *"there is no clear
  winner (yet)."*
- **Elbanna, G., Scheidwasser-Clow, N., Kegler, M., Beckmann, P., El Hajal, K., & Cernak, M.
  (2022). BYOL-S / hybrid handcrafted-learnable audio representations.** arXiv:2203.16637,
  arXiv:2206.12038 — hand-crafted ComParE features beat most SSL models on paralinguistic tasks.
  **Strongest existing evidence in your direction.**
- **Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., et al. (2022). WavLM: Large-Scale
  Self-Supervised Pre-Training for Full Stack Speech Processing.** IEEE JSTSP 16(6):1505–1518.
  arXiv:2110.13900. <https://arxiv.org/abs/2110.13900>
  **Load-bearing rhetorical point:** WavLM's utterance-mixing and denoising objectives were
  *designed* to preserve speaker identity. WavLM leaking identity is a design goal, not an
  accident. Say this explicitly — it turns your result from "a model underperformed" into "the
  model did exactly what it was built to do, and that is the problem."
- **Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., et al. (2016).
  The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective
  Computing.** IEEE Trans. Affective Computing 7(2):190–202. doi:10.1109/TAFFC.2015.2457417
- **Eyben, F., Wöllmer, M., & Schuller, B. (2010). openSMILE — The Munich Versatile and Fast
  Open-Source Audio Feature Extractor.** ACM Multimedia 2010, 1459–1462.
  <https://dl.acm.org/doi/10.1145/1873951.1874246>

**The cross-domain near-twin — cite it early and explicitly:**

- **Zare, M. (2026). A Negative-Control Protocol for Clinical EEG Foundation-Model Benchmarks:
  Dataset Identity and External-Cohort Stress Testing.** arXiv:2607.24519 (July 2026).
  <https://arxiv.org/abs/2607.24519>
  **This is your architecture in another modality, one month old.** Classical features reach 0.734
  macro-AUROC vs BIOT 0.677, CBraMod 0.669, REVE 0.568; on a held-out subset, classical 0.717 vs
  REVE 0.565; a *randomly initialised* encoder (0.667) beats pretrained REVE (0.570); and **all
  five encoders decode dataset identity at 1.000**, which the author ties directly to benchmark
  inflation. That is Claims 1, 4 and 6 combined. It is EEG, not audio, and single-author
  unrefereed — so cite it as **convergent cross-domain evidence**, which is a strength, not a
  scoop. Together with Lin et al. (2026), it lets you argue this is a *general property of frozen
  foundation-model benchmarks*, which is a bigger claim than a bioacoustics one.

Other handcrafted-beats-deep precedents:
**Pascu, O., Oneață, D., Cucu, H., & Müller, N. M. (2025). Easy, Interpretable, Effective:
openSMILE for voice deepfake detection.** ICASSP 2025. arXiv:2408.15775 — eGeMAPSv2's 88 features
outperform far more complex detectors.
**Vlasenko, B., et al. (2024). Comparing data-driven and handcrafted features for dimensional
emotion recognition.** ICASSP 2024 — handcrafted+wav2vec2 hybrid best for arousal/dominance.
*(Numbers unverified — the PDF would not parse; check before quoting.)*
**Wierucka et al. (2025)** MEE — MFCC + random forest is the most *consistently reliable*
combination across 16 mammal datasets.
**Schwinger et al. (2026)** *Foundation Models for Bioacoustics — a Comparative Review*
(arXiv:2508.01277) does **not** compare handcrafted features against foundation models, so the
bioacoustics-specific version of this comparison is genuinely open.

Benchmark a reviewer may ask about: **Zhang, Z., et al. (2024). ParaLBench: A Large-Scale
Benchmark for Computational Paralinguistics over Acoustic Foundation Models.** arXiv:2411.09349 —
10 datasets, 13 tasks, 14 foundation models; explicitly names cross-corpus generalisability as the
open problem.

> **⚠ Statistical caution.** 0.669 vs 0.660 is a 0.9-point gap. Without confidence intervals or a
> paired test across held-out labs, this will not survive review as "eGeMAPS **beats** WavLM."
> Either compute a paired test across the five labs, or soften to *"matches, and does so while
> leaking substantially less identity"* — which is the more interesting claim anyway and is robust
> to the gap being noise.

---

## 6. Claim 5 — the Soundwel feature set below chance under held-out-lab

**Verdict: NOVEL but fragile. Reword to avoid a strawman accusation.**

- **Briefer, E. F., Sypherd, C. C.-R., Linhart, P., Leliveld, L. M. C., Padilla de la Torre, M.,
  Read, E., et al. (2022). Classification of pig calls produced from birth to slaughter according
  to their emotional valence and context of production.** Scientific Reports 12:3409.
  doi:10.1038/s41598-022-07174-8 <https://www.nature.com/articles/s41598-022-07174-8>
  (Author Correction: Sci Rep 2023, doi:10.1038/s41598-023-45242-9.)

  Verified details you need:
  - 7,414 calls, 411 pigs, from **five research laboratories** (DE, CH, CZ, NO).
  - Neural network: *"The dataset was randomly split 70/30 into a training and validation set each
    time the neural network was trained"* — **random over calls, no lab held out.**
  - Pig ID was nested within providing team as a random effect in the *statistical* models only;
    the classifier saw no such structure.
  - NN: valence 91.5% (chance 50%), context 81.5% (chance 14.3%).
  - pDFA: valence 61.7% (chance 50.5%), **context 19.5% (chance 14.3%)**.
  - Feature set: duration, amplitude-modulation rate, spectral centre of gravity (Q50%), Wiener
    entropy.

**The risk:** their own pDFA context result is already only ~5 pp above chance. A reviewer can say
"they already showed the feature set barely works."

**How to word it.** Not "their feature set fails." Instead: *"Briefer et al. (2022) already report
that their four-descriptor set reaches only 19.5% context accuracy against 14.3% chance under a
random split. We complete that observation: under leave-one-lab-out — the evaluation the five-lab
provenance of the corpus makes possible but which was never run — the same feature set falls to
0.386 against 0.500 on valence, i.e. below chance. The published number was not merely weak; it
was measuring lab provenance."*

---

## 7. Claim 6 — the background-only control and the CatMeows room confound

**Verdict: PARTIALLY ANTICIPATED on method, NOVEL on finding. Revised after direct verification —
see the boxed note below, which corrects a harsher first reading.**

> **Verification note.** A first pass concluded that Stowell et al. (2019) had already published
> the exact background-only control. I read the full text (Europe PMC PMC6505557) and the truth is
> more favourable to you, but you must still cite them prominently.
>
> **Stowell, D., Petrusková, T., Šálek, M., & Linhart, P. (2019). Automatic acoustic
> identification of individuals in multiple species: improving identification across recording
> conditions.** Journal of the Royal Society Interface 16(153):20180940. doi:10.1098/rsif.2018.0940
> · preprint arXiv:1810.09273
>
> What they actually do: they keep the normally-discarded clips *"in which the vocal individual is
> silent"* and use them for **(a) adversarial background mixing** — *"we created datasets in which
> each foreground recording has been mixed with one background recording from some other
> individual… In the best case, this should make no difference"* — and **(b)** an
> *"'explicit-background' training category"*, which they report *"provided only a mild
> improvement."*
>
> What they do **not** do: train and test a classifier on background frames alone and read off the
> accuracy. And their target is **individual identity**, not behavioural context.
>
> Their motivating example is uncannily close to yours: *"if one territory is close to a river and
> another is not, then a black-box classifier might use the sounds of the river itself… rather than
> the sounds that the individuals themselves make."* And their closing recommendation —
> *"assessment of confounds should become a standard part of future studies to ensure they do not
> report over-optimistic results"* — is your paper's recommendation, published in 2019.
>
> **Net:** the diagnostic *family* is established in bioacoustics; the background-only probe of
> **behavioural-context** labels, and the CatMeows room-by-construction result, are new. Cite
> Stowell et al. as the method you are extending, in your Methods section, not buried in Related
> Work. Presenting the control as your invention is the single most likely cause of rejection by a
> bioacoustics reviewer.

**The CatMeows room confound itself is confirmed and unclaimed.** No published critique exists.

I verified the CatMeows induction protocol directly from the Europe PMC full text
(PMC6719916). Verbatim:

> *"Isolation in unfamiliar environment: Cats were transported by their owner, adopting the normal
> routine used to transport them for any other reason, to an unfamiliar environment (e.g., a room
> in a different apartment or an office, not far from their home environment)… where they stayed
> alone for a maximum of 5 min. Brushing: Cats were brushed by their owner in their home
> environment for a maximum of 5 min."*

And, revealingly, from the same paper's own data-acquisition section:

> *"the characteristics of the room (e.g., topological properties, reverberations, presence of
> furniture, etc.) should not affect the captured audio signals. Data acquisition could not be
> conducted in a controlled anechoic environment…"*

**The authors identified room acoustics as a threat, stated a requirement that it not affect the
signal, and then induced one of three classes by moving the cat to a different room.** Quote both
passages side by side — that is the figure caption for `room.png`.

**Citation correction (your draft has this wrong):** the dataset paper is
**Ludovico, L. A., Ntalampiras, S., Presti, G., Cannas, S., Battini, M., & Mattiello, S. (2021).
CatMeows: A Publicly-Available *Dataset* of Cat Vocalizations.** MultiMedia Modeling (MMM 2021),
LNCS 12573, Springer, 230–243. doi:10.1007/978-3-030-67835-7_20 — **not PeerJ CS, and "Dataset"
not "Corpus."**
The classification paper is **Ntalampiras, S., Ludovico, L. A., Presti, G., Prato Previde, E.,
Battini, M., Cannas, S., Palestrini, C., & Mattiello, S. (2019). Automatic Classification of Cat
Vocalizations Emitted in Different Contexts.** Animals 9(8):543. doi:10.3390/ani9080543 —
448 files, 21 cats, **ten-fold cross-validation with no stated cat-independence**, best result
95.94%.

**Cross-domain precedents to cite for the background-only ablation:**

- **Xiao, K., Engstrom, L., Ilyas, A., & Madry, A. (2021). Noise or Signal: The Role of Image
  Backgrounds in Object Recognition.** ICLR 2021. arXiv:2006.09994.
  <https://arxiv.org/abs/2006.09994> — background-only models reach 40–50% vs 11% chance. The
  direct visual analogue of your control.
- **Beery, S., Van Horn, G., & Perona, P. (2018). Recognition in Terra Incognita.** ECCV 2018.
  arXiv:1807.04975. <https://arxiv.org/abs/1807.04975> — camera-trap species recognition collapses
  at new locations. The ecology-facing analogue; a biology audience will know it.
- **Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018).
  Variable generalization performance of a deep learning model to detect pneumonia in chest
  radiographs: A cross-sectional study.** PLOS Medicine 15(11):e1002683. — the hospital-token
  shortcut.
- **DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic COVID-19 detection
  selects shortcuts over signal.** Nature Machine Intelligence 3:610–619.
  doi:10.1038/s42256-021-00338-7 — **co-authored by an ICBINB-BIO organizer.** Cite prominently.
- **Song, D., et al. (2024). Identifying bias in models that detect vocal fold paralysis from
  audio recordings using explainable machine learning and clinician ratings.** PLOS Digital Health
  3(5):e0000516. — recording-channel Clever Hans in clinical voice: patients with softer voices had
  mic gain increased, and the model exploited it. Under-cited and very close in spirit.

Also relevant, and it means you cannot claim "shortcuts exist in audio" as a contribution:
**SpurAudio: A Benchmark for Studying Shortcut Learning in Few-Shot Audio Classification**
(arXiv:2605.13672) — severe degradation when background correlations are disrupted, persisting
across large pretrained audio foundation models.

**The speech-domain twin of your control, which you must also cite:**

- **Müller, N. M., Dieckmann, F., Czempin, P., Canals, R., Böttinger, K., & Williams, J. (2021).
  Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?** ASVspoof
  2021 Workshop. doi:10.21437/asvspoof.2021-9 · arXiv:2106.12914
  <https://arxiv.org/abs/2106.12914>
  Leading-silence *duration alone* reaches ~85% accuracy / 15.1% EER on ASVspoof; trimming silence
  moves a standard system from 3.6% to 15.5% EER. Structurally identical to your 0.725 from
  background frames. *(These numbers came from indexed summaries rather than the PDF — verify
  before quoting them in the paper.)*

**No bioacoustics paper runs a background-only / non-vocal-frames-only baseline on animal
vocalisation *behavioural-context* classification.** The identity version exists (Stowell et al.
2019, via adversarial mixing); the context version does not. Keep the figure — just cite its
ancestry.

**Second pre-emption:** a reviewer can argue the room confound is *disclosed* in the CatMeows
protocol and your result is merely confirmatory. Answer this in the text: the interesting quantity
is the **magnitude** (0.725 on the binary task from background frames alone), and the fact that
the dataset paper's own stated design requirement — that room characteristics *"should not affect
the captured audio signals"* — is violated by its own class-induction procedure.

---

## 8. Corpus and benchmark references (verified)

- **Ludovico, L. A., Ntalampiras, S., Presti, G., Cannas, S., Battini, M., & Mattiello, S. (2021).
  CatMeows: A Publicly-Available Dataset of Cat Vocalizations.** MMM 2021, LNCS 12573, 230–243.
  doi:10.1007/978-3-030-67835-7_20
- **Ntalampiras, S., et al. (2019). Automatic Classification of Cat Vocalizations Emitted in
  Different Contexts.** Animals 9(8):543. doi:10.3390/ani9080543
- **Molnár, C., Kaplan, F., Roy, P., Pachet, F., Pongrácz, P., Dóka, A., & Miklósi, Á. (2008).
  Classification of dog barks: a machine learning approach.** Animal Cognition 11:389–400.
  doi:10.1007/s10071-007-0129-9 · free PDF:
  <https://www.francoispachet.fr/wp-content/uploads/2021/01/molnar-08a.pdf>
  6,000+ barks, 6 situations; **43% context, 52% individual** — again the Claim 1 pattern, in a
  2008 table. No reanalysis or critique exists.
- **Briefer, E. F., et al. (2022).** Sci Rep 12:3409 — see §6.
- **Prat, Y., Taub, M., & Yovel, Y. (2016). Everyday bat vocalizations contain information about
  emitter, addressee, context, and behavior.** Scientific Reports 6:39419.
  <https://www.nature.com/articles/srep39419>
  Their GMM-UBM reports *"different aggressive contexts for each emitter: 75%; emitters: 71%"* —
  they **conditioned on emitter rather than holding emitters out.** A partial control, not LOIO.
- **Prat, Y., Taub, M., Pratt, E., & Yovel, Y. (2017). An annotated dataset of Egyptian fruit bat
  vocalizations across varying contexts and during vocal ontogeny.** Scientific Data 4:170143.
- **Hagiwara, M., Hoffman, B., Liu, J.-Y., Cusimano, M., Effenberger, F., & Zacarian, K. (2023).
  BEANS: The Benchmark of Animal Sounds.** ICASSP 2023. arXiv:2210.12300.
  <https://arxiv.org/abs/2210.12300>
  Confirmed: classification datasets *"randomly split into 6:2:2 train:valid:test portions with
  stratification."* No individual or site grouping. **No published critique of this protocol
  exists — your critique is unclaimed.**
- **Hagiwara, M. (2023). AVES: Animal Vocalization Encoder based on Self-Supervision.** ICASSP
  2023. arXiv:2210.14493. — no probing of what the embedding encodes.
- **Rauch, L., Schwinger, R., Wirth, M., Heinrich, R., et al. (2025). BirdSet: A Large-Scale
  Dataset for Audio Classification in Avian Bioacoustics.** ICLR 2025. arXiv:2403.10380.
  Built around focal→soundscape covariate shift with geographically distinct test soundscapes.
  **The site-generalisation gap for birds is already benchmarked** — scope your site claim to
  non-avian / behavioural-context data, or frame it as "the identity analogue of BirdSet's site
  shift."
- **Schwinger, R., Vali Zadeh, P., Rauch, L., et al. (2026). Foundation models for bioacoustics —
  a comparative review.** Ecological Informatics. arXiv:2508.01277. Notes BirdNET v2.4 was trained
  on BirdSet soundscape *evaluation* subsets — an explicit contamination admission.
- **Wierucka, K., Murphy, D., Watson, S. K., Falk, N., Fichtel, C., León, J., Leu, S. T., Kappeler,
  P. M., et al. (2025). Same data, different results? Machine learning approaches in
  bioacoustics.** Methods in Ecology and Evolution 16:1574–1586. doi:10.1111/2041-210X.70091
  16 mammalian datasets, 3 feature extractors × 5 classifiers. **Identity is the task, not the
  confound.** Useful as the reference point for what a well-run identity classifier achieves.
- **Osiecka, A. N., Lefèvre, R. A., & Briefer, E. F. (2025). Emotional contexts influence vocal
  individuality in ungulates.** Animal Behaviour (preprint bioRxiv 2024.09.18.613506).
  Negative-valence calls carry *less* individual information; high arousal partially overrides
  identity. **Identity and affect are not independent** — this matters for how you interpret an
  identity probe, and you should acknowledge it in Limitations.
- **Lostanlen, V., Salamon, J., Farnsworth, A., Kelling, S., & Bello, J. P. (2019). Robust sound
  event detection in bioacoustic sensor networks.** PLOS ONE 14(10):e0214168. — sensor/site
  generalisation in PAM.

**Defensive citation you need:** *Multi-layer attentive probing improves transfer of audio
representations for bioacoustics* (arXiv:2605.10494) — encoder rankings change with probe head.
Pre-empt "your linear probe understates what the encoder encodes."

---

## 9. Claim 7 — channel robustness of context vs identity

**Verdict: NOVEL as a matched comparison — probably your single most defensible claim. But each
half is separately published, and there is a direct bioacoustic counter-example you must address.**

No paper anywhere makes the matched, difficulty-controlled side-by-side comparison with a
robustness ratio. Exhaustive arXiv abstract sweeps across `speaker identity × emotion ×
reverberation`, `emotion recognition × speaker verification × degradation`, and `paralinguistic ×
channel/codec/reverberation` returned nothing that does this. **The claim survives.**

**⚠ The landmine — a bioacoustics reviewer will raise this immediately:**

- **Mouterde, S. C., Theunissen, F. E., Elie, J. E., Vignal, C., & Mathevon, N. (2014). Acoustic
  Communication and Sound Degradation: How Do the Individual Signatures of Male and Female Zebra
  Finch Calls Transmit over Distance?** PLOS ONE 9(7):e102842. doi:10.1371/journal.pone.0102842
  Individual signature is **remarkably resistant** to propagation degradation, recoverable beyond
  100 m. **This contradicts your direction in the animal case.** The same group's J. Neurosci.
  37(13):3491 (2017) shows avian cortical neurons encode identity *and* propagation distance in
  naturally degraded calls, and the penguin literature (Aubin & Jouventin) argues identity coding
  *evolved* to resist degradation.

  **Your defence, which must be in the text:** you measure *artificial channel* degradation applied
  to *already-recorded* datasets, and your "identity axis" is **dataset-identity leakage** — a mix
  of vocal-tract signature, microphone, room and session — not an evolved biological signature.
  Those are different objects. Say this explicitly and early, or the objection lands.

**Each half, separately published:**

- **Bisht, et al. (2025). SVeritas: Benchmark for Robust Speaker Verification under Diverse
  Conditions.** Findings of EMNLP 2025. arXiv:2509.17091. <https://arxiv.org/abs/2509.17091>
  Identity-side degradation across noise, RT60, codecs and bandwidth. Notably **WavLM degrades
  fastest (EER 23% → >40%)** — which supports Claim 7's identity-fragility *and* Claim 4
  simultaneously. Useful double-duty citation.
- **Spectral Bandwidth Effects on Emotion Classification and Representation in Spoken and Sung
  Signals.** Signals 7(3):50 (2026). doi:10.3390/signals7030050
  RAVDESS low-passed at 8/12/16 kHz; bandwidth restriction has *"relatively modest effects"* on
  emotion accuracy. **The context-is-robust-to-low-pass half, in print two months before your
  deadline.**
- **Siegert, I., et al. (2016). Emotion Intelligibility within Codec-Compressed and Reduced
  Bandwidth Speech.** ITG Speech Communication.
- **Dineley, J., et al. (2023). Towards robust paralinguistic assessment for real-world mHealth
  monitoring: reverberation effects on speech.** arXiv:2305.12514
- **EMO-Codec** (arXiv:2407.15458); *A digital "flat affect"?* Frontiers in Communication (2023),
  doi:10.3389/fcomm.2023.972182
- **Dehak, N., Kenny, P. J., Dehak, R., Dumouchel, P., & Ouellet, P. (2011). Front-End Factor
  Analysis for Speaker Verification.** IEEE TASLP 19(4):788–798. doi:10.1109/TASL.2010.2064307
  Cite one such reference to show you know that channel/session variability is the *founding
  premise* of the speaker-verification field — the identity axis was always expected to be fragile.

**Adjacent and relevant to why your site probe works at all** — rooms and microphones are trivially
identifiable from audio:
Malik, H. (2013). *Acoustic Environment Identification and Its Applications to Audio Forensics.*
IEEE TIFS 8(11):1827–1837 · Zhao, H., & Malik, H. (2013). *Audio Recording Location Identification
Using Acoustic Environment Signature.* IEEE TIFS · Das, A., Borisov, N., & Caesar, M. (2014).
*Fingerprinting Smart Devices Through Embedded Acoustic Components.* ACM CCS. arXiv:1403.3366

---

## 10. Claim 8 — conformal prediction, per-group coverage

**Verdict: mixed and requiring the most careful rewriting of any claim in the paper.**

| Sub-claim | Verdict |
|---|---|
| Marginal ≈0.90 while per-group coverage collapses | **ALREADY PUBLISHED as a phenomenon** — novel only in domain and conditioning axis |
| Mondrian/group-conditional vacuous under group-disjoint splits | **TEXTBOOK** — one-line corollary of a published fact |
| Repair requires ≥9 labelled clips | **TEXTBOOK — this is exactly (1/α)−1. Must be attributed, not claimed.** |

### 10.1 The ≥9 result is a known bound. Fix this first.

I verified the source text directly. **Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I., &
Tibshirani, R. J. (2023). Class-Conditional Conformal Prediction with Many Classes.** NeurIPS 2023.
arXiv:2306.09335. Section 1, verbatim:

> *"Importantly, for any class y for which |ℐ^y| < (1/α) − 1, we will have q̂^y = ∞, hence any
> prediction set generated by classwise will include y, no matter the values of the conformal
> scores."*

At α = 0.10, (1/α) − 1 = **9**. Your "repair requires ≥9 labelled clips from the deployment group"
**is this bound.** The underlying arithmetic: q̂ is the ⌈(n+1)(1−α)⌉-th order statistic, which
exists only if ⌈(n+1)(1−α)⌉ ≤ n. At n=8, ⌈8.1⌉ = 9 > 8 (fails); at n=9, ⌈9⌉ = 9 ≤ 9 (works).

**Do not cite Angelopoulos & Bates for this** — their tutorial states the guarantee holds for any
n and treats calibration size purely as a variance question, which is the opposite of your point.

**This also settles the Mondrian sub-claim.** Under a group-disjoint split the deployment group has
n_g = 0 < (1/α) − 1, so q̂ = ∞ and the prediction set is all of 𝒴 immediately. Structural vacuity
is a one-line corollary, not a finding.

**Rewrite to:** *"The repair requires at least (1/α)−1 = 9 labelled clips from the deployment group
— the standard finite-sample floor below which the group-conditional quantile is infinite (Ding et
al., 2023). The contribution is not the bound but its ecological cost: nine labelled clips per
individual animal, in a field where labelling requires simultaneous behavioural observation, is a
substantial and rarely-budgeted burden."* That reframing is defensible and is an ecology
contribution rather than a statistics one.

### 10.2 Adopt their metric names

Ding et al. define **CovGap** = 100 × mean_y |ĉ_y − (1−α)| and, in Appendix C.2, a **fraction
undercovered** metric. Your "5 of 20 cats below 0.80" is FracUnderCov. Rename your metrics to
match — it makes the paper searchable and signals fluency. Note that "worst-group coverage" returns
essentially nothing in the literature, whereas "class-conditional coverage" and "coverage gap" are
the community's terms.

### 10.3 Concede the phenomenon, cite the impossibility results up front

- **Foygel Barber, R., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The limits of
  distribution-free conditional predictive inference.** Information and Inference 10(2):455–482.
  arXiv:1903.04684. **Open the subsection with this** so the reviewer sees the failure as expected.
- **Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric
  regression.** JRSS-B 76(1):71–96. arXiv:1203.5422.
- **Vovk, V. (2012). Conditional validity of inductive conformal predictors.** ACML 2012,
  PMLR 25:475–490 (journal version Machine Learning 92:349–376). arXiv:1209.2673.

### 10.4 Prior empirical subgroup-collapse demonstrations (the reviewer's ammunition)

1. **Ding et al. 2023** — ImageNet at α=0.10: marginal **89.8%**, class-conditional **50.8%** to
   99.2%; CovGap 5.2 (ImageNet), 7.0 (iNaturalist). The closest and most authoritative.
2. **Zhong, L., Wang, X., Huang, S., & Shi, Y. (2026). When Is a Conformal Guarantee Fair? Auditing
   Silent Subgroup Under-Coverage in Alzheimer's Disease Longitudinal Prediction.** arXiv:2608.04254
   (**4 Aug 2026 — thirteen days old**). ADNI + OASIS-3; under-coverage in **57 of 68 audited
   subgroup combinations despite nominal marginal coverage**; carries a **k/(n+1) rarity bound**
   that is the same finite-sample fact as your ≥9. **Your biggest framing threat — read it in full
   before finalising.** Mitigation: they condition on *attribute-defined* subgroups, not individual
   identity.
3. **Han & Qu (2026). The Label Complexity of Class-Conditional Coverage under Distribution
   Shift.** arXiv:2607.18088. NTU RGB+D 60 **cross-subject**: marginal **0.881**, worst class
   **0.699**, **10 of 60 classes below 0.80**; needs O(1/α) labels per class. Structurally near
   identical to your headline sentence — but conditions by *class*, not subject. Unrefereed
   two-author preprint; cite, do not lean on.
4. **Rafe & Das (2026). Socio-Conformal Calibration in Complex Survey Data: Marginal Validity Is
   Not Enough for Subgroup Reliability.** arXiv:2605.05562. **Most useful defensively** — on Pew
   ATP, Mondrian *worsens* the gap (+0.013) via calibration-cell fragmentation. Pre-empts "just use
   Mondrian."
5. **Cresswell, J. C., Kumar, B., Sui, Y., & Belbahri, M. (2025). Conformal Prediction Sets Can
   Cause Disparate Impact.** ICLR 2025 Spotlight. arXiv:2410.01888. Enforcing equalized coverage
   *increases* disparate impact — "the obvious fix isn't free."
6. **Mehrtens, H., Bucher, T., & Brinker, T. J. (2025). Pitfalls of Conformal Predictions for
   Medical Image Classification.** arXiv:2506.18162. Best *tonal* match — an ICBINB-style pitfalls
   paper. ⚠ The widely-quoted "92% white / 62% black" figure is an **illustrative hypothetical in
   that paper, not a measured result** — do not repeat it as data.
7. **Tursunbadalov & Tursunbadalov (2026). A Quiet Failure in Calibrated Virtual Screening.**
   arXiv:2607.06605. Biology domain; minority coverage 64.8% / 4.2% at 90% target. Unrefereed.
8. **Romano, Y., Barber, R. F., Sabatti, C., & Candès, E. J. (2020). With Malice Toward None:
   Assessing Uncertainty via Equalized Coverage.** Harvard Data Science Review 2(2).
   arXiv:1908.05428. Origin of equalized coverage — but be honest: its MEPS gap is 92.0% vs 87.1%,
   ~5 pp, not a collapse.
9. **Lu, C., Lemay, A., Chang, K., Höbel, K., & Kalpathy-Cramer, J. (2022). Fair Conformal
   Predictors for Applications in Medical Imaging.** AAAI 36:12008–12016. arXiv:2109.04392.
   Reports *disparity* (0.026→0.022), not absolute per-group coverage — **do not cite as a
   collapse.**
10. **Braun, et al. Conditional Coverage Diagnostics for Conformal Prediction.** arXiv:2512.11779.
    Shows CovGap is statistically underpowered — the right citation for *how* to measure this, and
    a caution about your own n=20 cats.

### 10.5 Group-conditional machinery you should name

Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction
Under Covariate Shift.* NeurIPS 32:2526–2536 · Bastani, O., Gupta, V., Jung, C., Noarov, G.,
Ramalingam, R., & Roth, A. (2022). *Practical Adversarial Multivalid Conformal Prediction.* NeurIPS
2022. arXiv:2206.01067 · Jung, C., Noarov, G., Ramalingam, R., & Roth, A. (2023). *Batch Multivalid
Conformal Prediction.* ICLR 2023. arXiv:2209.15145 · Gibbs, I., Cherian, J. J., & Candès, E. J.
(2025). *Conformal Prediction with Conditional Guarantees.* JRSS-B 87(4):1100–1126. arXiv:2305.12616

### 10.6 Few-shot calibration — for the recovery-curve section

- **Fisch, A., Schuster, T., Jaakkola, T., & Barzilay, R. (2021). Few-shot Conformal Prediction
  with Auxiliary Tasks.** ICML 2021, PMLR 139:3329–3339. arXiv:2102.08898. **Your canonical
  citation here.**
- **Park, S., Cohen, K. M., & Simeone, O. (2022). Few-Shot Calibration of Set Predictors via
  Meta-Learned Cross-Validation-Based Conformal Prediction.** arXiv:2210.03067. Preserves
  **per-task** rather than task-marginal calibration — the stronger notion, and the one you
  actually want per animal.
- **Silva-Rodríguez, J., Ben Ayed, I., & Dolz, J. (2025). Trustworthy Few-Shot Transfer of Medical
  VLMs through Split Conformal Prediction.** MICCAI 2025. arXiv:2506.17503. Most on-point for "new
  group, tiny labelled calibration set"; notes adapting on the calibration set breaks
  exchangeability — **a trap your recovery experiment may fall into; check it.**
- **Dunn, R., Wasserman, L., & Ramdas, A. (2022). Distribution-Free Prediction Sets for Two-Layer
  Hierarchical Models.** JASA. arXiv:1809.07441. The right formalism for group-clustered data.

### 10.7 Conformal prediction in bioacoustics/PAM — verified negative

The "no prior work" claim is **safe to make in print.** Exhaustive arXiv abstract searches:
`abs:conformal` AND (animal OR bioacoustic OR vocalisation) → 83 hits, **zero** genuine (all
"conformal" as geometry or Conformer architecture); `abs:bioacoustic*` AND (uncertainty OR
calibration) → 10 hits, **zero** conformal. An independent OpenAlex keyword sweep agreed.

Nearest neighbours, which you should cite so the claim looks researched rather than lazy:

- **Schwinger, R., McEwen, ..., Rauch, L., & Tomforde, S. (2025/26). Uncertainty Calibration of
  Multi-Label Bird Sound Classifiers.** Accepted ICAART 2026. arXiv:2511.08261.
  **Your must-cite "closest prior work in bioacoustics."** Benchmarks Perch v2, ConvNeXt_BS,
  AudioProtoPNet, BirdMAE on ECE/MCE, evaluating global, per-dataset *and per-class* calibration.
  Uses calibration, not conformal — no prediction sets, no coverage. This is exactly what makes
  your conformal framing non-redundant. Say so.
- **de Castelbajac, ..., Bonnet, P. (2025). Conformal taxonomic validation: A semi-automated
  validation framework for citizen science records.** Ecological Informatics.
  doi:10.1016/j.ecoinf.2025.103290. Conformal over taxonomic ranks, ~25,000 jellyfish records —
  **images, not audio**; no group-conditional analysis.
- **Poisot, T. (2024/25). Conformal Prediction quantifies the uncertainty of Species Distribution
  Models.** EcoEvoRxiv. doi:10.32942/X2CD1J.
- **Melki, P., Bombrun, L., Diallo, B., Dias, J., & da Costa, J.-P. (2023). Group-Conditional
  Conformal Prediction via Quantile Regression Calibration for Crop and Weed Classification.**
  ICCV Workshops 2023. arXiv:2308.15094.

*Caveat on the negative:* the sweep of non-indexed ecology journals (Methods in Ecology and
Evolution, Remote Sensing in Ecology and Conservation) was via web search rather than exhaustive
API enumeration. Phrase the claim as "to our knowledge" rather than absolutely.

### 10.8 What actually survives

1. **Conditioning axis.** Every strong exemplar conditions on *class label* or an
   *attribute-defined subpopulation*. **Zero** papers condition on *individual entity identity*
   (per-animal / per-subject / per-speaker) and report per-individual coverage. **This is your most
   defensible novelty in Claim 8.**
2. **Domain.** No conformal prediction in bioacoustics/PAM.
3. **Severity.** Worst group 0.412 is at the extreme end of the published range (0.508 Ding,
   0.699 Han & Qu), beaten only by an unrefereed preprint.

---

## 11. Submission logistics — ICBINB-BIO @ NeurIPS 2026

All confirmed from <https://icbinb-bio.github.io/submit/> and
<https://icbinb-bio.github.io/reviewer-guidelines/> on 17 Aug 2026.

| Item | Status |
|---|---|
| **Deadline** | **29 August 2026, 11:59 p.m. AoE** — marked *tentative*, unchanged from your draft. **12 days.** |
| **Page limit** | **Full papers: up to 8 pages**, excluding references and appendices. **Tiny papers: up to 4 pages** main text, same formatting. |
| **Template** | Workshop LaTeX template = NeurIPS 2026 style. Direct download (Google Drive zip, 11.6 KB, contains `neurips_2026.sty` + `neurips_2026.tex`): <https://drive.google.com/file/d/1YM_VG1bk6mokRHFwC69Z_nJnAJFvbHu0/view> — **I have extracted it to `/Users/vish/esp-lab/paper/template/`.** |
| **Correct invocation** | `\usepackage[dblblindworkshop]{neurips_2026}` and `\workshoptitle{I Can't Believe It's Not Better (ICBINB): Failure Modes of AI in Biology}`. Both `\title{}` and `\workshoptitle{}` are required. Camera-ready later becomes `[dblblindworkshop, final]`. For arXiv use `[preprint]`. |
| **Anonymity** | **Double-blind.** *"Submissions are double-blind; linked material must also preserve anonymity."* Code/data links must be anonymised. |
| **Archival** | **Non-archival.** *"Accepted papers will appear on OpenReview, and the workshop remains non-archival."* |
| **Portal** | OpenReview, **open now**: <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/ICBINB-BIO> |
| **Dual submission** | Allowed. *"Concurrent submissions are welcome when they comply with the target venue's policy."* Work under review at NeurIPS 2026 main track is explicitly welcome. |
| **Eligibility** | Not already accepted in previous conference proceedings. |
| **Appendices** | Unlimited, but reviewers are not required to read them. |
| **Extras not counted** | Ethics statement and reproducibility statement do not count toward the page limit. |
| **LLM disclosure** | **Required** — a short paragraph describing the role of any LLM used. |
| **Other dates** | Review 29 Aug – 21 Sep · Notification 29 Sep · Camera-ready & poster 20 Oct · Workshop 11 **or** 12 Dec 2026, Sydney (in person; exact date, venue and room TBA). |
| **Awards** | Entropic Award (most surprising negative result), Didactic Award (best-explained). Reviewers nominate. |
| **Organizers** | Brbić (EPFL), Koo (CSHL), Dumitrascu (Columbia), Lee (UW), Panigrahi (EPFL), Nagai (CSHL), Beker (Columbia), Gadgil (UW). |

**Still unstated by the organizers:** the final workshop date (11 vs 12 Dec); the venue/room;
whether tiny papers are reviewed on the same track; the program committee; and whether there is a
supplementary-material upload separate from the appendix. Everything on the site is labelled
*tentative* — re-check the CFP page in the last 48 hours before the deadline.

**Explicit evaluation criteria to write against** (from the CFP and reviewer guidelines): clarity
of problem and claims; technical rigour and reproducibility; **faithfulness to the biological
setting**; **depth of failure analysis**; quality of empirical documentation; novelty and
significance of the insights; quality of the limitations discussion. The reviewer guidelines also
say submissions that mainly report improved SOTA without failure analysis score *lower* on
workshop alignment — your paper is the opposite, which is good.

The CFP asks full papers to contain four named elements. Map your sections onto them explicitly,
using their words as headings or signposts:
1. **Problem** — behavioural-context/affect decoding from animal vocalisations with frozen encoders;
   target metrics; assumptions.
2. **Proposed approach** — frozen encoder + linear probe + random split, i.e. the field's default,
   stated as the thing under investigation.
3. **Observed outcome** — the +0.10–0.24 inflation, the below-chance published feature set, the
   background-only 0.725.
4. **Reason for failure** — identity and site decodability in the embedding; the CatMeows room
   confound; conformal per-group collapse.

---

## 12. The must-cite list

Fifteen references, grouped by theme. Omitting any of the six starred (★) items is a credible
rejection risk.

**A. The papers that establish your mechanism (cite in the intro, use their vocabulary)**
1. ★ **Chaibub Neto et al. 2019**, npj Digital Medicine 2:99 — coined **"identity confounding."**
2. ★ **Han et al. 2022**, npj Digital Medicine 5:16 — the random-vs-participant split gap measured
   in audio (AUC 0.71 participant-independent vs sens 0.84 / spec 0.78 random).
3. **Saeb et al. 2017**, GigaScience 6(5):gix019 — subject-wise vs record-wise CV.

**B. Bioacoustics prior art — the "we are not the first to say this" set**
4. ★ **Ghani, Baier, Kalkman & Stowell 2026**, PLOS Comput Biol 22(8):e1014604 — *"Avoid purely
   random dataset splits."* The 2026 best-practice guidance you are measuring.
5. ★ **Stowell, Petrusková, Šálek & Linhart 2019**, J R Soc Interface 16:20180940 — background
   confounds and adversarial background mixing. **Cite in Methods, not Related Work.**
6. **Colonna, Gama & Nakamura 2016**, CAEPIA/LNCS 9868 — k-fold-by-specimen; the decade-old
   precedent.
7. **Arnaud et al. 2023**, PLOS Comput Biol 19(4):e1010325 — leakage evaluation as a bioacoustics
   workflow step.
8. **Lefèvre, Sypherd & Briefer 2025**, iScience 28(2):111834 — the Briefer group already groups by
   individual; they do not group by lab. Your opening.

**C. The benchmarks and encoders you are auditing**
9. **Hagiwara et al. 2023**, BEANS, ICASSP — confirmed 6:2:2 stratified *random* split.
10. **Miron et al. 2026**, AVEX, ICLR 2026 — label-stratified random splits; probes identity but
    never as a confound.
11. **Nolasco et al. 2026**, Interspeech 2026 — eGeMAPS-probing of bioacoustic embeddings; your
    closest methodological sibling.

**D. Cross-domain convergent evidence (this audience's native literature)**
12. ★ **Lin, Wu & Jung 2026**, arXiv:2606.06647 — "The Identity Trap" in EEG foundation models.
13. ★ **Zare 2026**, arXiv:2607.24519 — classical features beat EEG foundation models; dataset
    identity decoded at 1.000.
14. ★ **DeGrave, Janizek & Lee 2021**, Nat Mach Intell 3:610–619 — shortcuts over signal.
    **Co-authored by an ICBINB-BIO organizer.**
15. **Geirhos et al. 2020** (Nat Mach Intell 2:665–673) and **Kapoor & Narayanan 2023**
    (Patterns 4:100804) — the shortcut-learning and leakage canon.

**Plus, if you keep the conformal section (you should):**
16. **Ding et al. 2023**, NeurIPS — the (1/α)−1 bound, CovGap, and the ImageNet demonstration.
17. **Foygel Barber et al. 2021**, Information and Inference 10(2):455–482 — why conditional
    coverage is impossible.
18. **Schwinger et al. 2026**, arXiv:2511.08261 — the nearest bioacoustics UQ work, and why yours
    is not redundant.

**And, if you keep the SER framing (a reviewer from speech will expect at least one):**
19. **Schuller et al. 2010**, IEEE TAC 1(2):119–131 — speaker-disjoint partitioning in SER.
20. **Wagner et al. 2023**, IEEE TPAMI — transformers are fair to gender but not to individual
    speakers.

---

## 13. Reframes required — exact wording

Six changes. The first three are mandatory.

**1. Claim 8c — the ≥9 threshold (MANDATORY).**
> *Before:* "repair requires ≥9 labelled clips from the deployment group."
> *After:* "repair requires at least (1/α)−1 = 9 labelled clips from the deployment group — the
> standard finite-sample floor below which the group-conditional quantile is infinite (Ding et al.,
> 2023). The contribution is not the bound but its ecological cost: nine labelled clips per
> individual animal, each requiring simultaneous behavioural observation, is a burden no
> deployment protocol currently budgets for."

**2. Claim 6 — the background-only control (MANDATORY).**
> *Before:* "we introduce a background-only control."
> *After:* "we apply the background-confound diagnostic of Stowell et al. (2019) — who used
> normally-discarded silent segments to expose environmental confounding in individual
> identification — to **behavioural-context** labels, and find that on CatMeows the background
> alone reaches 0.725 on the binary task. The dataset's own protocol explains why: its negative
> class was induced by moving the cat to a different room."

**3. Claim 2 — the inflation result (MANDATORY).**
> *Before:* "we show random splits inflate reported accuracy."
> *After:* "identity confounding (Chaibub Neto et al., 2019) is documented in clinical audio, where
> participant-independent evaluation costs ≈0.13 AUC (Han et al., 2022), and bioacoustics guidance
> has warned against random splits for a decade (Colonna et al., 2016; Ghani et al., 2026). We
> measure what that warning is worth in the frozen-encoder era: +0.10 to +0.24 across four corpora
> and six feature sets — and we show that the benchmarks the field currently runs on (BEANS, AVEX)
> both use random splits."

**4. Claim 4 — the eGeMAPS result (strongly recommended).**
Soften "beats" to "matches, while leaking substantially less identity," unless you can produce a
paired test across the five labs. A 0.9-point gap will not survive a reviewer. And add one
paragraph engaging Tang et al. (2023), who report the opposite in cross-language SER.

**5. Claim 3 — the cross-corpus null (strongly recommended).**
Replace "does not hold" with "we do not detect… the test is severely underpowered at n=12."

**6. Claim 7 — channel robustness (recommended).**
Add one sentence distinguishing artificial channel degradation of recorded datasets from natural
propagation degradation of evolved signatures, citing Mouterde et al. (2014) as the contrast.

**One thing to add rather than change:** the framing sentence that ties the paper together for this
audience. Something like — *"Frozen foundation-model benchmarks inherit a nuisance axis from their
data. This has now been shown for EEG (Lin et al., 2026; Zare, 2026), for clinical audio (Chaibub
Neto et al., 2019; Han et al., 2022) and for radiographs (DeGrave et al., 2021). We show it for
animal vocalisations, where the nuisance axis — individual identity — is also, awkwardly, a
genuine biological signal."* That last clause is the thing no other domain has, and it is the most
interesting sentence available to you.

---

## 14. Reference list (paste-ready)

Alphabetical. Verified against OpenAlex / Europe PMC / arXiv unless marked *unverified*.

Abzaliev, A., Pérez-Espinosa, H., & Mihalcea, R. (2024). Towards dog bark decoding: Leveraging
human speech processing for automated bark classification. *LREC-COLING 2024.* arXiv:2404.18739.

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations
and Trends in Machine Learning, 16*(4), 494–591. arXiv:2107.07511.

Arnaud, V., Pellegrino, F., Keenan, S., St-Gelais, X., Mathevon, N., Levréro, F., & Coupé, C.
(2023). Improving the workflow to crack small, unbalanced, noisy, but genuine (SUNG) datasets in
bioacoustics: The case of bonobo calls. *PLOS Computational Biology, 19*(4), e1010325.

Baek, C., Jiang, Y., Raghunathan, A., & Kolter, J. Z. (2022). Agreement-on-the-line: Predicting the
performance of neural networks under distribution shift. *NeurIPS 2022.* arXiv:2206.13089.

Beery, S., Van Horn, G., & Perona, P. (2018). Recognition in Terra Incognita. *ECCV 2018.*
arXiv:1807.04975.

Briefer, E. F., Sypherd, C. C.-R., Linhart, P., Leliveld, L. M. C., Padilla de la Torre, M., Read,
E., et al. (2022). Classification of pig calls produced from birth to slaughter according to their
emotional valence and context of production. *Scientific Reports, 12*, 3409.
(Author Correction: *Scientific Reports*, 2023, doi:10.1038/s41598-023-45242-9.)

Chaibub Neto, E., Pratap, A., Perumal, T. M., Tummalacherla, M., Snyder, P., Bot, B. M., et al.
(2019). Detecting the impact of subject characteristics on machine learning-based diagnostic
applications. *npj Digital Medicine, 2*, 99.

Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., et al. (2022). WavLM: Large-scale
self-supervised pre-training for full stack speech processing. *IEEE Journal of Selected Topics in
Signal Processing, 16*(6), 1505–1518. arXiv:2110.13900.

Colonna, J. G., Gama, J., & Nakamura, E. F. (2016). How to correctly evaluate an automatic
bioacoustics classification method. *CAEPIA 2016*, LNCS 9868, 37–47. Springer.

Coppock, H., Nicholson, G., Kiskin, I., Koutra, V., Baker, K., Budd, J., et al. (2024). Audio-based
AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers.
*Nature Machine Intelligence, 6*(2), 229–242.

Cresswell, J. C., Kumar, B., Sui, Y., & Belbahri, M. (2025). Conformal prediction sets can cause
disparate impact. *ICLR 2025 (Spotlight).* arXiv:2410.01888.

DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects
shortcuts over signal. *Nature Machine Intelligence, 3*, 610–619.

Dehak, N., Kenny, P. J., Dehak, R., Dumouchel, P., & Ouellet, P. (2011). Front-end factor analysis
for speaker verification. *IEEE TASLP, 19*(4), 788–798.

Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I., & Tibshirani, R. J. (2023).
Class-conditional conformal prediction with many classes. *NeurIPS 2023.* arXiv:2306.09335.

Dumpala, S. H., Dikaios, K., Rodriguez, S., Langley, R., Rempel, S., Uher, R., et al. (2023).
Manifestation of depression in speech overlaps with characteristics used to represent and recognize
speaker identity. *Scientific Reports, 13*, 6141.

Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., et al. (2016). The
Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing.
*IEEE Transactions on Affective Computing, 7*(2), 190–202.

Eyben, F., Wöllmer, M., & Schuller, B. (2010). openSMILE: The Munich versatile and fast open-source
audio feature extractor. *ACM Multimedia 2010*, 1459–1462.

Fisch, A., Schuster, T., Jaakkola, T., & Barzilay, R. (2021). Few-shot conformal prediction with
auxiliary tasks. *ICML 2021*, PMLR 139, 3329–3339. arXiv:2102.08898.

Foygel Barber, R., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The limits of
distribution-free conditional predictive inference. *Information and Inference, 10*(2), 455–482.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A.
(2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*, 665–673.

Ghani, B., Baier, A. L., Kalkman, V. J., & Stowell, D. (2026). Twelve quick tips for applying deep
learning to animal sounds. *PLOS Computational Biology, 22*(8), e1014604.

Gibbs, I., Cherian, J. J., & Candès, E. J. (2025). Conformal prediction with conditional
guarantees. *Journal of the Royal Statistical Society: Series B, 87*(4), 1100–1126.
arXiv:2305.12616.

Hagiwara, M. (2023). AVES: Animal vocalization encoder based on self-supervision. *ICASSP 2023.*
arXiv:2210.14493.

Hagiwara, M., Hoffman, B., Liu, J.-Y., Cusimano, M., Effenberger, F., & Zacarian, K. (2023). BEANS:
The benchmark of animal sounds. *ICASSP 2023.* arXiv:2210.12300.

Han, J., Xia, T., Spathis, D., Bondareva, E., Brown, C., Chauhan, J., et al. (2022). Sounds of
COVID-19: Exploring realistic performance of audio-based digital testing. *npj Digital Medicine,
5*, 16.

Hummel, H. I., Bhulai, S., van der Mei, R. D., & Ghani, B. (2026). Decodable but not structured:
Linear probing enables underwater acoustic target recognition with pretrained audio embeddings.
arXiv:2601.08358.

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-
based science. *Patterns, 4*(9), 100804.

Lefèvre, R. A., Sypherd, C. C. R., & Briefer, E. F. (2025). Machine learning algorithms can predict
emotional valence across ungulate vocalizations. *iScience, 28*(2), 111834.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free
predictive inference for regression. *JASA, 113*(523), 1094–1111.

Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric regression.
*JRSS-B, 76*(1), 71–96.

Lin, J., Wu, Y.-C., & Jung, T.-P. (2026). The identity trap in EEG foundation models: A diagnostic
audit. arXiv:2606.06647.

Lostanlen, V., Salamon, J., Farnsworth, A., Kelling, S., & Bello, J. P. (2019). Robust sound event
detection in bioacoustic sensor networks. *PLOS ONE, 14*(10), e0214168.

Ludovico, L. A., Ntalampiras, S., Presti, G., Cannas, S., Battini, M., & Mattiello, S. (2021).
CatMeows: A publicly-available dataset of cat vocalizations. *MultiMedia Modeling (MMM 2021)*,
LNCS 12573, 230–243. Springer.

Miller, J. P., Taori, R., Raghunathan, A., Sagawa, S., Koh, P. W., Shankar, V., et al. (2021).
Accuracy on the line: On the strong correlation between out-of-distribution and in-distribution
generalization. *ICML 2021*, PMLR 139. arXiv:2107.04649.

Miron, M., Robinson, D., Alizadeh, M., et al. (2026). AVEX: What matters for animal vocalization
encoding. *ICLR 2026.* arXiv:2508.11845.

Molnár, C., Kaplan, F., Roy, P., Pachet, F., Pongrácz, P., Dóka, A., & Miklósi, Á. (2008).
Classification of dog barks: A machine learning approach. *Animal Cognition, 11*, 389–400.

Mouterde, S. C., Theunissen, F. E., Elie, J. E., Vignal, C., & Mathevon, N. (2014). Acoustic
communication and sound degradation: How do the individual signatures of male and female zebra
finch calls transmit over distance? *PLOS ONE, 9*(7), e102842.

Müller, N. M., Dieckmann, F., Czempin, P., Canals, R., Böttinger, K., & Williams, J. (2021). Speech
is silver, silence is golden: What do ASVspoof-trained models really learn? *ASVspoof 2021
Workshop.* arXiv:2106.12914.

Nolasco, I., Cauzinille, J., Miron, M., Narula, G., Alizadeh, M., et al. (2026). Beyond task
performance: Decoding bioacoustic embeddings with speech features. *Interspeech 2026.*
arXiv:2606.14662.

Ntalampiras, S., Ludovico, L. A., Presti, G., Prato Previde, E., Battini, M., Cannas, S., et al.
(2019). Automatic classification of cat vocalizations emitted in different contexts. *Animals,
9*(8), 543.

Osiecka, A. N., Lefèvre, R. A., & Briefer, E. F. (2025). Emotional contexts influence vocal
individuality in ungulates. *Animal Behaviour.* (Preprint: bioRxiv 2024.09.18.613506.)

Pascu, O., Oneață, D., Cucu, H., & Müller, N. M. (2025). Easy, interpretable, effective: openSMILE
for voice deepfake detection. *ICASSP 2025.* arXiv:2408.15775.

Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., et al. (2020).
Spatial validation reveals poor predictive performance of large-scale ecological mapping models.
*Nature Communications, 11*, 4540.

Prat, Y., Taub, M., Pratt, E., & Yovel, Y. (2017). An annotated dataset of Egyptian fruit bat
vocalizations across varying contexts and during vocal ontogeny. *Scientific Data, 4*, 170143.

Prat, Y., Taub, M., & Yovel, Y. (2016). Everyday bat vocalizations contain information about
emitter, addressee, context, and behavior. *Scientific Reports, 6*, 39419.

Rauch, L., Schwinger, R., Wirth, M., Heinrich, R., Huseljic, D., Herde, M., et al. (2025). BirdSet:
A large-scale dataset for audio classification in avian bioacoustics. *ICLR 2025.* arXiv:2403.10380.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017).
Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure. *Ecography, 40*(8), 913–929.

Roberts, M., Driggs, D., Thorpe, M., Gilbey, J., Yeung, M., Ursprung, S., et al. (2021). Common
pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19
using chest radiographs and CT scans. *Nature Machine Intelligence, 3*, 199–217.

Romano, Y., Barber, R. F., Sabatti, C., & Candès, E. J. (2020). With malice toward none: Assessing
uncertainty via equalized coverage. *Harvard Data Science Review, 2*(2).

Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with valid and adaptive coverage.
*NeurIPS 33*, 3581–3591. arXiv:2006.02544.

Rybka, J., & Janicki, A. (2013). Comparison of speaker dependent and speaker independent emotion
recognition. *International Journal of Applied Mathematics and Computer Science, 23*(4), 797–808.

Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Körding, K. P. (2017). The need to approximate
the use-case in clinical machine learning. *GigaScience, 6*(5), gix019.

Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded
error levels. *JASA, 114*(525), 223–234.

Schuller, B., Steidl, S., & Batliner, A. (2009). The INTERSPEECH 2009 Emotion Challenge.
*Interspeech 2009*, 312–315.

Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G.
(2010). Cross-corpus acoustic emotion recognition: Variances and strategies. *IEEE Transactions on
Affective Computing, 1*(2), 119–131.

Schwinger, R., McEwen, ..., Rauch, L., & Tomforde, S. (2026). Uncertainty calibration of
multi-label bird sound classifiers. *ICAART 2026.* arXiv:2511.08261. *(author list incomplete —
verify)*

Schwinger, R., Vali Zadeh, P., Rauch, L., Kurz, M., Hauschild, T., Lapp, S., & Tomforde, S. (2026).
Foundation models for bioacoustics: A comparative review. *Ecological Informatics.*
arXiv:2508.01277.

Stowell, D., Petrusková, T., Šálek, M., & Linhart, P. (2019). Automatic acoustic identification of
individuals in multiple species: Improving identification across recording conditions. *Journal of
the Royal Society Interface, 16*(153), 20180940.

Tang, D., Kuppens, P., Geurts, L., & van Waterschoot, T. (2023). End-to-end transfer learning for
speaker-independent cross-language and cross-corpus speech emotion recognition. arXiv:2311.13678.

Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019). Conformal prediction
under covariate shift. *NeurIPS 32*, 2526–2536.

Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML 2012*, PMLR 25,
475–490.

Vovk, V., Gammerman, A., & Shafer, G. (2022). *Algorithmic learning in a random world* (2nd ed.).
Springer.

Wagner, J., Schiller, D., Seiderer, A., & André, E. (2018). Deep learning in paralinguistic
recognition tasks: Are hand-crafted features still relevant? *Interspeech 2018*, 147–151.

Wagner, J., Triantafyllopoulos, A., Wierstorf, H., Schmitt, M., Burkhardt, F., Eyben, F., &
Schuller, B. W. (2023). Dawn of the transformer era in speech emotion recognition: Closing the
valence gap. *IEEE TPAMI, 45*(9), 10745–10759. arXiv:2203.07378.

Whalen, S., Schreiber, J., Noble, W. S., & Pollard, K. S. (2022). Navigating the pitfalls of
applying machine learning in genomics. *Nature Reviews Genetics, 23*, 169–181.

Wierucka, K., Murphy, D., Watson, S. K., Falk, N., Fichtel, C., León, J., Leu, S. T., Kappeler,
P. M., et al. (2025). Same data, different results? Machine learning approaches in bioacoustics.
*Methods in Ecology and Evolution, 16*, 1574–1586.

Wilson, O., Schoeman, D. S., Bradley, A. P., & Clemente, C. J. (2025). Practical guidelines for
validation of supervised machine learning models in accelerometer-based animal behaviour
classification. *Journal of Animal Ecology.* doi:10.1111/1365-2656.70054.

Xiao, K., Engstrom, L., Ilyas, A., & Madry, A. (2021). Noise or signal: The role of image
backgrounds in object recognition. *ICLR 2021.* arXiv:2006.09994.

Yang, S.-w., Chi, P.-H., Chuang, Y.-S., Lai, C.-I. J., Lakhotia, K., et al. (2021). SUPERB: Speech
processing universal PERformance benchmark. *Interspeech 2021*, 3161–3165. arXiv:2105.01051.

Yeh, S.-L., Sun, ..., Mower Provost, E., & Sisman, B. (2026). Who is speaking or who is depressed?
A controlled study of speaker leakage in speech-based depression detection. *Interspeech 2026.*
arXiv:2604.14354. *(author list incomplete — verify)*

Zare, M. (2026). A negative-control protocol for clinical EEG foundation-model benchmarks: Dataset
identity and external-cohort stress testing. arXiv:2607.24519.

Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018).
Variable generalization performance of a deep learning model to detect pneumonia in chest
radiographs: A cross-sectional study. *PLOS Medicine, 15*(11), e1002683.

Zhong, L., Wang, X., Huang, S., & Shi, Y. (2026). When is a conformal guarantee fair? Auditing
silent subgroup under-coverage in Alzheimer's disease longitudinal prediction. arXiv:2608.04254.

---

## 15. Things to verify before submitting

- [ ] **Read Zhong et al. 2026 (arXiv:2608.04254) in full** — 13 days old, closest framing threat
      to Claim 8.
- [ ] **Read Lin et al. 2026 (arXiv:2606.06647) and Zare 2026 (arXiv:2607.24519) in full** — decide
      whether to frame them as convergent evidence (recommended) or as related work.
- [ ] Verify the Müller et al. 2021 numbers (~85% from silence duration, 3.6%→15.5% EER) from the
      PDF before quoting.
- [ ] Verify author lists flagged *incomplete* above: Schwinger et al. 2026 (ICAART), Yeh et al.
      2026.
- [ ] Get the Colonna et al. 2016 chapter itself (paywalled) for its exact numbers — a reviewer who
      knows it will expect you to cite the magnitude.
- [ ] Run a paired test across the five pig labs for the eGeMAPS-vs-WavLM comparison, or soften the
      wording (see §5).
- [ ] Re-check <https://icbinb-bio.github.io/submit/> in the last 48 hours — everything is marked
      tentative.
- [ ] Anonymise any code/data links before submitting (double-blind extends to linked material).
- [ ] Write the LLM-use disclosure paragraph — it is required.
