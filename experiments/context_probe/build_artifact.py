import json, os
D = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(D, "out")
b64 = {k: open(os.path.join(O, k + ".b64")).read() for k in ["pca_context.png", "confusion_loco.png", "per_cat.png"]}
R = json.load(open(os.path.join(O, "results.json")))["results"]
loco = R["loco_logreg"]["bal_acc"]; loco_knn = R["loco_knn5"]["bal_acc"]
rnd = R["random_logreg"]["bal_acc"]; rnd_knn = R["random_knn5"]["bal_acc"]
ident = R["identity_acc"]["acc"]
pc = R["per_cat_acc"]; n_above = sum(v > 1/3 for v in pc.values())

html = f"""<title>The Context Probe</title>
<style>
:root{{--bg:#EFF2F3;--surface:#FBFCFC;--surface-2:#E4EAEB;--surface-3:#DAE2E3;--ink:#0F1A1C;--ink-2:#42565A;--ink-3:#6C8084;--rule:#CBD6D8;--rule-soft:#DDE5E6;--accent:#9A5205;--accent-bright:#C87A12;--cyan:#0A626D;--mag:#8A3459;--ok:#2C6349;--ok-bg:#DCEBE3;--warn:#7E5A0C;--warn-bg:#F0E7D0;--bad:#8A3459;--bad-bg:#F2DEE7;--dim:#5C6E71;--dim-bg:#E1E7E8;--shadow:0 1px 2px rgba(15,26,28,.06),0 8px 24px -16px rgba(15,26,28,.28);--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,"Cascadia Mono","Roboto Mono",monospace;--serif:Charter,"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;--measure:70ch;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0A1113;--surface:#111A1D;--surface-2:#162226;--surface-3:#1D2C30;--ink:#E7EEEF;--ink-2:#A5B7BA;--ink-3:#78898D;--rule:#243337;--rule-soft:#1B282B;--accent:#F0AA3E;--accent-bright:#FFC463;--cyan:#4FC7CE;--mag:#E58BB0;--ok:#6ECB99;--ok-bg:#163024;--warn:#E0B357;--warn-bg:#2E2617;--bad:#E58BB0;--bad-bg:#33202A;--dim:#8B9C9F;--dim-bg:#1A2427;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0A1113;--surface:#111A1D;--surface-2:#162226;--surface-3:#1D2C30;--ink:#E7EEEF;--ink-2:#A5B7BA;--ink-3:#78898D;--rule:#243337;--rule-soft:#1B282B;--accent:#F0AA3E;--accent-bright:#FFC463;--cyan:#4FC7CE;--mag:#E58BB0;--ok:#6ECB99;--ok-bg:#163024;--warn:#E0B357;--warn-bg:#2E2617;--bad:#E58BB0;--bad-bg:#33202A;--dim:#8B9C9F;--dim-bg:#1A2427;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 clamp(16px,4vw,40px)}}.col{{max-width:var(--measure)}}
h1,h2,h3,h4{{text-wrap:balance;margin:0;line-height:1.16;font-weight:600}}h1{{font-size:clamp(2.3rem,6vw,3.9rem);letter-spacing:-.022em;line-height:1.02}}h2{{font-size:clamp(1.45rem,3vw,1.95rem);letter-spacing:-.014em}}h3{{font-size:1.1rem}}h4{{font-size:.95rem}}p{{margin:0}}
a{{color:var(--cyan);text-underline-offset:3px;text-decoration-thickness:1px}}a:hover{{color:var(--accent)}}:focus-visible{{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}}
.eyebrow{{font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);font-weight:500}}
.lede{{font-size:clamp(1.05rem,2vw,1.22rem);color:var(--ink-2);line-height:1.55}}.small{{font-size:.9rem;color:var(--ink-2)}}
code{{font-family:var(--mono);font-size:.845em;background:var(--surface-2);padding:.12em .38em;border-radius:4px;border:1px solid var(--rule-soft)}}
pre{{font-family:var(--mono);font-size:.8rem;line-height:1.7;background:var(--surface);border:1px solid var(--rule);border-left:2px solid var(--accent);padding:14px 16px;border-radius:6px;overflow-x:auto;margin:0}}pre code{{background:none;border:none;padding:0;font-size:inherit}}
header.hero{{border-bottom:1px solid var(--rule);background:var(--surface);padding:clamp(34px,6vw,64px) 0 clamp(28px,4vw,44px)}}header.hero h1{{margin:14px 0 18px}}.accent-word{{color:var(--accent)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:8px;overflow:hidden;margin-top:30px}}.stat{{background:var(--surface);padding:13px 14px}}.stat b{{display:block;font-family:var(--mono);font-size:1.36rem;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}.stat span{{font-family:var(--mono);font-size:.63rem;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3)}}.stat.hi b{{color:var(--accent)}}.stat.bad b{{color:var(--mag)}}
nav.jump{{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--rule)}}nav.jump ul{{display:flex;list-style:none;margin:0;padding:0;overflow-x:auto;scrollbar-width:thin}}nav.jump a{{display:block;white-space:nowrap;padding:11px 14px;font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);text-decoration:none;border-bottom:2px solid transparent}}nav.jump a:hover{{color:var(--ink);border-bottom-color:var(--accent)}}
section{{padding:clamp(44px,7vw,80px) 0;border-bottom:1px solid var(--rule-soft)}}section:last-of-type{{border-bottom:none}}.sec-head{{display:flex;flex-direction:column;gap:9px;margin-bottom:28px}}.sec-head p{{max-width:var(--measure)}}.stack{{display:flex;flex-direction:column;gap:22px}}
.chip{{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);font-size:.66rem;letter-spacing:.05em;text-transform:uppercase;font-weight:500;padding:3px 8px;border-radius:999px;white-space:nowrap}}.chip::before{{content:"";width:5px;height:5px;border-radius:50%;background:currentColor}}.ch-ok{{color:var(--ok);background:var(--ok-bg)}}.ch-warn{{color:var(--warn);background:var(--warn-bg)}}.ch-bad{{color:var(--bad);background:var(--bad-bg)}}.ch-dim{{color:var(--dim);background:var(--dim-bg)}}
.tscroll{{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}}table{{border-collapse:collapse;width:100%;font-size:.87rem;min-width:640px}}th,td{{text-align:left;padding:10px 13px;border-bottom:1px solid var(--rule-soft);vertical-align:top}}thead th{{font-family:var(--mono);font-size:.64rem;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);font-weight:500;background:var(--surface-2);border-bottom:1px solid var(--rule)}}tbody tr:last-child td{{border-bottom:none}}td.name{{font-family:var(--mono);font-size:.82rem;white-space:nowrap}}td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}.t-note{{font-size:.85rem;color:var(--ink-2);line-height:1.5}}
.grid{{display:grid;gap:16px}}.g2{{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}.g3{{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}}
.card{{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:18px 19px;display:flex;flex-direction:column;gap:9px;box-shadow:var(--shadow)}}.card p{{font-size:.9rem;color:var(--ink-2);line-height:1.55}}.card .tag{{font-family:var(--mono);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}}.card .meta{{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;padding-top:4px}}
figure{{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:8px}}figure img{{width:100%;height:auto;border-radius:4px;display:block}}figcaption{{font-size:.85rem;color:var(--ink-2);line-height:1.5}}figcaption b{{color:var(--ink)}}
.note{{border:1px solid var(--rule);border-left:2px solid var(--accent);background:var(--surface);border-radius:6px;padding:15px 17px;display:flex;flex-direction:column;gap:7px}}.note h4{{font-family:var(--mono);font-size:.7rem;letter-spacing:.11em;text-transform:uppercase;color:var(--accent)}}.note p,.note li{{font-size:.9rem;color:var(--ink-2);line-height:1.55}}.note ul{{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:6px}}.note.bad{{border-left-color:var(--mag)}}.note.bad h4{{color:var(--mag)}}
.diagram{{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:18px;overflow-x:auto}}.diagram svg{{display:block;min-width:620px;width:100%;height:auto}}
ol.phases{{list-style:none;counter-reset:ph;margin:0;padding:0}}ol.phases li{{counter-increment:ph;display:grid;grid-template-columns:auto 1fr;gap:18px;padding:18px 0;border-bottom:1px solid var(--rule-soft)}}ol.phases li:last-child{{border-bottom:none}}ol.phases li::before{{content:counter(ph);font-family:var(--mono);font-size:.78rem;font-weight:600;color:var(--accent);border:1px solid var(--rule);background:var(--surface);width:34px;height:34px;display:grid;place-items:center;border-radius:50%}}.ph-body{{display:flex;flex-direction:column;gap:7px}}.ph-body h3{{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px}}.ph-when{{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);font-weight:400}}.ph-body p{{font-size:.94rem;color:var(--ink-2)}}
ul.plain{{margin:0;padding-left:19px;display:flex;flex-direction:column;gap:8px}}ul.plain li{{font-size:.93rem;color:var(--ink-2)}}ul.plain li b{{color:var(--ink)}}
footer{{padding:34px 0 56px;color:var(--ink-3);font-size:.84rem}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}@media (max-width:640px){{body{{font-size:16px}}ol.phases li{{grid-template-columns:1fr;gap:10px}}}}
</style>

<header class="hero"><div class="wrap">
<p class="eyebrow">Experiment 1 &middot; run 17 Aug 2026 &middot; laptop only, no GPU</p>
<h1>The Context <span class="accent-word">Probe</span></h1>
<p class="lede col">You asked whether we could tell &ldquo;food&rdquo; from &ldquo;danger&rdquo; from the sound alone. Tonight we tried &mdash; on 440 cat meows, three contexts, 21 cats, with nothing but the laptop. The answer is more useful than yes or no: it tells us exactly what a real encoder has to beat, and it caught the field's favourite mistake red-handed.</p>
<div class="stats">
<div class="stat"><b>440</b><span>meows</span></div>
<div class="stat"><b>21</b><span>cats</span></div>
<div class="stat"><b>3</b><span>contexts</span></div>
<div class="stat hi"><b>{loco:.2f}</b><span>honest accuracy</span></div>
<div class="stat"><b>{rnd:.2f}</b><span>leaky accuracy</span></div>
<div class="stat bad"><b>{ident:.2f}</b><span>identity accuracy</span></div>
</div></div></header>

<nav class="jump" aria-label="Sections"><div class="wrap"><ul>
<li><a href="#reframe">The reframe</a></li><li><a href="#ran">What we ran</a></li><li><a href="#found">What it says</a></li><li><a href="#trap">The trap</a></li><li><a href="#next">Experiments</a></li><li><a href="#data">Datasets</a></li><li><a href="#cross">Cross-species</a></li><li><a href="#compute">Compute</a></li><li><a href="#rules">Rules</a></li>
</ul></div></nav>

<main>
<section id="reframe"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Orientation</p><h2>&ldquo;Words&rdquo; are called contexts here, and the datasets exist</h2>
<p class="lede">What you described as a word dataset is real. The field calls it a <b>behavioral-context dataset</b>: each recording is labelled with the situation the animal was in &mdash; food present, predator present, isolated, playing, fighting, begging. That is exactly &ldquo;food&rdquo; and &ldquo;danger.&rdquo; It is how the classic science was done: vervet alarm calls, meerkat alarms, chicken food calls, dog barks in six situations, cat meows before feeding versus alone.</p></div>
<div class="grid g3">
<div class="card"><span class="tag">Keep</span><h3>Context</h3><p>&ldquo;A food call is a call recorded when food was present.&rdquo; Exactly as strong as the data, and every biologist will accept it.</p></div>
<div class="card"><span class="tag">Drop</span><h3>Meaning</h3><p>&ldquo;The cat is saying it wants food.&rdquo; Not testable from audio alone. The word that makes researchers stop reading.</p></div>
<div class="card"><span class="tag">The real question</span><h3>Predictability</h3><p>Can the context be predicted from the sound, for an animal the model never heard? That is a clean ML claim, and it is what we measured.</p></div>
</div></div></section>

<section id="ran"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Tonight</p><h2>What we actually ran</h2>
<p class="lede">The floor first. No deep model &mdash; hand-crafted spectral features and a linear classifier. If a frozen encoder can't beat this, it isn't earning its keep.</p></div>
<div class="grid g2">
<div class="card"><span class="tag">Dataset &middot; verified</span><h3>CatMeows</h3><p>Ludovico, Ntalampiras et al. 2021. 440 meows from 21 cats (Maine Coon, European Shorthair), each recorded in one of three induced situations: <b>brushing</b> at home, <b>waiting for food</b>, and <b>isolation</b> in an unfamiliar room. Filenames encode context, cat, breed, sex, owner and session &mdash; so we can split by animal.</p><div class="meta"><span class="chip ch-ok">Zenodo 4008297</span><span class="chip ch-ok">CC-BY-4.0</span><span class="chip ch-dim">8.9 MB</span></div></div>
<div class="card"><span class="tag">Embedding &middot; baseline</span><h3>85-d clip vector</h3><p>16 kHz mono &rarr; 40-band log-mel spectrogram (25 ms / 10 ms) &rarr; per clip: mean and std of 20 MFCCs, mean log-mel per band, energy, spectral centroid, log duration. Pure numpy/scipy. This is the &ldquo;embedding&rdquo; for tonight; the deep encoders slot in at the same point.</p><div class="meta"><span class="chip ch-ok">runs on stock python</span><span class="chip ch-dim">~2 s total</span></div></div>
</div>

<div class="diagram"><svg viewBox="0 0 900 250" role="img" aria-label="Pipeline: audio to log-mel to clip vector to standardised probe, split by cat">
<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6.4" refY="3.2" orient="auto"><path d="M0,0 L6.4,3.2 L0,6.4 z" fill="var(--ink-3)"/></marker></defs>
<g font-family="var(--mono)" font-size="11.5" fill="var(--ink)">
<rect x="14" y="60" width="150" height="70" rx="7" fill="var(--surface-2)" stroke="var(--rule)"/><text x="30" y="86">440 .wav</text><text x="30" y="104" font-size="10" fill="var(--ink-3)">B_ANI01_MC_FN_SIM01_101</text><text x="30" y="118" font-size="10" fill="var(--ink-3)">ctx · cat · breed · session</text>
<rect x="196" y="60" width="140" height="70" rx="7" fill="var(--surface-2)" stroke="var(--rule)"/><text x="212" y="86">log-mel</text><text x="212" y="104" font-size="10" fill="var(--ink-3)">40 bands × T frames</text><text x="212" y="118" font-size="10" fill="var(--ink-3)">25 ms / 10 ms</text>
<rect x="368" y="60" width="150" height="70" rx="7" fill="var(--surface-3)" stroke="var(--cyan)" stroke-width="1.5"/><text x="384" y="86">clip vector</text><text x="384" y="104" font-size="10" fill="var(--ink-3)">85-d (MFCC stats…)</text><text x="384" y="118" font-size="10" fill="var(--cyan)">← encoders plug in here</text>
<rect x="550" y="60" width="150" height="70" rx="7" fill="var(--surface-2)" stroke="var(--rule)"/><text x="566" y="86">standardise</text><text x="566" y="104" font-size="10" fill="var(--ink-3)">fit on train fold only</text><text x="566" y="118" font-size="10" fill="var(--ink-3)">z-score</text>
<rect x="732" y="60" width="150" height="70" rx="7" fill="var(--surface-3)" stroke="var(--accent)" stroke-width="1.5"/><text x="748" y="86">probe</text><text x="748" y="104" font-size="10" fill="var(--ink-3)">logistic regression</text><text x="748" y="118" font-size="10" fill="var(--ink-3)">+ kNN(5) for contrast</text>
</g>
<g stroke="var(--ink-3)" stroke-width="1.3" marker-end="url(#ar)" fill="none"><path d="M164,95 L192,95"/><path d="M336,95 L364,95"/><path d="M518,95 L546,95"/><path d="M700,95 L728,95"/></g>
<rect x="14" y="160" width="868" height="74" rx="7" fill="var(--surface)" stroke="var(--mag)" stroke-dasharray="4 3"/>
<text x="30" y="184" font-family="var(--mono)" font-size="10.5" letter-spacing="1.6" fill="var(--mag)">THE SPLIT — the most important line in the file</text>
<text x="30" y="206" font-family="var(--serif)" font-size="13" fill="var(--ink)">Leave-one-cat-out: all meows from a cat are either train or test. Never random.</text>
<text x="30" y="224" font-family="var(--serif)" font-size="11.5" fill="var(--ink-3)">Random splits let the classifier learn which cat instead of which context. We ran both, on purpose, to measure the gap.</text>
</svg></div>

<div class="grid g2">
<figure><img src="data:image/png;base64,{b64['pca_context.png']}" alt="PCA of 85-d clip vectors coloured by context"><figcaption><b>The embedding space, first two principal components.</b> Isolation meows (magenta) drift right along PC1; brushing and food overlap heavily. There is structure, but not a clean three-way separation &mdash; and PC1 carries 45% of the variance, which in audio features usually means loudness and length as much as anything else.</figcaption></figure>
<figure><img src="data:image/png;base64,{b64['confusion_loco.png']}" alt="Confusion matrix, leave-one-cat-out"><figcaption><b>Held-out cats, logistic regression.</b> Brushing and isolation are recovered at 0.44 each. &ldquo;Waiting for food&rdquo; is at 0.29 &mdash; <em>below</em> chance: those meows are the least distinctive to these features, and get filed as isolation. The one axis with any signal is calm-at-home versus alone-in-a-strange-room, which is an arousal axis, not a lexical one.</figcaption></figure>
</div>
<figure><img src="data:image/png;base64,{b64['per_cat.png']}" alt="Per-cat held-out accuracy"><figcaption><b>Does it generalise to a cat it never heard?</b> Sometimes. {n_above} of 21 cats land above chance; the spread runs from 0.00 to 0.86. A number this variable across individuals is the signature of a model that has learned a few animals' voices rather than a shared code &mdash; which is the finding.</figcaption></figure>
</div></section>

<section id="found"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Reading it</p><h2>What the numbers say</h2></div>
<div class="grid g3">
<div class="card"><span class="tag">Honest</span><h3>{loco:.2f} balanced accuracy</h3><p>Leave-one-cat-out, chance is 0.33. Contexts are <em>barely</em> predictable from hand-crafted features once the classifier can't recognise the cat. kNN sits at {loco_knn:.2f} &mdash; at chance.</p></div>
<div class="card"><span class="tag">Leaky</span><h3>{rnd:.2f} &ndash; {rnd_knn:.2f}</h3><p>Same features, same classifiers, random 5-fold split. The number nearly doubles. Nothing about the sound changed; the classifier just met each cat during training. This is what most published context-classification numbers look like.</p></div>
<div class="card"><span class="tag">Why</span><h3>{ident:.2f} identity accuracy</h3><p>Ask the same 85 features &ldquo;which cat is this?&rdquo; and they answer correctly {ident*100:.0f}% of the time across 21 cats (chance 5%). The sound is dominated by <em>who</em>, not <em>why</em>. Any random split rides that.</p></div>
</div>
<div class="note"><h4>The bar is now set</h4><p>Whatever frozen encoder we run next &mdash; ESP's AVEX BEATs, BirdAVES, WavLM, Perch&nbsp;2 &mdash; has one job on this dataset: <b>beat {loco:.2f} under leave-one-cat-out.</b> If it can't, it doesn't hear context better than a spectrogram summary. If it does, we know precisely how much of context is in the sound versus in the speaker. Either way we learn something a random-split number could never tell us.</p></div>
</div></section>

<section id="trap"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Hard-won</p><h2>The trap, caught live</h2></div>
<div class="note bad"><h4>Identity leakage</h4><ul>
<li>In context datasets, all the &ldquo;food&rdquo; recordings tend to come from the same sessions, rooms and animals. A classifier will learn the room and the animal and report it as context.</li>
<li>Tonight that inflated accuracy from {loco:.2f} to {rnd:.2f}&ndash;{rnd_knn:.2f} with no change to the model. Published papers on this dataset report far higher accuracies; before believing any of them, check the split.</li>
<li>Rule for everything downstream: <b>split by individual and session, never randomly</b>, and always report identity accuracy alongside context accuracy so the reader can see how much leak was available.</li>
<li>The upside: if the literature's numbers collapse under proper splits across several species, <em>that itself is a paper</em> &mdash; the leakage audit from IDEAS.md, on real data.</li>
</ul></div>
</div></section>

<section id="next"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Sequence</p><h2>The four experiments, cheapest first</h2><p class="lede">Every one runs on the laptop or on Kaggle's free tier. None needs the cluster.</p></div>
<ol class="phases">
<li><div class="ph-body"><h3>Same data, real encoders <span class="ph-when">next &middot; one evening</span></h3><p>Embed the 440 meows with AVEX <code>esp_aves2_sl_beats_all</code>, <code>esp_aves2_effnetb0_bio</code>, BirdAVES, WavLM and Perch&nbsp;2. Same leave-one-cat-out probe. One table: encoder &times; {{honest, leaky, identity}}. Win condition: any encoder clears {loco:.2f} honest. Bonus finding: which encoder carries the <em>least</em> identity &mdash; that's the one that hears context.</p><div class="meta"><span class="chip ch-ok">CPU, minutes</span><span class="chip ch-dim">avex + torch installed locally</span></div></div></li>
<li><div class="ph-body"><h3>Add species, same protocol <span class="ph-when">week 1&ndash;2</span></h3><p>Dogs (BEANS <code>dogs</code>: barks in six situations), Egyptian fruit bats (15k calls with feeding / mating / sleep / perch contexts), pigs (calls across 19 contexts of known valence), meerkats, zebra finches (11 call types with behavioural context), marmosets. Same table per species. Now the leakage audit has teeth.</p><div class="meta"><span class="chip ch-ok">CPU or Kaggle</span><span class="chip ch-warn">datasets to verify one by one</span></div></div></li>
<li><div class="ph-body"><h3>Cross-species transfer <span class="ph-when">week 3&ndash;4 &middot; the interesting one</span></h3><p>Collapse every dataset's labels onto one shared axis: <b>high-arousal / negative</b> (alarm, distress, aggression, isolation) versus <b>low-arousal / positive</b> (contact, food, brushing, play). Train the probe on species A, test zero-shot on species B, for every pair. Above chance across many pairs means the encoder captures shared acoustic structure across mammals &mdash; the rigorous version of &ldquo;can a dog and a cat talk.&rdquo; Tonight's data already hints at it: the one axis with signal was calm-at-home versus alone-and-anxious.</p><div class="meta"><span class="chip ch-ok">numpy on cached embeddings</span><span class="chip ch-warn">novelty check pending</span></div></div></li>
<li><div class="ph-body"><h3>What drives it, and can we trust it <span class="ph-when">month 2</span></h3><p>Correlate the probe direction with pitch, duration and spectral tilt (the acoustic-universals hypothesis, at the embedding level). Look for a sparse-autoencoder feature that fires for distress across species. Then wrap the tags in the conformal method we already built, so a &ldquo;distress&rdquo; label comes with a guarantee. That's chapters 5 and 1 of the thesis, reused.</p><div class="meta"><span class="chip ch-ok">laptop</span></div></div></li>
</ol>
</div></section>

<section id="data"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Corpora</p><h2>Datasets that carry context labels</h2><p class="lede">All small. All public or academic-access. Only the first is verified end-to-end tonight; the rest are candidates I've seen cited and will confirm before we depend on them.</p></div>
<div class="tscroll"><table>
<thead><tr><th>Dataset</th><th>Species</th><th>Contexts / labels</th><th class="num">Size</th><th>Individuals</th><th>Status</th></tr></thead>
<tbody>
<tr><td class="name">CatMeows</td><td>domestic cat</td><td class="t-note">brushing &middot; waiting for food &middot; isolation</td><td class="num">440 &middot; 8.9 MB</td><td class="t-note">21, in filename</td><td><span class="chip ch-ok">verified, run</span></td></tr>
<tr><td class="name">BEANS dogs (Moln&aacute;r 2008)</td><td>domestic dog</td><td class="t-note">barks in ~6 situations: stranger, fight, walk, alone, ball, play</td><td class="num">~700 barks</td><td class="t-note">~10 dogs</td><td><span class="chip ch-warn">via alp-data, verify</span></td></tr>
<tr><td class="name">Egyptian fruit bat (Prat 2016)</td><td>bat</td><td class="t-note">feeding &middot; mating &middot; sleep &middot; perch; emitter + addressee</td><td class="num">~15k calls</td><td class="t-note">yes, per call</td><td><span class="chip ch-warn">candidate, verify</span></td></tr>
<tr><td class="name">Pig calls (Briefer 2022)</td><td>pig</td><td class="t-note">19 contexts, positive / negative valence</td><td class="num">7,414 calls</td><td class="t-note">many, per call</td><td><span class="chip ch-warn">candidate, verify access</span></td></tr>
<tr><td class="name">Meerkat calls</td><td>meerkat</td><td class="t-note">alarm &middot; close call &middot; short note &middot; lead &middot; move &middot; aggression</td><td class="num">hours</td><td class="t-note">tagged individuals</td><td><span class="chip ch-warn">via Voxaboxen demo / alp-data</span></td></tr>
<tr><td class="name">Zebra finch (Elie &amp; Theunissen)</td><td>zebra finch</td><td class="t-note">11 call types incl. distress, alarm, aggressive, begging, contact</td><td class="num">thousands</td><td class="t-note">yes</td><td><span class="chip ch-warn">alp-data ZebraFinchJulieElie</span></td></tr>
<tr><td class="name">InfantMarmosetsVox</td><td>marmoset</td><td class="t-note">call types (phee, twitter, trill, tsik&hellip;)</td><td class="num">&mdash;</td><td class="t-note">yes</td><td><span class="chip ch-warn">alp-data</span></td></tr>
<tr><td class="name">Elephant rumbles (ESP, Keen 2026)</td><td>African elephant</td><td class="t-note">social context (EB vs non-EB), convergence exchanges</td><td class="num">repo 87 MB</td><td class="t-note">&mdash;</td><td><span class="chip ch-ok">repo exists, ESP</span></td></tr>
<tr><td class="name">GiantOtters</td><td>giant otter</td><td class="t-note">22 call types</td><td class="num">&mdash;</td><td class="t-note">&mdash;</td><td><span class="chip ch-warn">alp-data</span></td></tr>
</tbody></table></div>
<p class="small col">Sizes and label lists for the unverified rows are from memory of the papers and may be off; the verification pass that was meant to confirm them tonight hit a session limit and will be re-run.</p>
</div></section>

<section id="cross"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Experiment 3, sketched</p><h2>The cross-species picture</h2><p class="lede">What experiment 3 produces: a species-by-species grid. Row = trained on, column = tested on. Diagonal is the within-species result; everything off the diagonal is transfer. The version below is <b>illustrative</b> &mdash; a hypothesis of what a shared arousal axis would look like, not data.</p></div>
<div class="diagram"><svg viewBox="0 0 900 330" role="img" aria-label="Illustrative species-by-species transfer matrix">
<g font-family="var(--mono)" font-size="11" fill="var(--ink-2)">
<text x="200" y="40" text-anchor="middle">cat</text><text x="290" y="40" text-anchor="middle">dog</text><text x="380" y="40" text-anchor="middle">pig</text><text x="470" y="40" text-anchor="middle">bat</text><text x="560" y="40" text-anchor="middle">meerkat</text><text x="650" y="40" text-anchor="middle">finch</text>
<text x="150" y="82" text-anchor="end">cat</text><text x="150" y="132" text-anchor="end">dog</text><text x="150" y="182" text-anchor="end">pig</text><text x="150" y="232" text-anchor="end">bat</text><text x="150" y="282" text-anchor="end">meerkat</text>
</g>
<text x="60" y="30" font-family="var(--mono)" font-size="9.5" letter-spacing="1.4" fill="var(--ink-3)">TRAIN ↓ / TEST →</text>
<g stroke="var(--rule)">
<rect x="160" y="55" width="80" height="45" fill="var(--cyan)" opacity=".85"/><rect x="250" y="55" width="80" height="45" fill="var(--cyan)" opacity=".45"/><rect x="340" y="55" width="80" height="45" fill="var(--cyan)" opacity=".4"/><rect x="430" y="55" width="80" height="45" fill="var(--cyan)" opacity=".18"/><rect x="520" y="55" width="80" height="45" fill="var(--cyan)" opacity=".3"/><rect x="610" y="55" width="80" height="45" fill="var(--cyan)" opacity=".1"/>
<rect x="160" y="105" width="80" height="45" fill="var(--cyan)" opacity=".5"/><rect x="250" y="105" width="80" height="45" fill="var(--cyan)" opacity=".85"/><rect x="340" y="105" width="80" height="45" fill="var(--cyan)" opacity=".45"/><rect x="430" y="105" width="80" height="45" fill="var(--cyan)" opacity=".2"/><rect x="520" y="105" width="80" height="45" fill="var(--cyan)" opacity=".35"/><rect x="610" y="105" width="80" height="45" fill="var(--cyan)" opacity=".12"/>
<rect x="160" y="155" width="80" height="45" fill="var(--cyan)" opacity=".4"/><rect x="250" y="155" width="80" height="45" fill="var(--cyan)" opacity=".4"/><rect x="340" y="155" width="80" height="45" fill="var(--cyan)" opacity=".85"/><rect x="430" y="155" width="80" height="45" fill="var(--cyan)" opacity=".2"/><rect x="520" y="155" width="80" height="45" fill="var(--cyan)" opacity=".3"/><rect x="610" y="155" width="80" height="45" fill="var(--cyan)" opacity=".1"/>
<rect x="160" y="205" width="80" height="45" fill="var(--cyan)" opacity=".18"/><rect x="250" y="205" width="80" height="45" fill="var(--cyan)" opacity=".18"/><rect x="340" y="205" width="80" height="45" fill="var(--cyan)" opacity=".2"/><rect x="430" y="205" width="80" height="45" fill="var(--cyan)" opacity=".85"/><rect x="520" y="205" width="80" height="45" fill="var(--cyan)" opacity=".2"/><rect x="610" y="205" width="80" height="45" fill="var(--cyan)" opacity=".1"/>
<rect x="160" y="255" width="80" height="45" fill="var(--cyan)" opacity=".3"/><rect x="250" y="255" width="80" height="45" fill="var(--cyan)" opacity=".35"/><rect x="340" y="255" width="80" height="45" fill="var(--cyan)" opacity=".3"/><rect x="430" y="255" width="80" height="45" fill="var(--cyan)" opacity=".2"/><rect x="520" y="255" width="80" height="45" fill="var(--cyan)" opacity=".85"/><rect x="610" y="255" width="80" height="45" fill="var(--cyan)" opacity=".15"/>
</g>
<g font-family="var(--serif)" font-size="11.5" fill="var(--ink-2)">
<text x="720" y="80">■ diagonal: within-species (the E1 number)</text>
<text x="720" y="102">■ off-diagonal: zero-shot transfer</text>
<text x="720" y="140">Hypothesis: mammals share an</text><text x="720" y="156">arousal axis (pitch, duration,</text><text x="720" y="172">spectral tilt), so mammal→mammal</text><text x="720" y="188">transfers and mammal→bird mostly</text><text x="720" y="204">doesn't. If the grid comes out flat,</text><text x="720" y="220">that's a clean negative result.</text>
<text x="720" y="258" fill="var(--mag)">Illustrative only. Not data.</text>
</g>
</svg></div>
</div></section>

<section id="compute"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Hardware</p><h2>None of this needs the cluster</h2></div>
<div class="tscroll"><table>
<thead><tr><th>Step</th><th>Where it runs</th><th class="num">Time</th><th>Notes</th></tr></thead>
<tbody>
<tr><td class="t-note">Tonight's baseline (MFCC + probe)</td><td class="name">laptop, stock python</td><td class="num">2 s</td><td class="t-note">done</td></tr>
<tr><td class="t-note">Embed 440 meows with 5 frozen encoders</td><td class="name">laptop CPU / MPS</td><td class="num">minutes</td><td class="t-note">~90M-param models, 8 GB is plenty; venv now installs cleanly on the APFS SSD</td></tr>
<tr><td class="t-note">Embed every dataset in the table (~30 h audio)</td><td class="name">laptop overnight, or Kaggle T4</td><td class="num">1&ndash;3 h</td><td class="t-note">one pass, cache to disk, never touch audio again</td></tr>
<tr><td class="t-note">Probes, transfer grid, SAE, conformal</td><td class="name">laptop, numpy</td><td class="num">seconds each</td><td class="t-note">this is where all the science happens</td></tr>
<tr><td class="t-note">NatureLM-audio &ldquo;is this an alarm call?&rdquo; on the same clips</td><td class="name">HF Space or Kaggle 8-bit</td><td class="num">optional</td><td class="t-note">the only step that wants a GPU, and only as a comparison row</td></tr>
<tr><td class="t-note">Fine-tuning anything</td><td class="name">SFSU cluster, one-shot</td><td class="num">&mdash;</td><td class="t-note">only if frozen probes plateau. Currently no reason to.</td></tr>
</tbody></table></div>
</div></section>

<section id="rules"><div class="wrap stack">
<div class="sec-head"><p class="eyebrow">Discipline</p><h2>Rules and kill conditions</h2></div>
<div class="grid g2">
<div class="note"><h4>Vocabulary</h4><ul><li>Say <b>context</b>, <b>call type</b>, <b>arousal</b>, <b>predictability</b>, <b>shared acoustic structure</b>.</li><li>Never say <b>meaning</b>, <b>words</b>, <b>translate</b>, <b>what they're saying</b> &mdash; not in code comments, not in emails, not in the paper.</li><li>Report identity accuracy next to every context accuracy. Always.</li></ul></div>
<div class="note bad"><h4>Kill conditions</h4><ul><li>No encoder beats {loco:.2f} honest on cats &rarr; context isn't in the sound at clip level for this species; move on, keep the negative.</li><li>Cross-species grid is flat &rarr; no shared code in the embedding; publish the clean negative and pivot the chapter to the leakage audit.</li><li>Datasets don't expose individual IDs &rarr; we can't split honestly &rarr; don't use them, however tempting.</li></ul></div>
</div>
</div></section>
</main>
<footer><div class="wrap"><p>Run 17 Aug 2026 on an Apple M2 with stock anaconda Python. Code: <code>context_probe.py</code> (numpy/scipy/sklearn only). Data: CatMeows, Zenodo 4008297, CC-BY-4.0. Companion pages: The Earth Species Stack (capability map). Living plan: <code>/Volumes/SSD/esp-lab/PLAN.md</code> &middot; ideas: <code>IDEAS.md</code>.</p></div></footer>
"""
out = os.path.join(D, "context-probe.html")
open(out, "w").write(html)
print("wrote", out, len(html)//1024, "KB")
