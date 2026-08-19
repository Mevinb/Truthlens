# TruthLens v2 — Detector Rebuild Plan

**Status:** Phase 0 complete — protocol built, baseline recorded. Phases 1–4 pending.
**Date:** 2026-08-16
**Hardware:** RTX 4050 Laptop, 6 GB VRAM · torch 2.13.0+cu130 · 139 GB free on `/home/mevlec/Data`

---

## 0. The honest baseline *(measured 2026-08-16, `src/eval_protocol.py`)*

The shipped `resnet18_modern.pth`, scored through the exact TTA path the app
serves, on 2 745 images across the four tiers:

| tier | what it is | n | AUC | balanced | real | fake recall | recall @5% FPR |
|---|---|---|---|---|---|---|---|
| T1 in-dist | same sources as train | 1694 | 0.992 | 94.2 % | 97.2 % | 91.2 % | 97.0 % |
| **T2 held-out source** | gpt-image-2, unseen repo | 200 | **0.634** | **53.5 %** | 95.0 % | **12.0 %** | 14.0 % |
| T3 held-out generator | chatgpt-4o-native, unseen repo | 500 | 0.993 | 95.5 % | 95.0 % | 96.0 % | 96.5 % |
| T4 wild | civitai / unsplash / openverse | 351 | 0.920 | 86.1 % | 86.3 % | 85.8 % | 76.5 % |

**The surprise is T3.** A generator the model has never seen scores **95.5 %**,
while a generator it *has* trained on, from a different repository, scores
**53.5 %**. Generator novelty is not the problem at all — provenance and
rendition are. chatgpt-4o-native generalizes for free because its rendition is a
statistical outlier (saturation 0.55 vs 0.28–0.31 for camera photos); pristine
photoreal gpt-image-2 fails because it looks like a photograph. That reframes the
target: **the job is photoreal images from unseen pipelines, not new generators.**

And the operational number is worse than the accuracy suggests. Fit the threshold
on T4 to hold false alarms at 5 %, apply it to T2 — the honest out-of-sample
operating point — and fake recall is **1.0 %**. On T2 there is no threshold that
helps: even an oracle allowed to tune on T2's own reals reaches only 14.0 %.

Both files are on disk: `results/metrics/protocol_board_baseline_resnet18_modern.json`,
and per-image scores at `results/metrics/scores/baseline_resnet18_modern.jsonl`.

---

## 1. Context — why rebuild

The shipped `resnet18_modern.pth` reports **93.95 %** on `datasets/prepared/modern_v2/test` and the README
claims **92.8 % / 0.983 AUC**. Both numbers are real arithmetic on the splits they name, and both
are misleading: the test split draws from the *same source repositories* as train, so it cannot
detect the failure mode the model actually has.

The failure, measured: **the same generator from a different source drops to chance.**

| eval set | what it is | served model |
|---|---|---|
| `datasets/prepared/modern_v2/test` | same source repos as train | 93.9 % acc |
| gpt-image-2 from `Scam-AI/gpt-image-2` (in test) | trained-on source | **93.3 %** |
| gpt-image-2 from `Sarim-Hash/liars-dividend` (holdout) | *unseen* source, same generator | **12.0 %** |
| `datasets/evaluation/internet` (civitai / unsplash) | genuinely independent | 86.0 % balanced |

A drop from 93 % to 12 % on **one generator across two sources** is not noise. The model learned
each source's rendition, not the generator.

### What it is not — three hypotheses tested and rejected

| hypothesis | test | result |
|---|---|---|
| JPEG / container artifact | re-encode holdout at q95 / q88 / q75 | `mean_p_fake` 0.249 → 0.262. **No effect.** |
| Resize / resample history | downscale ×0.75, ×0.5, down-then-up, crop-512 | 0.214 → 0.242 across all variants. **No effect.** |
| Subject matter | the holdout is *caption-paired* — each fake was generated from a real COCO photo's caption | Content is controlled by construction, and the gap persists. **Not content.** |

So the existing `RandomRecompress` / `RandomAspectPad` / `NativeScaleCrop` augmentations are not
the missing piece — more augmentation of that kind will not move this.

Content statistics say the same thing. Pristine gpt-image-2 sits *inside* the real distribution on
every measure; the sources the model does well on are the ones that sit outside it:

| source | model acc | sat_mean | grad_mean | highlight_frac |
|---|---|---|---|---|
| camera reals (3 sources) | 0.95–0.99 | 0.28–0.31 | 0.036–0.039 | 0.005–0.008 |
| gpt-image-2 **wild** (trained on) | 0.93 | 0.19 | **0.051** | **0.045** |
| chatgpt-4o-native (holdout) | 0.96 | **0.55** | 0.031 | 0.000 |
| gpt-image-2 **pristine** (holdout) | **0.12** | 0.25 | 0.034 | 0.003 |

The model aces sources that are statistical outliers and fails the one that is not.

---

## 2. Was ResNet18 too small? — measured: no

Capacity was held constant and only the *training* was varied. Same 11 M-parameter backbone, same
deterministic input view, same 5 000 training images. AUC shown; `paired_holdout` is the hard set.

| condition | paired_holdout | chatgpt4o | internet | own_test |
|---|---|---|---|---|
| **A** fine-tuned + its own head *(as served)* | **0.548** | 0.961 | 0.907 | 0.958 |
| **B** fine-tuned backbone frozen + fresh probe | 0.666 | 0.969 | 0.893 | 0.965 |
| **C** ImageNet backbone frozen, *never saw an AI image* | **0.695** | 0.920 | 0.776 | 0.930 |

**An ImageNet ResNet18 that has never seen a single AI image beats the fine-tuned one on the hard
set (0.695 vs 0.548).** Identical parameter count. Fine-tuning on this corpus did not merely fail
to help — it destroyed the signal, while buying real gains on distributions that resemble training
(own_test 0.958 vs 0.930, internet 0.907 vs 0.776). That is the shortcut trade-off, priced.

Two consequences:

1. **A bigger CNN fine-tuned the same way on the same data will not fix this.** More capacity fits
   the shortcut better. This is checked on the record in Phase 3, not assumed.
2. **But the representation genuinely is the bottleneck** — ResNet18-ImageNet features top out at
   0.695. That ceiling is a property of the features, and it is what a stronger *frozen* backbone
   is for. The instinct "change the model" is right; the correction is **freeze it**.

### The fix, already validated

Frozen CLIP ViT-L/14 (openai, quickgelu), fp16, same 5 000 training images, linear head. AUC:

| condition | paired_holdout | chatgpt4o | internet | own_test |
|---|---|---|---|---|
| resnet18 fine-tuned *(baseline)* | 0.548 | 0.961 | 0.907 | 0.958 |
| resnet18 ImageNet frozen *(ceiling to beat)* | 0.695 | 0.920 | 0.776 | 0.930 |
| CLIP whole-frame resize | 0.645 | 0.990 | **0.977** | 0.998 |
| **CLIP native 1:1 crop** | **0.923** | 0.993 | 0.943 | 0.996 |
| CLIP frame+native concatenated | 0.877 | **0.999** | 0.982 | **1.000** |

**0.548 → 0.923 on the set where the current model is at chance.** Three things this tells us that
shape the design:

- **Native 1:1 crops are what unlock the hard set** (0.923 vs 0.645 whole-frame). Downscaling to
  224 destroys the evidence — the `NativeScaleCrop` intuition was correct, it just needed a
  backbone strong enough to use it.
- **Whole-frame is better on the internet set** (0.977 vs 0.943). Both views carry real signal;
  neither dominates. Multi-view is required.
- **Concatenation is the wrong fusion** — it drops paired AUC from 0.923 to 0.877 by letting the
  head lean on frame features that mislead on that set. Fuse *scores*, not features.

**Calibration is a separate problem from signal.** `clip_native` reaches AUC 0.923 but only 0.690
balanced accuracy at threshold 0.5 (fake recall 0.390). The ranking is good; the threshold is in
the wrong place. That is fixable. Compare the ResNet18 at AUC 0.548, where *no* threshold helps.

**Cost, measured:** 304 M frozen params, **1.22 GB peak VRAM**, **13.8 ms/view**, so a 6-view pass
is **83 ms/image**. Fits the 4050 with room to spare.

> **One bookkeeping note so these tables stay comparable.** Every AUC in section 2
> is a *single deterministic view* (centre native crop, q88) — that is what makes
> the ResNet18 and CLIP arms directly comparable, since the only thing changed is
> the frozen feature extractor. The served ResNet18 with its full 13-view TTA does
> better than its single-view 0.548: it reaches **0.634** on T2 (section 0). So the
> honest gap to close is **0.634 → 0.90+**, and the CLIP arm has not yet been given
> its own multi-view treatment. Do not mix a single-view number with a TTA one.

---

## 3. Plan

### Phase 0 — Lock the evaluation protocol ✅ *done*

Nothing downstream is trustworthy until measurement is. `src/eval_protocol.py`.

- **Tiers are derived from the manifest, not hardcoded.** A generator counts as held-out
  because no training row mentions it, and a source counts as held-out because no
  training row shares its `source_key`. Adding a source in Phase 1 reclassifies the
  evaluation automatically — the alternative is a tier list that silently drifts out of
  step with the corpus, which is how one-source-per-generator shipped unnoticed.
- **A held-out fake from a trained source is reported as a leak** and excluded, rather
  than quietly averaged in. Pinned by `tests/test_eval_protocol.py`.
- **Four tiers, reported separately, never pooled.** Pooling is not a stylistic
  preference: the *pooled* holdout balanced accuracy is **87.1 %**, which looks fine and
  completely hides the 12.0 % inside it. Separation is what made the failure visible.
- **Three metrics, always together** — balanced accuracy at the served threshold (what
  the product does), AUC (what the features could do), and recall at 5 % FPR (the
  operational number). The pilot has these diverging by 23 points, so any one alone
  hides a different problem.
- **Thresholds can be fitted on one tier and applied to the others** (`--threshold-from`),
  because a threshold tuned on the tier it is scored on is an oracle, not a product. The
  in-tier figure is still reported, labelled `oracle`, alongside the sample-size
  resolution — 5 % of 100 reals is 5 images, and the report says so rather than implying
  three decimals it cannot support.
- **Per-image scores are cached** to `results/metrics/scores/<tag>.jsonl`, so Phase 2's
  calibration, fusion weights and threshold search all reuse one expensive pass.
- **Fixed** the `--checkpoint` double-join in `diag_eval_all.py` — `models/foo.pth`
  became `models/models/foo.pth` and surfaced as "nothing scored", which reads as an
  empty dataset rather than a mistyped flag. Now resolves a bare filename, a
  repo-relative path or an absolute path, and raises with the available checkpoints
  listed. Regression-tested.

### Phase 1 — Corpus: source diversity, not volume *(~20–30 GB)*

The failure is per-source, so the fix is **more sources per generator**, not more images per source.
Most generators currently have exactly one source, which is precisely why the overfitting was
invisible:

| generator | sources | repo |
|---|---|---|
| gpt-image-2 | 2 (wild + pristine) | `Scam-AI/gpt-image-2`, `Sarim-Hash/liars-dividend` *(holdout)* |
| gpt-image-1 | **1** | `a3xrfgb/gpt-image-mega-4k` |
| flux.1-dev | **1** | `ash12321/flux-1-dev-generated-10k` |
| midjourney-v6 | **1** | `Photoroom/midjourney-v6-recap` |
| seedream-4.5 | **1** *(160 train images)* | `ash12321/seedream-4.5-generated-2k` |
| chatgpt-4o-native | **1** *(holdout, eval-only licence)* | `wafflefan/felix-chatgpt-generated-images` |
| camera reals | 3 | megalith, open-images, bm-subnet-fullsize |

**T3 changes the priority order.** An unseen *generator* already scores 95.5 %, so hunting for
generator coverage is not the win it looked like. What fails is **photoreal output from an unseen
pipeline**. So rank candidate sources by how photographic they are, not by how many generator names
they add: a second pristine source for a generator already in the corpus is worth more than a first
source for a novel one.

#### Source survey — 42 repos checked against the HF API *(2026-08-16)*

The plan previously listed candidate names that had never been verified. They have now been
searched (28 keyword sweeps) and inspected (size, licence, splits, columns, real row values).
Nine of the named guesses did not exist; `competitions/aiornot` is gated. Raw output in
`results/metrics/hf_source_{candidates,inspect,rows,detail}.json`.

**Selected: `saberzl/So-Fake-OOD`** — cc-by-nc-4.0, 91 370 redistributable rows, 13 modern
photoreal generators plus social-platform reals, with per-row `generator` / `source_type` /
`source_platform`. Four measured reasons, in priority order:

1. **1 334 GPT-Image-2 images** — a *third independent source* for the exact generator that scores
   12 % from an unseen repo. This is the only intervention the evidence supports, applied to the
   precise point of failure.
2. **Its fakes are JPEG.** The audit reports PNG as 100 % fake — a pure container shortcut — and a
   large JPEG-fake population is what removes it.
3. **Its reals are social-platform photos at 1.5–1.9 MP**, the band the audit reports as 98.5 %
   fake, and a fourth real provenance against three curated photo sets today.
4. **Its card guarantees original bytes** — "we do not resize, crop, decode/re-encode, or
   normalize" — matching this collector's own guarantee, so nothing is laundered through a
   re-encode.

Partitioned on ingest: 9 generators and Reddit reals train; **FLUX 2, Imagen 4, Seedream 3.0 and
Recraft v3** (four separate vendors) plus **Tumblr and Bluesky** reals are quarantined. That gives
T3 a hard unseen-generator tier instead of one stylised source, and gives T2/T3 roughly 800
unseen-source reals instead of 100 — 5 % FPR on 100 reals is five images, a resolution the protocol
already flags as too coarse to report.

**Rejected, with reasons on the record** — so these are not re-litigated later:

| repo | why not |
|---|---|
| `34data/communityforensics-{real,fake}` | reals are JPEG, fakes are PNG. Format perfectly predicts label — it would *install* the shortcut the audit is trying to remove. |
| `TheKernel01/AIGC-Detection-Benchmark` | apache-2.0, 32 GB, 17 generator families — but all GAN-era/early-diffusion at 256 px. As holdout it would dilute T3 with easy images and inflate the headline tier. Deferred as a *separate* comparability tier, which needs a `tier` field the protocol does not have. |
| `marco-willi/synthbuster-plus` | 20 GB of pristine RAISE-based photoreal, which is exactly the right *shape* — but **no licence declared at all**, and its generators (DALL·E 2/3, GLIDE, Firefly, MJ v5) predate the corpus by three years. |
| `OwensLab/CommunityForensics-Small` | 260 GB, cc-by-nc-sa-4.0. Enormous checkpoint diversity, but sampled rows are 512×512 anime LoRA output — model diversity without photoreal content, which is the axis that already works. Note the full `OwensLab/CommunityForensics` is **cc-by-4.0**, a better licence, if this is revisited. |
| `saberzl/So-Fake-Set` (1.28 TB), `elsaEU/ELSA_D3` (2.6 TB), `jzousz/GenImage` (679 GB), `ENSTA-U2IS/GenImage` (653 GB) | too large for the budget; GenImage is also old-generator. |

- Extend `SOURCES` in `dataset/collect_modern_generators.py`. Target **≥3 independent source repos
  per generator**, and for each generator at least one **pristine** (direct API output) and one
  **laundered** (web-scraped, recompressed) source. ✅ `rendition: pristine|wild` now exists as a
  per-row field, so a whole rendition regime can be held out — a T5 tier, essentially free.
- **Reals need the same treatment** — 3 sources for the entire real class is its own leak axis, and
  T4's real accuracy (86.3 %, with unsplash at 83.9 %) is the weakest real-class number on the
  board. So-Fake's Reddit/Tumblr/Bluesky reals add a fourth, and a genuinely *wild* one.
- **seedream-4.5 has 160 training images.** Either bring it up to parity or drop it; at that size
  it contributes noise to the fake class and a 70.0 % T1 score that flatters nothing. So-Fake
  carries Seedream 3.0, a different version — quarantined, so it does not paper over this.
- Keep the content-hash split (leak-safe *within* a source) and keep caption-paired sets in eval
  only.
- **Gate:** `diag_corpus_audit.py` must show `leak: false` on every panel before training. It
  currently fails three — PNG 100 % fake, 1–2 MP 98.5 % fake, portrait 73 % fake.

#### Collector changes this required

`Source.label`, `.generator` and `.holdout` were per-source constants and the fetchers yielded bare
bytes, but So-Fake-OOD varies all four per row. Rather than one entry per partition — four full
passes over a 135 GB repo — the fetcher contract is now `(bytes, overrides)` and `Source` gained
`where` / `where_not` row filters, `label_column` + `label_map`, `generator_column` +
`generator_map`, `rendition_column` + `rendition_map`, `holdout_when`, and `extra_fields`.

Two of those are load-bearing rather than plumbing, and both are pinned by tests in
`tests/test_collector_provenance.py`:

- **`generator_map` must land a known generator on its existing corpus spelling.** Ingesting
  So-Fake's `GPT-image-2` under that name would register a *new* generator, so T2 would lose its
  subject and the fix would be invisible in the number it was meant to move. Unmapped values are
  skipped with a warning rather than passed through.
- **A quarantined row gets a distinct `source_key`** (`sofake_holdout`). Without it the protocol
  sees a held-out image whose provenance the model trained on, and correctly refuses to count it.
- And the excluded class is excluded **by name**: So-Fake ships `TAMPERED` (34 046 rows) whose
  `authenticity` field reads FAKE. Anything keying on `authenticity` alone would sweep locally
  retouched *real photographs* into the fake class.

### Phase 2 — Model: frozen CLIP + calibrated linear head

Design follows the pilot measurements directly.

- **Backbone:** frozen CLIP ViT-L/14 (openai, quickgelu), fp16. Never fine-tuned — proven above to
  be the thing that broke generalization.
- **Views:** 1 whole-frame (CLIP resize) + K native 1:1 224 crops (K=5: centre + quadrants),
  crop scores averaged.
- **Fusion:** one logistic head per view, then average *calibrated probabilities*. Fusion weight
  learned on a **held-out-source** validation set — never on in-distribution val.
- **Calibration as a first-class step:** Platt/isotonic on held-out-source data, operating point
  chosen at a fixed FPR (5 % on reals), not 0.5.
- **Feature cache:** `features/<corpus>/<view>.npy` + label/manifest index. Extraction is the only
  GPU cost; the head then trains in seconds and adding a generator re-extracts one source.
- **New files:** `src/features_clip.py` (extract + cache), `src/train_head.py` (head, calibration,
  threshold). Register a `clip` model type in `src/predict.py` so `app/inference.py` picks it up
  through the existing registry.

### Phase 3 — Backbone bake-off *(answers the capacity question on the record)*

All arms on identical data, splits and metrics.

| arm | params | trained how | cost |
|---|---|---|---|
| ResNet18 | 11 M | fine-tuned *(current baseline)* | done |
| ResNet18 | 11 M | frozen ImageNet + probe | done |
| ResNet50 | 25 M | fine-tuned | ~1.5 h — **the direct capacity test** |
| ConvNeXt-Tiny | 28 M | fine-tuned | ~1.5 h |
| CLIP ViT-B/32 | 88 M | frozen + probe | minutes — **control: size, or pretraining+freezing?** |
| CLIP ViT-L/14 | 304 M | frozen + probe | done (pilot) |
| DINOv2 ViT-L/14 | 304 M | frozen + probe | minutes — self-supervised, no language |

The ViT-B/32 arm is the one that matters for defensibility: if an 88 M frozen backbone beats a 25 M
fine-tuned one, the variable is pretraining and freezing, not parameter count.

### Phase 4 — Correct the README and app

- Replace the 92.8 % / 0.983 AUC headline with the T1–T4 table. T1 stays, explicitly marked
  in-distribution; **T2/T3 become the headline**.
- `app/inference.py`'s measured-metrics block reads from the new protocol output.
- The unvalidated per-generator guess: drop it or keep the existing explicit disclaimer.
- Classical ML models stay as the intended comparison baseline, stated plainly as at-chance
  (51–57 %).

---

## 4. Verification

```bash
# 0. the protocol — works now, this is the baseline everything is measured against
.venv/bin/python src/eval_protocol.py --all --tag baseline_resnet18_modern
.venv/bin/python src/eval_protocol.py --all --tag baseline_resnet18_modern \
    --reuse-scores --threshold-from T4_wild      # instant re-report, off-tier threshold

# 1. corpus is clean — every panel leak:false
.venv/bin/python diag_corpus_audit.py

# 2. features + head
.venv/bin/python src/features_clip.py --corpus datasets/prepared/modern_v2 --views frame,native
.venv/bin/python src/train_head.py --calibrate --target-fpr 0.05

# 3. the honest board for the new model, same protocol
.venv/bin/python src/eval_protocol.py --all --model clip --tag clip_v1

# 4. bake-off table
.venv/bin/python src/eval_protocol.py --bakeoff

# 5. app paths + unit tests
.venv/bin/python diag_app_paths.py && .venv/bin/python -m pytest
```

**Acceptance targets.** Baselines are the served numbers from section 0, so these are
gaps against what actually ships:

| metric | now (served) | target |
|---|---|---|
| **T2 AUC** | 0.634 | **≥ 0.90** |
| **T2 fake recall @5 % FPR, threshold fitted off-tier** | **0.010** | **≥ 0.60** |
| T3 AUC | 0.993 | hold ≥ 0.97 |
| T4 balanced acc | 0.861 | **≥ 0.93** |
| T1 in-distribution balanced | 0.942 | may *fall* — expected and acceptable |

Two of these deserve emphasis. The **off-tier** recall is the one that describes the
product: 1.0 % today, because the threshold that keeps false alarms tolerable on real
photos rejects essentially every pristine fake. And T3 is a *hold*, not a climb —
95.5 % is already there, so the risk in this rebuild is trading it away.

That last row matters too: if T1 does not drop somewhat, the shortcut probably survived.

---

## 5. Risks

- **App latency** goes from ~14 ms (ResNet18 TTA) to ~83 ms (6-view CLIP). Acceptable; if not,
  measure ViT-B/16 before compromising the views.
- **Grad-CAM** does not transfer to a frozen ViT + linear head. Needs replacing with attention
  rollout or per-crop score maps, or the feature drops. Not yet scoped.
- **Licensing:** `wafflefan/felix-chatgpt-generated-images` is all-rights-reserved, eval-only. It
  must stay `holdout=True`. Already handled — do not regress it.
- **Unverified dataset names** in Phase 1 need checking against HF before download; web search was
  unavailable this session.
- **Head saturation:** the pilot used 5 000 images and hit train-fit 1.000. Linear probes on frozen
  features saturate fast, so the corpus work in Phase 1 buys *diversity*, not volume — do not
  expect gains from simply adding more images per source.

---

## Appendix — measurements this plan rests on

Written by the probes run 2026-08-16; all raw output under `results/metrics/`.

| file | what it establishes |
|---|---|
| `protocol_board_baseline_resnet18_modern.json` | the honest T1–T4 baseline for the shipped model |
| `scores/baseline_resnet18_modern.jsonl` | 2 745 per-image scores, reusable for calibration |
| `probe_capacity_vs_representation.json` | capacity exonerated — frozen ImageNet 0.695 > fine-tuned 0.548 |
| `probe_clip_pilot.json` | frozen CLIP native-crop 0.923 on the hard set |
| `probe_resize_history.json` | resize/resample rejected as the cause |
| `recompress_skew.json` | JPEG/container rejected as the cause |
| `probe_content_gap.json` | pristine photoreal sits inside the real distribution |
| `corpus_audit_train.json` | three structural leaks still open |
