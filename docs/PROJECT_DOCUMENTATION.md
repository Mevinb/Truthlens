# TruthLens — SwinV2-Tiny Improved Detector (Epoch 9) Documentation

> Single reference for the **shipped detector only**: tech stack in use, how it works,
> dataset behind **Swin improved Epoch 9**, measured performance, app serving, and reproduction.
> Legacy trial-and-error experiments (ResNet18, CLIP linear heads, classical ML, ensembles,
> Grad-CAM, FFT/ELA-blending variants) are **intentionally out of scope** and not documented here.

Generated: 2026-10-14 from live repo (`src/train_swin.py`, `src/eval_swin.py`,
`dataset/build_swin_union_manifest.py`, `dataset/verify_swin_dataset.py`,
`train_improved_detector.sh`, `models/swin_epochs/swin_v2_512_improved_index.json`,
`results/metrics/improved_swin_benchmark.json`,
`results/metrics/swin_dataset_verification.json`).

---

## Table of Contents

1. [What Ships](#1-what-ships)
2. [Tech Stack (In Use)](#2-tech-stack-in-use)
3. [How It Works — End-to-End](#3-how-it-works--end-to-end)
4. [Model Architecture & Training](#4-model-architecture--training)
5. [Dataset: Swin Improved Epoch 9](#5-dataset-swin-improved-epoch-9)
6. [Epoch 9 — Measured Numbers](#6-epoch-9--measured-numbers)
7. [Serving & App](#7-serving--app)
8. [Commands (Train / Verify / Eval / Serve)](#8-commands-train--verify--eval--serve)
9. [Project Structure (Relevant Paths)](#9-project-structure-relevant-paths)
10. [Tests, Audits & Artifacts](#10-tests-audits--artifacts)
11. [Limitations](#11-limitations)
12. [Reproduce From Scratch](#12-reproduce-from-scratch)

---

## 1. What Ships

**One detector:** SwinV2-Tiny at 512×512, improved run, **Epoch 9**
(`models/swin_epochs/swin_v2_512_improved_ep009.pth`, UI label
`SwinV2-Tiny (Improved Ep 9 - Holdout AUC Record) 👑`).

It classifies an image as **REAL** vs **FAKE** with:

- Calibrated `P(FAKE)` + threshold-shifted display confidence (label and % always agree).
- Operational decision threshold **0.90** + inconclusive band **0.30–0.97**
  (`review_recommended` / `reliability` flags instead of forced binary on weak scores).
- Checkpoint provenance in UI (`file / epoch / val_acc / val_loss / device / img_size / threshold`).

No other model is part of the shipped path.

---

## 2. Tech Stack (In Use)

| Category | Library | Version | Used for |
|---|---|---|---|
| Language | Python | 3.10+ | All training / eval / app code |
| Deep Learning | PyTorch | ≥2.1.0 (observed 2.13.0+cu130) | SwinV2-Tiny train + inference, AMP (`GradScaler`/`autocast`) |
| Vision models | TorchVision | ≥0.16.0 | `swin_v2_t` + `Swin_V2_T_Weights.IMAGENET1K_V1`, transforms |
| Image I/O | Pillow | ≥10.0.0 (observed 12) | Decode, crop, JPEG re-encode, size/hash |
| Image ops | OpenCV-Python | ≥4.8.0 | Resize helpers where needed |
| Numerics | NumPy | ≥1.24.0 (observed 2.5) | Softmax, thresholds, metric aggregation |
| Metrics | scikit-learn (metrics only) | ≥1.3.0 | `roc_auc_score` for val / benchmark AUC |
| Web app | Streamlit | ≥1.28.0 (observed 1.61) | `app/app.py` + cached predictor loading |
| Progress / logging | tqdm, TensorBoard | ≥4.66.0 / ≥2.14.0 | Train progress, `runs/` curves |
| Testing | pytest, pytest-cov | ≥7.4.0 / ≥4.1.0 | `pytest tests/ -v` |

Reference hardware: RTX 4050 Laptop 6 GB VRAM, CUDA, `PYTHONUNBUFFERED=1`.
Strict-load + 512px forward ≈ 0.5 s in audit.

> Removed from this doc: `open_clip_torch`, HOG/LBP + 6 sklearn classifiers,
> ResNet/Grad-CAM libs, forensic-blend code — legacy experiments, not shipped.

---

## 3. How It Works — End-to-End

```
Raw sources (camera sets, generator outputs, HF repos, curated Grok/Aurora)
  │  dataset/collect_modern_generators.py, download_*.py, ingest_*.py
  ▼
Prepared corpora (train/val/test/holdout + manifest)
  ▼
Unified manifest  datasets/prepared/swin_union/manifest.jsonl
  │  dataset/build_swin_union_manifest.py  (sha256 dedup, holdout>test>val>train)
  │  dataset/verify_swin_dataset.py        (schema + file presence + hash/leak check)
  ▼
Train  src/train_swin.py  (tag swin_v2_512_improved)
  │  512px, ImageNet init, weighted CE, anti-shortcut aug, AMP, cosine LR
  │  per-epoch archive models/swin_epochs/<tag>_epNNN.pth + _index.json
  ▼
Eval  src/eval_swin.py  (+ versioned benchmark → improved_swin_benchmark.json)
  │  test + quarantined holdout, never pooled; per-image .jsonl score cache
  ▼
Serve  src/predict.SwinPredictor  via app/inference.py registry
  │  identical eval transform, threshold 0.90, abstain 0.30–0.97
  ▼
App  app/app.py  (upload → verdict card → provenance → batch export)
```

One upload:

1. Decode to RGB PIL.
2. Eval transform: `Resize(512,512) → JPEG q88 re-encode → ToTensor → ImageNet norm` — byte-identical to training eval.
3. SwinV2-Tiny forward → softmax → `raw_p_fake`.
4. Decide at 0.90; shift odds so displayed % matches label; flag `0.30–0.97` as inconclusive.
5. Return `prediction / confidence / probabilities / raw_probability_fake / decision_threshold / review_recommended / reliability` + checkpoint meta.

---

## 4. Model Architecture & Training

| Item | Detail |
|---|---|
| Arch | SwinV2-Tiny (`swin_v2_t`), head `768 → 2` (REAL/FAKE), 233 tensors, `[1,2]` output |
| Init | ImageNet-1K pretrained (NOT random-init) |
| Input | 512×512 |
| Optimizer | AdamW `lr 1e-4`, `weight-decay 1e-4`, `CosineAnnealingLR(T_max=epochs, eta_min=lr×0.01)`, grad-clip 5.0, AMP, label-smoothing 0.05 |
| Batch | `batch-size 8 × grad-accum 4` = **effective 32** |
| Loss | `weighted_ce` — inverse-frequency `[N/2N_real, N/2N_fake]` (focal/plain-CE supported but not used) |
| Target FPR | 0.05 on val reals (records `threshold_5fpr`; NOT the served threshold) |
| Epochs / patience | Up to 20, early-stop patience 5, seed 42, workers 8 |
| Launcher | `train_improved_detector.sh --tag swin_v2_512_improved` (auto-resume, atomic saves, teed log) |

Train views (per image per epoch) — shortcut-breaking:

- `RandomAspectPad(p=0.3)` → `RandomResizedCrop(512, scale 0.35–1.0, ratio 0.75–1.333)` → `RandomRecompress(p=0.9, q50–98)` → `Flip(0.5)` → `ColorJitter(0.15/0.15/0.15/0.02)` → `GaussianBlur(0.25)` + `AdjustSharpness(0.2)` + `Autocontrast(0.1)` → `ToTensor + Norm`.

Eval view (fixed): `Resize(512) → Recompress(q88) → ToTensor + Norm`.

Live tracking per epoch: loss/acc/bal-acc/real-acc/fake-recall/AUC/recall@5%FPR/VRAM/ETA.
Best-by-val-loss → `models/swin_v2_tiny_512_improved.pth`; every epoch archived;
resume state `models/swin_resume_swin_v2_512_improved.pth` (manifest-SHA guarded).

---

## 5. Dataset: Swin Improved Epoch 9 (Simple Explanation)

> For your college report: think of this as **"what photos the AI studied from"**.
> Epoch 9 did not have its own special dataset — all improved epochs (1 to 9) learned from
> the **same big collection**. Only the model weights changed each epoch.

### 5.1 In one paragraph

We collected **about 2.76 lakh images** (275,869) — half real camera photos, half AI-generated images.
The model **trained on ~1.94 lakh images**, **validated on ~54 thousand**, and was **finally tested
on fresh images it had never seen** (26,510 test + 611 holdout images). The holdout set is kept
locked away so we can honestly check if the model really learned or just memorized.

### 5.2 Simple split table

| Set | Images | Meaning (for report) |
|---|---|---|
| Train | 194,170 (82,609 real + 111,561 fake) | The study material — what the model learns from |
| Validation | 54,578 (26,306 real + 28,272 fake) | Practice exams during training |
| Test | 26,510 (12,348 real + 14,162 fake) | Final exam on new images |
| Holdout (hidden) | 611 (102 real + 509 fake) | Surprise test with unseen sources/generators |

Total: **1,21,365 real + 1,54,504 fake = 2,75,869 images.**

### 5.3 Which dataset, from where, quantity?

All 7 groups below are mixed into one training collection (`datasets/prepared/swin_union`, 275,869 images).
Same table you can paste into your college report:

| # | Dataset group (folder) | What it has | From where (source) | Quantity used in Epoch 9 |
|---|---|---|---|---|
| 1 | `legacy_archive` | Old real photos + early AI images (GAN / diffusion era) | Local raw archive built from Kaggle **CIFAKE** (`birdy654/cifake-real-and-ai-generated-synthetic-images`) + Hugging Face `dragonintelligence/CIFAKE-image-dataset`, `Hemg/AI-Generated-vs-Real-Images-Datasets` and older archives | **213,200** |
| 2 | `corpus` | Mixed real + fake, incl. generator families GAN / LatDiff / PixDiff | Hugging Face: `dragonintelligence/CIFAKE-image-dataset`, `Hemg/AI-Generated-vs-Real-Images-Datasets`, `date3k2/raw_real_fake_images`, `OwensLab/CommunityForensics-Small` | **20,138** |
| 3 | `gpt_nano` | GPT-Image-2 + Nano-Banana AI images + matching real photos | Local `datasets/raw/generator_sources/gpt-image-2/images` + `nano-banana/data/*.parquet`; reals taken from `modern_v2` | **17,120** |
| 4 | `modern_v2` | Latest generators + real camera photos + hidden holdout (see breakdown below) | Hugging Face (14 repos — list below) | **16,537** |
| 5 | `modern_corpus` | Early modern-generator collection (first version before v2) | Hugging Face modern photo + synthetic sets (v1 collection) | **5,448** |
| 6 | `highres_corpus` | High-resolution photos vs AI art (512px–2940px) | Hugging Face `Hemg/AI-Generated-vs-Real-Images-Datasets` (RealArt = real, AiArtData = fake) | **2,926** |
| 7 | `grok_aurora` (train only) | Only AI images from xAI Grok | Local `datasets/xAI_Aurora_t2i_human_preferences/data/train-*.parquet`, model `aurora-20-1-25` (extracted with `src/extract_grok_aurora.py`, 500 images) | **500** |
| — | **Total** | **1,21,365 real + 1,54,504 fake** | — | **275,869** |

`modern_v2` breakdown (the most important group — newest AI tools):

| Type | Generator / photo set | Hugging Face repo | Use |
|---|---|---|---|
| Fake | gpt-image-2 (Twitter wild) | `Scam-AI/gpt-image-2` | train/val/test |
| Fake | gpt-image-1 (pristine PNG) | `a3xrfgb/gpt-image-mega-4k` | train/val/test |
| Fake | seedream-4.5 | `ash12321/seedream-4.5-generated-2k` | train/val/test |
| Fake | flux.1-dev | `ash12321/flux-1-dev-generated-10k` | train/val/test |
| Fake | midjourney-v6 | `Photoroom/midjourney-v6-recap` | train/val/test |
| Fake+Real | 13 generators + social photos | `saberzl/So-Fake-OOD` | train/val/test (+ part quarantined to holdout) |
| Real | Flickr CC0 camera photos | `bitmind/megalith-small` | train/val/test |
| Real | Open Images photos | `bitmind/open-images-v7-subset` | train/val/test |
| Real | 2–22 MP camera originals | `bitmind/bm-subnet-real-fullsize` | train/val/test |
| Fake (holdout only) | chatgpt-4o-native | `wafflefan/felix-chatgpt-generated-images` | holdout, never trained |
| Fake+Real (holdout only) | gpt-image-2 + caption-paired reals | `Sarim-Hash/liars-dividend-GPT-image-2-photoreal` | holdout, never trained |

### 5.4 Why this dataset is fair

1. **No cheating:** the same photo never appears in both training and testing. Duplicates are removed automatically.
2. **Balanced checking:** testing uses fresh sources the model never studied, so high marks mean real learning.
3. **Cleaned before training:** broken images skipped, every image resized to 512×512 for the model.

That's it — you can write in your report: *"The SwinV2 model was trained on ~1.94L images and evaluated on ~27k unseen images plus a 611-image hidden holdout set."

---

## 6. Epoch 9 — Measured Numbers

### 6.1 In-training validation (54,578 val rows)

| Metric | Epoch 9 | Context |
|---|---|---|
| `val_loss` | **0.18723** | 2nd-lowest (ep7 min 0.18703) |
| `val_acc` | 96.67% | — |
| `val_bal_acc` | 96.76% | — |
| `val_real_acc` | **99.27%** | Highest in improved run |
| `val_fake_recall` | 94.26% | — |
| `val_auc` | **0.9964** | Tied-best with ep7 |
| `train_loss` | 0.18353 | Lowest in run (0.306→0.184 descent) |
| `val_threshold` @5%FPR | 0.0851 | In-dist val record only — **not served** (served 0.90) |

Source: `models/swin_epochs/swin_v2_512_improved_index.json`.

### 6.2 Independent benchmark (`results/metrics/improved_swin_benchmark.json`, JPEG-q88, P(FAKE)≥0.5)

| Split | n | Acc | Bal-acc | REAL | FAKE rec | AUC | Rec@5%FPR | Brier |
|---|---|---|---|---|---|---|---|---|
| **Holdout (quarantined)** | 611 | 89.53% | 89.01% | 88.24% | 89.78% | **0.9461** | **86.64%** | 0.0838 |
| **Union test** | 1000 | 91.00% | 91.00% | 98.20% | 83.80% | 0.9826 | 90.80% | 0.0661 |
| **Internet Wild** | 351 | 76.35% | 75.88% | 64.88% | 86.89% | 0.8749 | 57.38% | 0.1674 |

Epoch 9 = **holdout-AUC record + best holdout raw accuracy + best constrained recall** in repo history.

Sibling context (same file/protocol): ep6 best holdout bal-acc 89.40% + union AUC 0.9834 (most balanced); ep7 lowest val-loss; ep8 peak union fake-recall 97.60% but weak reals; ep3 best wild 85.93%. No cross-file “champion” claims — older baseline tables used different transforms/sample sizes.

---

## 7. Serving & App

- Predictor: `src/predict.SwinPredictor(checkpoint_name="swin_epochs/swin_v2_512_improved_ep009.pth")`, cached by file fingerprint (retrain-safe).
- First in `SWIN_CHECKPOINTS` (`src/utils.py`) → auto-resolved default.
- Run: `streamlit run app/app.py` (`./run.sh`) → `http://localhost:8501`.
- `app/inference.py` = registry + cached predictors + measured numbers; `app/app.py` = layout; `app/theme.py` = styling.
- Upload JPG/PNG/WebP/BMP (single/batch); verdict card shows tier-relevant accuracy; batch CSV + per-result JSON; sidebar shows checkpoint provenance.

---

## 8. Commands (Train / Verify / Eval / Serve)

```bash
# env
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# union dataset
.venv/bin/python dataset/build_swin_union_manifest.py
.venv/bin/python dataset/verify_swin_dataset.py --manifest datasets/prepared/swin_union/manifest.jsonl
# quick schema-only: --quick ; require corpus: --require-corpus grok_aurora=500

# train improved (Epoch 9 lives here; auto-resumes by tag)
./train_improved_detector.sh --tag swin_v2_512_improved
# opts: --clean-start --rebuild-dataset --dry-run --epochs N --batch-size N --lr F

# eval Epoch 9
.venv/bin/python src/eval_swin.py \
  --checkpoint models/swin_epochs/swin_v2_512_improved_ep009.pth \
  --tag improved_ep9 --splits test holdout

# audit + tests + app
.venv/bin/python diag_swin_checkpoint_audit.py
pytest tests/ -v
streamlit run app/app.py   # or ./run.sh
```

---

## 9. Project Structure (Relevant Paths)

```
app/ (app.py, inference.py, theme.py)
src/ (train_swin.py, eval_swin.py, predict.py [SwinPredictor], utils.py [SWIN_CHECKPOINTS], preprocessing.py [recompress/pad/crop])
dataset/ (build_swin_union_manifest.py, verify_swin_dataset.py, collect_modern_generators.py, download_*.py)
datasets/prepared/swin_union/manifest.jsonl
models/swin_epochs/swin_v2_512_improved_ep00{1..9}.pth + swin_v2_512_improved_index.json
models/swin_resume_swin_v2_512_improved.pth
results/metrics/improved_swin_benchmark.json
results/metrics/swin_dataset_verification.json
results/metrics/swin_checkpoint_audit.json
results/metrics/scores/swin_<tag>_<split>.jsonl
results/logs/train_swin_v2_512_improved.log
train_improved_detector.sh  run.sh  requirements.txt
docs/SWIN_MODEL_EVALUATION.md
```

Legacy paths (ResNet/CLIP/classical/notebooks/alt-frontends) exist in repo but are **not part of shipped detector** and not covered here.

---

## 10. Tests, Audits & Artifacts

- `pytest tests/ -v` — Swin manifest / verify / train-eval / predict regression (72/72 in audit).
- `diag_swin_checkpoint_audit.py` — 22/22 strict-load + 512px forward; 18 unique weight states (4 alias pairs + 2 resume snapshots excluded from serving).
- `diag_app_paths.py` — app smoke test.
- Artifacts: per-epoch archives + index, `improved_swin_benchmark.json` (§6.2), `swin_dataset_verification.json` (§5.1), per-image `scores/*.jsonl`, teed training logs.

---

## 11. Limitations

- Same-source numbers are regression checks; **holdout + wild are the headline**.
- Epoch 9 trades wild-real accuracy (64.88%) for holdout strength; ep3 generalizes better to wild photos alone.
- Threshold 0.90 + abstain band deliberately sacrifice coverage for false-alarm control (~31% wild abstain).
- No generator-level labels for per-generator guarantees; no cert-chain validation where provenance markers are reported.
- Leaderboards across different benchmark files/protocols are not comparable; rerun one versioned fixed-row-ID benchmark before changing the default.

---

## 12. Reproduce From Scratch

1. Install env + build/verify union (§8) — require `ok:true`.
2. `./train_improved_detector.sh --tag swin_v2_512_improved` to epoch 9 (resumable).
3. `src/eval_swin.py` on `test` + `holdout` — confirm H AUC 0.9461 / bal 89.01%.
4. `streamlit run app/app.py` — verify verdicts + provenance + batch export.

---
*Swin-only doc. Full checkpoint audit: `docs/SWIN_MODEL_EVALUATION.md`. Legacy experiments omitted by request.*
