<div align="center">

# 🔍 TruthLens

### AI-Generated Image Detection using Machine Learning & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**TruthLens** classifies an uploaded image as ✅ **Real** or 🤖 **AI-Generated** using a ResNet18 deep learning model and 6 classical ML algorithms. The Streamlit web app provides confidence scores, Grad-CAM heatmaps, and AI explanations.

</div>

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Retrained CLIP detector** | Frozen CLIP ViT-L/14 + four calibrated linear view heads; 98.0% accuracy / 0.997 AUC on the modern same-source test and 90.8% balanced accuracy / 0.965 AUC on the quarantined holdout |
| 🧪 **ResNet18 baseline** | Retained for comparison and Grad-CAM, but no longer the app default because it overfits source-specific rendition cues |
| 📐 **6 Classical ML Models** | LR, DT, RF, SVM, k-NN, Naive Bayes — for comparison only (51–57%, near the 50% chance line) |
| 🔥 **Grad-CAM Heatmaps** | Visual explainability — shows where the network looked |
| 📊 **Confidence Scores** | Real-time probability breakdown (REAL % vs FAKE %) |
| 🎨 **Generator Detection** | Unvalidated heuristic guess at Stable Diffusion / Midjourney / FLUX / DALL-E — the corpus carries no per-generator labels, so this is not measured |
| 🧾 **Metadata & Provenance** | Reads EXIF/XMP/PNG text, camera/editor/generator fields, C2PA/JUMBF markers, ICC/alpha/container structure, appended payloads, and exact/perceptual hashes |
| 💧 **Watermark Checks** | Detects explicit watermark identifiers, possible visible border text/logos, hidden RGB under alpha, and weak LSB/steganography indicators |
| 💡 **AI Explanation** | Rule-based bullet-point explanation of the prediction |
| 🌐 **Streamlit Web App** | Dark UI, batch upload, held-out samples with true labels, resolution-calibrated accuracy |
| 📈 **Model Comparison** | Leaderboard with accuracy, F1, AUC across all models, read live from the benchmark file |

---

## 🏗️ Architecture

```
TruthLens/
│
├── dataset/
│   └── download_cifake.py       ← Automated dataset downloader
│
├── datasets/
│   ├── raw/                     ← Original downloads and generator sources
│   ├── prepared/                ← Training corpora, including CIFAKE
│   ├── evaluation/              ← Independent evaluation sets
│   └── fixtures/                ← Small test corpora
│
├── src/
│   ├── utils.py                 ← Config, logging, plotting helpers
│   ├── preprocessing.py         ← Transforms, Dataset class, HOG+LBP
│   ├── train.py                 ← ResNet18 CNN training loop
│   ├── features_clip.py         ← Frozen CLIP multi-view feature extraction
│   ├── train_head.py            ← Calibrated linear-head retraining
│   ├── score_clip.py            ← Exact train/serve-consistent CLIP scorer
│   ├── train_classical.py       ← 6 classical ML models
│   ├── evaluate.py              ← Full model evaluation + reports
│   ├── image_forensics.py       ← Metadata, C2PA, watermark, container and LSB checks
│   └── predict.py               ← Inference + Grad-CAM + Explanation
│
├── app/
│   ├── app.py                   ← Streamlit layout (tabs, inputs, renderers)
│   ├── inference.py             ← Model registry, cached predictors, measured metrics
│   └── theme.py                 ← Design tokens + HTML component helpers
│
├── .streamlit/
│   └── config.toml              ← Dark theme for native Streamlit widgets
│
├── notebooks/
│   ├── 01_EDA.py
│   ├── 02_Classical_ML.py
│   ├── 03_CNN_Training.py
│   └── 04_GradCAM_Demo.py
│
├── diag_benchmark.py            ← Benchmarks every model + per-resolution tiers
├── diag_forensics.py            ← Measures FFT/ELA signal strength alone
├── diag_app_paths.py            ← Smoke-tests all 8 models × 2 images
├── models/                      ← Saved model checkpoints (.pth / .pkl)
├── results/                     ← Plots, metrics, Grad-CAM outputs
└── tests/                       ← pytest unit tests
```

### CNN Architecture

```
ResNet18 (ImageNet pretrained)
  │
  ├── layer1 → layer2 → layer3 → layer4  [FROZEN in Phase 1]
  │                                       [UNFROZEN in Phase 2]
  └── Custom Classifier Head:
        Dropout(0.4)
        Linear(512 → 256)
        ReLU
        Dropout(0.2)
        Linear(256 → 2)   ← [REAL, FAKE]
```

### Grad-CAM Pipeline

```
Input Image
    ↓
ResNet18 Forward Pass
    ↓  (hooks capture feature maps at layer4)
Backward pass (class score gradient)
    ↓
GAP of gradients → channel weights
    ↓
Weighted sum of feature maps → ReLU
    ↓
Upsample to original size
    ↓
JET colormap overlay
    ↓
Heatmap Image
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Mevinb/proj-v1.git
cd truthlens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option A: Kaggle API (fastest)
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
python dataset/download_cifake.py --method kaggle

# Option B: Manual download
python dataset/download_cifake.py --method manual
# Follow the printed instructions

# Verify dataset
python dataset/download_cifake.py --method info
```

For a broader corpus spanning real photographs, GANs, latent diffusion, pixel
diffusion, and newer generators, use the single resumable corpus builder:

```bash
python dataset/download_corpus.py \
  --per-architecture 10000 \
  --highres-per-class 10000 \
  --output-dir datasets/prepared/corpus
```

The modern So-Fake-OOD source is very slow over HTTP range reads. Collect it in
small resumable targets rather than committing to the full 3,000-image target
in one command:

```bash
python dataset/collect_modern_generators.py --sources sofake --target 250
# Later:
python dataset/collect_modern_generators.py --sources sofake --target 500
```

The builder streams high-resolution public Hugging Face sources, rejects source
images smaller than 512px on either edge, and resizes only after that quality gate.
CIFAKE is excluded by default because its 32x32 images are only useful as a
legacy low-resolution stress set; add `--cifake-per-class N` if that stress
test is specifically required. The builder then
deduplicates by pixel hash, excludes NSFW-flagged rows when the source exposes
that field, and resumes from `manifest.csv`. It writes `sources.json` with
source URLs and licenses. Review each upstream dataset card before
redistribution: licenses and generator labels are source-specific. The output
can be used with `--data-dir datasets/prepared/corpus`.

### 3. Train Models

```bash
# Train CNN (ResNet18) — ~15 min GPU / ~2 hrs CPU
# --data-dir defaults to datasets/prepared/cifake (32px CIFAKE); pass the multi-resolution
# corpus to reproduce the shipped checkpoint.
python src/train.py --data-dir datasets/prepared/multires

# Train all 6 classical models
python src/train_classical.py

# Quick test run with 2000 samples per class
python src/train_classical.py --max-samples 2000
```

### 4. Run Web App

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Dataset Setup

The shipped checkpoints were trained on **`datasets/prepared/multires`** — a balanced
three-tier corpus built by `dataset/build_multires_dataset.py`, so the detector
sees every resolution regime rather than only 32px thumbnails:

| Tier | Resolution | REAL source | FAKE source |
|---|---|---|---|
| High-res | 512–2048px | DSLR / phone photographs | DALL·E 3, SDXL, Midjourney |
| Medium-res | 256px | multi-domain photo archive | multi-domain synthetic archive |
| Low-res | 32–64px | CIFAR-10 | Stable Diffusion v1.4 (CIFAKE) |

| Split | REAL | FAKE | Total |
|---|---|---|---|
| Train | 10,480 | 10,480 | 20,960 |
| Val   | 1,310  | 1,310  | 2,620  |
| Test  | 1,310  | 1,310  | 2,620  |

```bash
python dataset/build_multires_dataset.py
```

**CIFAKE** on its own is still supported as a quick-start / low-resolution stress
set — 60,000 real (CIFAR-10) + 60,000 fake (Stable Diffusion v1.4) at 32×32:

```bash
python dataset/download_cifake.py --method kaggle
```

**Source:** [Kaggle — CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

A model trained on CIFAKE alone will not transfer to full-resolution images;
that resolution gap is why the multi-resolution corpus exists.

---

## 🎓 Training

### ResNet18 CNN

```bash
python src/train.py \
  --data-dir datasets/prepared/multires \
  --epochs 20 \
  --batch-size 64 \
  --lr-head 1e-3 \
  --lr-tune 1e-4 \
  --freeze-epochs 5 \
  --patience 5
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `datasets/prepared/cifake` | Corpus root; use `datasets/prepared/multires` for the multi-resolution set |
| `--epochs` | 20 | Total training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr-head` | 1e-3 | LR for head-only phase |
| `--lr-tune` | 1e-4 | LR for fine-tuning phase |
| `--freeze-epochs` | 5 | Epochs to freeze backbone |
| `--patience` | 5 | Early stopping patience |

A `val/` split is mandatory — training aborts if it is missing rather than
silently validating on shuffled, augmented training data (which would make early
stopping and best-checkpoint selection meaningless).

The best checkpoint is selected by **lowest validation loss**. Note that the
shipped `models/resnet18_truthlens.pth` comes from epoch 1 with
`--freeze-epochs 1`: val_loss reached 0.2707 there and was never beaten, even
though val_acc later peaked at 91.83% (epoch 6) versus 91.49% at epoch 1.
Retraining with `--freeze-epochs 5` as above gives the head a proper warm-up
before the backbone unfreezes, and is likely to do better.

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

### Classical ML

```bash
python src/train_classical.py
python src/train_classical.py --max-samples 5000   # quick test
```

### Retrain the default CLIP detector

The default app model keeps CLIP frozen and retrains only linear heads. This is
intentional: fine-tuning the old ResNet18 improved same-source scores while
destroying the signal on the hardest unseen-source set.

```bash
# Run once after the corpus changes (GPU recommended).
python -m src.features_clip --corpus datasets/prepared/modern_v2

# Cheap retraining step; uses the cached features and completes in seconds.
python -m src.train_head \
  --cache features/clip_vitl14 \
  --out models/clip_linear \
  --eval-splits test holdout \
  --tag clip_linear
```

The app reads `models/clip_linear/heads.joblib` and uses the same re-encode and
view-rendering functions as feature extraction, preventing train/serve skew.

For a controlled source-adaptation experiment, without consuming the entire
hard holdout:

```bash
python -m src.adapt_head \
  --out models/clip_linear_adapted \
  --n-per-class 80
```

This reserves 20 paired real and 20 paired GPT-Image-2 examples as an untouched
evaluation slice. The adapted candidate improves that slice substantially, but
it is **not automatically promoted to the app**: the current candidate also
increases false alarms on independent web photographs. A model must improve
both panels before replacing the default.

### Fine-tune the legacy ResNet18 for multiple AI generators

No image detector can guarantee catching every current or future AI image.
Generators, editing tools, resizing, screenshots, and compression change over
time. The reliable approach is to train on a diverse generator dataset and
measure recall separately for generators that were not used during training.

Collect representative outputs from Gemini, Imagen, DALL-E, Midjourney, Stable
Diffusion, FLUX, and other generators in one fake folder. Put real images from
the same subject domains in another folder. Keep a separate holdout set for
final testing.

Prepare a training dataset:

```bash
python dataset/prepare_custom_dataset.py \
  --fake-dir /path/to/all_ai_images \
  --real-dir /path/to/real_images \
  --output-dir datasets/prepared/gemini
```

#### GPT-Image and Nano-Banana images

The generator images added under `datasets/raw/generator_sources/` can be prepared with the same
`train/val/test/{real,fake}` layout without copying the GPT-Image files.
Nano-Banana parquet rows are extracted into `datasets/raw/generator_sources/_cache/nano-banana` on first
use, and the prepared dataset uses symlinks plus a provenance manifest:

```bash
.venv/bin/python dataset/prepare_generator_dataset.py \
  --fake-root datasets/raw/generator_sources \
  --real-root datasets/prepared/modern_v2 \
  --output-dir datasets/prepared/gpt_nano
```

This creates `datasets/prepared/gpt_nano/{train,val,test}/{real,fake}` and
`datasets/prepared/gpt_nano/manifest.csv`. Train and evaluate the existing model on the
new corpus with:

```bash
.venv/bin/python src/train.py --data-dir datasets/prepared/gpt_nano
.venv/bin/python src/evaluate.py --model all --data-dir datasets/prepared/gpt_nano
```

The default split seed is `42`; pass `--seed` to reproduce a different split.

The fake folder may contain generator subfolders, for example
`all_ai_images/gemini`, `all_ai_images/dalle`, and `all_ai_images/flux`.
Fine-tune from the existing multi-resolution checkpoint. Use more total epochs
than frozen epochs so the backbone is actually updated:

```bash
python src/train.py \
  --data-dir datasets/prepared/gemini \
  --model-name resnet18_gemini.pth \
  --initial-checkpoint models/resnet18_truthlens.pth \
  --epochs 15 \
  --freeze-epochs 3 \
  --batch-size 32
```

Evaluate on the held-out custom test split before using the checkpoint in the
app. Do not train on the same Gemini images used to judge performance. To serve a
new checkpoint, point `Config.cnn_model_name` at it — the app's model selector
lists a single CNN entry and loads whichever checkpoint is configured, rather
than offering per-variant options that all resolve to the same file.

To inspect per-generator fake recall:

```bash
python src/evaluate_generators.py \
  --fake-root /path/to/heldout_ai_images \
  --model-name resnet18_gemini.pth
```

---

## 📊 Evaluation

```bash
# Evaluate all models
python src/evaluate.py --model all

# Evaluate specific model
python src/evaluate.py --model resnet18
python src/evaluate.py --model svm
python src/evaluate.py --model rf
```

---

## 🌐 Web Application

```bash
streamlit run app/app.py
```

Three tabs — **Detect**, **Benchmarks**, **About** — split across three files so
the layout code stays separate from the model code:

| File | Role |
|---|---|
| `app/app.py` | Layout only. Tabs, inputs, and the render functions. |
| `app/inference.py` | Model layer. Registry, cached predictors, and every measured number the UI shows. |
| `app/theme.py` | Design tokens and HTML component helpers (verdict card, split bar, notes, stats). |
| `.streamlit/config.toml` | Dark theme, so native widgets theme themselves instead of being overridden with `!important`. |

What it does:
- **Upload one or many** JPG/PNG/WebP/BMP images, or click a held-out **sample**
  from `datasets/prepared/multires/test/` — samples carry their true label, so the app can be checked
  rather than believed
- **Verdict card** with the probability split, and the accuracy *for that
  image's resolution tier* rather than the flattering aggregate
- **Inconclusive routing** when the calibrated score is not extreme or the four
  CLIP views disagree. The underlying lean remains visible, but the app no
  longer presents every upload as a trustworthy forced binary answer.
- **Grad-CAM** overlay from layer3+layer4, backpropagated from the predicted
  class (CNN only — the toggle is disabled for classical models)
- **2D FFT and ELA panels**, each shown against measured per-class reference
  ranges. Both are labelled descriptive and neither moves the verdict: alone
  they reach AUC 0.562 and 0.514 against a 0.500 chance line.
- **Provenance & watermark panel** that inspects the original encoded file for
  EXIF/XMP and PNG text, explicit AI-generator/editor tags, camera fields,
  C2PA/JUMBF container markers, ICC/alpha structure, trailing payloads, possible
  visible border text/logos, watermark identifiers (including SynthID strings),
  hidden colour under transparent pixels, and weak LSB/steganography statistics.
  SHA-256 and perceptual dHash fingerprints are included for matching against a
  separate known-image database.
  These findings are reported separately and never silently change the model
  verdict. Proprietary invisible watermarks still require the provider's
  official detector or verification key for confirmation.
- **Batch summary** table with CSV export, and per-result JSON download
- **Checkpoint provenance** in the sidebar (file, epoch, val accuracy, device),
  so a stale checkpoint is visible instead of implied

ResNet/classical figures in the UI are read from
`results/metrics/benchmark_test_split.json`; CLIP figures are read from
`results/metrics/clip_linear.json`. Re-running the corresponding evaluation or
head-training command updates the displayed measurements.

---

## 📈 Model Performance

The default CLIP model was retrained on August 17, 2026, from 12,951 cached
training rows and 1,660 validation rows. The two most relevant measured panels
are deliberately kept separate:

| Panel | Images | Accuracy / balanced accuracy | AUC | REAL acc | FAKE acc |
|---|---:|---:|---:|---:|---:|
| Modern same-source test | 1,694 | 97.99% accuracy | 0.9966 | 98.39% | 97.50% |
| Quarantined source/generator holdout | 600 | 90.8% balanced @ 0.5 | 0.9648 | — | — |

The same-source number is a regression check, not a promise for arbitrary web
uploads. See `RETRAIN_PLAN.md` and `src/eval_protocol.py` for the independent
source, unseen-generator, and wild-web tiers.

Measured on the held-out `datasets/prepared/multires/test` split by `diag_benchmark.py`.
The CNN is scored on all 2,620 test images; the classical models on a 600-image
stratified subsample (HOG+LBP extraction is the bottleneck). Reproduce with:

```bash
python diag_benchmark.py          # writes results/metrics/benchmark_test_split.json

# src/evaluate.py defaults to --data-dir datasets/prepared/cifake (the 32px CIFAKE split), so
# point it at the corpus the shipped checkpoint was actually trained on:
python src/evaluate.py --model all --data-dir datasets/prepared/multires
```

| Model | Accuracy | F1 | AUC | REAL acc | FAKE acc | Notes |
|---|---|---|---|---|---|---|
| **CLIP ViT-L/14 + linear heads** | **97.99%** | **0.978** | **0.997** | **98.4%** | **97.5%** | App default; modern test |
| ResNet18 (CNN + TTA) | 92.79% | 0.928 | 0.983 | 92.4% | 93.1% | Legacy baseline + Grad-CAM |
| ResNet18 (single crop) | 92.52% | 0.926 | 0.982 | 91.8% | 93.3% | Bare checkpoint |
| Logistic Regression | 56.67% | 0.467 | 0.585 | 75.3% | 38.0% | Best classical |
| Random Forest | 55.50% | 0.369 | 0.568 | 85.0% | 26.0% | |
| SVM (RBF) | 55.33% | 0.402 | 0.569 | 80.7% | 30.0% | |
| k-NN | 54.50% | 0.312 | 0.549 | 88.3% | 20.7% | |
| Decision Tree | 54.00% | 0.509 | 0.557 | 60.3% | 47.7% | |
| Naive Bayes | 51.33% | 0.305 | 0.518 | 81.3% | 21.3% | |

> **Chance is 50%.** HOG+LBP texture statistics do not separate modern
> generators from real photographs on this corpus — every classical model
> defaults heavily toward REAL and catches only 21–48% of fakes. They are kept
> for comparison, not for use. Only the selected learned model produces a
> verdict; the FFT and ELA
> panels in the app are descriptive and never alter it (measured alone they reach
> AUC 0.562 and 0.514, and blending them into the CNN score lowers AUC
> monotonically — see `diag_forensics.py`).

> Earlier revisions of this table quoted CIFAKE-era figures (ResNet18 ~95%,
> Random Forest ~78%, SVM ~76%). Those never held on `datasets/prepared/multires`.

### Accuracy by resolution tier

The 92.79% headline blends three tiers of very different difficulty, and the
blend flatters the one that matters most. `datasets/prepared/multires` is built in tiers,
so the CNN can be scored inside each:

| Resolution tier | Accuracy | F1 | AUC | REAL acc | FAKE acc | Images |
|---|---|---|---|---|---|---|
| High-res (512px+) | **88.50%** | 0.876 | 0.961 | 90.7% | 86.0% | 774 |
| Medium (256px) | 94.51% | 0.940 | 0.989 | 94.8% | 94.1% | 1,111 |
| Low-res (32px) | 93.96% | 0.938 | 0.989 | 90.0% | 98.5% | 563 |

A real upload — a photograph off a phone, an SDXL or Midjourney render — is
almost always in the high-resolution tier, which scores **4.3 points below the
headline** and misses **14% of AI images**. The app detects the uploaded image's
tier from its pixel dimensions and quotes that tier's accuracy on the result
page instead of the aggregate. Reproduced by `score_by_tier()` in
`diag_benchmark.py`; the numbers land in the `by_resolution` block of
`results/metrics/benchmark_test_split.json`, which is what the UI reads.

---

## 🔬 Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_EDA.py` | Dataset statistics, class balance, pixel histograms, FFT analysis |
| `notebooks/02_Classical_ML.py` | Feature extraction walkthrough, classical model training |
| `notebooks/03_CNN_Training.py` | ResNet18 training, loss/accuracy curves |
| `notebooks/04_GradCAM_Demo.py` | Grad-CAM visualisation on real and fake images |

Run notebooks:
```bash
python notebooks/01_EDA.py
python notebooks/02_Classical_ML.py
python notebooks/03_CNN_Training.py
python notebooks/04_GradCAM_Demo.py --image path/to/image.jpg
```

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

Tests cover:
- Image transforms (train/val/test)
- HOG + LBP feature extraction
- Single-image preprocessor
- Grad-CAM overlay generation
- Generator detection
- Explanation generation
- Config serialisation

---

## 🔌 API Reference

### Single Image Prediction

```python
from src.predict import predict_image
from src.utils import Config
from PIL import Image

cfg = Config()
img = Image.open("image.jpg")

result, heatmap = predict_image(
    image_source = img,
    cfg          = cfg,
    model_type   = "resnet18",   # or "ensemble", "svm", "random_forest", ...
    with_gradcam = True,
)

print(result["prediction"])        # "REAL" or "FAKE"
print(result["confidence"])        # e.g. 97.3
print(result["probabilities"])     # {"REAL": 2.7, "FAKE": 97.3}
print(result["likely_generator"])  # heuristic guess, or "N/A" when REAL
print(result["explanation"])       # list of explanation strings
print(result["spectral_analysis"]) # 2D FFT metrics — display only
print(result["ela_metrics"])       # Error Level Analysis metrics — display only
print(result["provenance_analysis"]) # Metadata, credentials, watermarks, container
```

`predict_image` is the single inference path: `app/app.py`, the
`python src/predict.py --image ...` CLI, and the tests all go through it, so the
result schema cannot drift between them. Predictors are cached per
`(model_type, checkpoint)`; call `src.predict.clear_predictor_cache()` after
retraining or swapping a checkpoint in a long-lived process.

`spectral_analysis` and `ela_metrics` are attached for display and never affect
the verdict — the CNN (or the selected classical model) decides alone.
`model_type="ensemble"` adds an `ensemble_votes` dict showing what the classical,
FFT and ELA signals would have said, labelled advisory; the verdict still comes
from the CNN.

`provenance_analysis` is also advisory. Explicit generator metadata is strong
evidence about the file's history, but metadata can be forged or stripped.
C2PA/JUMBF detection currently reports the presence of a container marker; it
does not perform certificate-chain or signature validation. Visible-watermark
and LSB/invisible-watermark scores are heuristics, not proof.

Heatmaps are returned for `"resnet18"` and `"ensemble"` only; classical models
return `None`.

### Feature Extraction

```python
from src.preprocessing import extract_features
import numpy as np

img    = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
features = extract_features(img)   # HOG + LBP → numpy array
```

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Deep Learning | PyTorch 2.13, TorchVision |
| Classical ML | scikit-learn 1.9 |
| Image Processing | OpenCV 5.0, Pillow 12 |
| Feature Extraction | scikit-image 0.26 (HOG, LBP) |
| Web App | Streamlit 1.61 |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Data | NumPy 2.5, Pandas |
| Logging | TensorBoard |
| Testing | pytest, pytest-cov |

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with ❤️ for the TruthLens AI Image Detection Project
</div>
