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
| 🧠 **ResNet18 CNN** | Transfer learning from ImageNet, ~95% accuracy on CIFAKE |
| 📐 **6 Classical ML Models** | LR, DT, RF, SVM, k-NN, Naive Bayes for comparison |
| 🔥 **Grad-CAM Heatmaps** | Visual explainability — shows where the network looked |
| 📊 **Confidence Scores** | Real-time probability breakdown (REAL % vs FAKE %) |
| 🎨 **Generator Detection** | Heuristic identification of Stable Diffusion, Midjourney, FLUX, DALL-E |
| 💡 **AI Explanation** | Rule-based bullet-point explanation of the prediction |
| 🌐 **Streamlit Web App** | Dark glassmorphism UI with drag-and-drop upload |
| 📈 **Model Comparison** | Leaderboard with accuracy, F1, AUC across all models |

---

## 🏗️ Architecture

```
TruthLens/
│
├── dataset/
│   ├── download_cifake.py       ← Automated dataset downloader
│   ├── train/  {real/, fake/}
│   ├── val/    {real/, fake/}
│   └── test/   {real/, fake/}
│
├── src/
│   ├── utils.py                 ← Config, logging, plotting helpers
│   ├── preprocessing.py         ← Transforms, Dataset class, HOG+LBP
│   ├── train.py                 ← ResNet18 CNN training loop
│   ├── train_classical.py       ← 6 classical ML models
│   ├── evaluate.py              ← Full model evaluation + reports
│   └── predict.py               ← Inference + Grad-CAM + Explanation
│
├── app/
│   └── app.py                   ← Streamlit web application
│
├── notebooks/
│   ├── 01_EDA.py
│   ├── 02_Classical_ML.py
│   ├── 03_CNN_Training.py
│   └── 04_GradCAM_Demo.py
│
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
git clone https://github.com/your-username/truthlens.git
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

### 3. Train Models

```bash
# Train CNN (ResNet18) — ~15 min GPU / ~2 hrs CPU
python src/train.py

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

We use **CIFAKE** — a benchmark dataset for real vs AI-generated image classification.

| Split | REAL | FAKE | Total |
|---|---|---|---|
| Train | ~48,000 | ~48,000 | ~96,000 |
| Val   | ~6,000  | ~6,000  | ~12,000 |
| Test  | 10,000  | 10,000  | 20,000  |

**Source:** [Kaggle — CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Real images: CIFAR-10 (natural photographs)
Fake images: Stable Diffusion v1.4 (AI-generated equivalents)

---

## 🎓 Training

### ResNet18 CNN

```bash
python src/train.py \
  --epochs 20 \
  --batch-size 64 \
  --lr-head 1e-3 \
  --lr-tune 1e-4 \
  --freeze-epochs 5 \
  --patience 5
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 20 | Total training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr-head` | 1e-3 | LR for head-only phase |
| `--lr-tune` | 1e-4 | LR for fine-tuning phase |
| `--freeze-epochs` | 5 | Epochs to freeze backbone |
| `--patience` | 5 | Early stopping patience |

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

### Classical ML

```bash
python src/train_classical.py
python src/train_classical.py --max-samples 5000   # quick test
```

### Fine-tune for multiple AI generators

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
  --output-dir dataset_gemini
```

The fake folder may contain generator subfolders, for example
`all_ai_images/gemini`, `all_ai_images/dalle`, and `all_ai_images/flux`.
Fine-tune the existing high-resolution model. Use more total epochs than frozen
epochs so the backbone is actually updated:

```bash
python src/train.py \
  --data-dir dataset_gemini \
  --model-name resnet18_gemini.pth \
  --initial-checkpoint models/resnet18_highres.pth \
  --epochs 15 \
  --freeze-epochs 3 \
  --batch-size 32
```

Evaluate on the held-out custom test split before using the checkpoint in the
app. Do not train on the same Gemini images used to judge performance.

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

Features:
- **Upload** any JPG/PNG/WebP image
- **Select model** from sidebar (ResNet18 recommended)
- **View prediction** with animated confidence gauge
- **Toggle Grad-CAM** for visual explanation (ResNet18 only)
- **Read explanation** bullets describing the network's reasoning

---

## 📈 Model Performance

> Results on CIFAKE test set (10,000 real + 10,000 fake images)

| Model | Accuracy | F1 | AUC | Notes |
|---|---|---|---|---|
| **ResNet18 (CNN)** | **~95%** | **~0.95** | **~0.99** | Best overall |
| Random Forest | ~78% | ~0.78 | ~0.86 | Best classical |
| SVM (RBF) | ~76% | ~0.76 | ~0.84 | |
| Logistic Regression | ~71% | ~0.71 | ~0.78 | Fastest |
| k-NN | ~68% | ~0.68 | ~0.74 | |
| Decision Tree | ~65% | ~0.65 | ~0.65 | |
| Naive Bayes | ~62% | ~0.62 | ~0.68 | |

> **Note:** Actual results will vary based on your training run, hardware, and hyperparameters.

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
    model_type   = "resnet18",   # or "svm", "random_forest", etc.
    with_gradcam = True,
)

print(result["prediction"])       # "REAL" or "FAKE"
print(result["confidence"])       # e.g. 97.3
print(result["likely_generator"]) # e.g. "Stable Diffusion"
print(result["explanation"])      # list of explanation strings
```

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
| Deep Learning | PyTorch 2.1, TorchVision |
| Classical ML | scikit-learn 1.3 |
| Image Processing | OpenCV 4.8, Pillow 10 |
| Feature Extraction | scikit-image (HOG, LBP) |
| Web App | Streamlit 1.28 |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Data | NumPy, Pandas |
| Logging | TensorBoard |
| Testing | pytest, pytest-cov |

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with ❤️ for the TruthLens AI Image Detection Project
</div>
