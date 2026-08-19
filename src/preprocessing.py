#!/usr/bin/env python3
"""
TruthLens — src/preprocessing.py
==================================
Dataset classes, transforms, and feature extraction for both:
  • Deep Learning  → PyTorch Dataset + DataLoaders
  • Classical ML   → HOG + LBP feature vectors
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from src.utils import Config, get_logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

logger = get_logger(__name__)


# ─── ImageNet normalisation constants ─────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─── Provenance-shortcut normalisation ────────────────────────────────────────
class RandomRecompress:
    """Re-encode through JPEG at a random quality.

    Applied to **both** classes, which is the whole point. ``datasets/prepared/modern_v2``
    stores original bytes, so container format correlates with class purely by
    where each source came from: the GPT-Image sets are pristine PNG, the
    Megalith and OpenImages photo sets are JPEG. A model trained on that learns
    "no JPEG artifacts ⇒ fake", reports high accuracy offline, and then fails on
    a real photo saved as PNG or an AI image posted to a platform that
    recompressed it.

    Why the default is ``p=1.0`` and not a coin flip
    ------------------------------------------------
    Leaving half the images untouched does not remove the leak, it halves it:
    pristine input would still occur only in the fake class, and "pristine"
    survives ``ToTensor`` as the absence of 8x8 blocking. Re-encoding *every*
    image puts both classes on the same footing — the fakes become
    single-generation JPEG, the reals double-generation, and at the top of the
    quality range that difference is close to a no-op.

    The range is deliberately wide and open at the top. q98 is nearly lossless,
    so it stands in for the pristine PNG a user actually uploads straight out of
    ChatGPT; q40 covers what a messaging app does to the same image. Training
    across the whole span keeps both ends in distribution instead of trading one
    failure for the other. The generator fingerprint survives moderate JPEG,
    which is what makes this affordable.

    Where it sits in the pipeline
    -----------------------------
    **After** the crop, not before. The leak closes either way — all that is
    required is that both classes pass through the same encoder — but before the
    crop this re-encodes and re-decodes at full source resolution, which on the
    hi-res reals means a 60 MP round trip to produce a 224x224 tensor. Measured
    at 61 ms/image, 52% of the total preprocessing cost for that source, enough
    to hold GPU utilisation at 0%. A 1:1 native crop preserves pixel scale, so
    the JPEG 8x8 grid lands at the same scale relative to image texture in both
    orders; the whole-frame branch instead gets the grid at 224 rather than at
    source scale, a mild realism mismatch on the branch that carries the least
    evidence anyway.

    ``min_side`` is therefore vestigial for the modern pipeline: ``NativeScaleCrop``
    and ``CenterNativeCrop`` both return exactly ``img_size`` square, so nothing
    that reaches this class is ever below the threshold. It is kept for direct
    callers, and the original reasoning no longer bites — the 32x32 CIFAKE tier
    used to be exempt because q40 across sixteen macroblocks deletes the picture
    rather than marking it, but post-crop those images arrive as bicubic
    upsamples to 224 with no high-frequency content left to destroy, so q40 is
    close to a no-op on them. The tier gains consistent treatment instead of an
    exemption.
    """

    def __init__(
        self,
        p: float = 1.0,
        quality: Tuple[int, int] = (40, 98),
        min_side: int = 96,
    ):
        self.p = p
        self.q_lo, self.q_hi = quality
        self.min_side = min_side

    def __call__(self, img: "Image.Image") -> "Image.Image":
        import random
        from io import BytesIO

        if random.random() >= self.p or min(img.size) < self.min_side:
            return img
        buf = BytesIO()
        img.convert("RGB").save(
            buf, format="JPEG", quality=random.randint(self.q_lo, self.q_hi)
        )
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomAspectPad:
    """Letterbox to a random aspect ratio, with some probability.

    Guards the second confound the corpus audit found: GPT-Image output is
    portrait-or-square (1024x1536, 1024x1024) while Flickr photographs are
    mostly 3:2 landscape. ``Resize((224, 224))`` squashes non-square input, so
    the *degree of squash* is itself a class signal — the model can read
    orientation off the distortion without ever looking at a texture.

    Randomly re-letterboxing both classes decouples orientation from label.

    Why the pad is bounded
    ----------------------
    Hitting the target ratio exactly means growing the short side without limit,
    and on a wide high-resolution frame that explodes. The corpus holds a
    11364x5682 camera original (64.6 MP); asking for 0.667 wants an
    11364x17037 canvas — 193.6 MP, past Pillow's 179 MP decompression-bomb
    ceiling, which is a raised exception rather than a slow path. That killed a
    training run at epoch 2 once ``real_hires_camera`` put genuine 20-60 MP
    originals in the corpus.

    ``max_growth`` caps the padded area at a multiple of the original, so the
    ratio is approached rather than reached on extreme input. The default is
    just above 2.25 for a specific reason: letterboxing only ever *grows* the
    short side, so turning the corpus's typical 3:2 landscape photograph into
    the 2:3 portrait that GPT-Image emits costs exactly 2.25x the area. Cap
    below that and landscape input can never reach the portrait ratio, which
    leaves "ratio 0.667 ⇒ fake" intact — a tighter bound would have quietly
    reintroduced the very confound this class exists to remove. Measured over
    4000 draws with the default, a 3:2 real and a 2:3 fake both span
    {0.667, 1.0, 1.5}; capped at 1.5 the real only reached {1.0, 1.5}.

    ``max_pixels`` skips the pad entirely above a ceiling. Past that size the
    canvas allocate-and-paste costs more dataloader time than the augmentation
    returns, and the handful of images involved cannot shift a population
    statistic. Padding is a no-op on the native-crop branch downstream anyway —
    only the whole-frame branch ever sees the squash this exists to disrupt.
    """

    def __init__(
        self,
        p: float = 0.3,
        ratios: Tuple[float, ...] = (0.667, 1.0, 1.5),
        max_growth: float = 2.3,
        max_pixels: int = 24_000_000,
    ):
        self.p = p
        self.ratios = ratios
        self.max_growth = max_growth
        self.max_pixels = max_pixels

    def __call__(self, img: "Image.Image") -> "Image.Image":
        import random

        if random.random() >= self.p:
            return img
        w, h = img.size
        if w * h > self.max_pixels:
            return img

        target = random.choice(self.ratios)
        budget = int(w * h * self.max_growth)
        if w / h > target:                       # too wide → pad top/bottom
            new_h = min(max(h, int(round(w / target))), max(1, budget // max(w, 1)))
            canvas = Image.new("RGB", (w, new_h), (0, 0, 0))
            canvas.paste(img, (0, (new_h - h) // 2))
        else:                                    # too tall → pad left/right
            new_w = min(max(w, int(round(h * target))), max(1, budget // max(h, 1)))
            canvas = Image.new("RGB", (new_w, h), (0, 0, 0))
            canvas.paste(img, ((new_w - w) // 2, 0))
        return canvas


class NativeScaleCrop:
    """Sample a view at (or near) the image's own pixel scale, not the whole frame.

    This is the single biggest reason a 1024x1536 ChatGPT image scores like a
    photograph. ``Resize((224, 224))`` averages roughly 7x7 source pixels into
    every output pixel, and the evidence that separates a neural image decoder
    from a camera sensor lives almost entirely in that high-frequency band —
    demosaic and shot-noise structure on one side, upsampler and VAE
    reconstruction artifacts on the other. Downscale the frame and what survives
    is composition, colour and anatomy, which modern generators already get
    right. The model is then being asked to judge on exactly the evidence that
    has been thrown away.

    Cropping at 1:1 keeps that band. The cost is field of view: a 224 crop of a
    1536px image covers about 3% of it, so global cues — a hand with six fingers,
    impossible reflections — fall outside the view. Both cues are real, so this
    transform samples both: with probability ``p_native`` a native-scale crop,
    otherwise the whole frame downscaled as before. The model sees each image
    both ways across epochs and can use whichever generalises.

    ``max_downscale`` widens the native branch to crops of up to ``size *
    max_downscale`` pixels scaled down to ``size``, so the model spans a range of
    scales rather than memorising one. Images too small to crop meaningfully —
    CIFAKE is 32x32 — have no native scale to preserve and are upscaled to
    ``size`` exactly as the legacy pipeline did.

    ``frame_short_range`` — why the whole-frame branch needs its own fix
    -------------------------------------------------------------------
    The native branch is already scale-blind: a 224 crop taken at 1:1 looks the
    same whether it came from a 600px photo or a 1536px generation, so the
    source's resolution cannot be read off it. The *whole-frame* branch is not.
    Squashing a 600px image to 224 averages about 2.7x2.7 source pixels per
    output pixel; squashing a 1536px image averages 6.9x6.9. More averaging means
    less residual noise, so "how smooth is this downscaled frame" is a direct
    read-out of source resolution — and the corpus audit measured resolution as
    the corpus's strongest leak (768-1400px short side ran 0.82 fake, 1-2MP ran
    0.99 fake, because every modern generator emits 1024px+ while the Flickr
    photo sets are capped near 768).

    Resizing the short side to a *common random target* before the squash
    equalises that. Both classes then end on the same "target → 224" step, so the
    averaging factor no longer carries the label. It costs nothing the model
    needs: the high-frequency evidence the native branch exists to preserve is
    untouched, because this only ever runs on the branch that was throwing that
    evidence away anyway.

    This is not a complete fix on its own. An image whose short side is already
    below the top of the range can never be downscaled *up* into the rest of it,
    so a corpus where only the fakes exceed 768px keeps a residual cue at the top
    of the range. Closing that needs real photographs in the same resolution band
    as the generators — augmentation equalises the population that overlaps, the
    corpus has to supply the overlap.
    """

    def __init__(
        self,
        size: int = 224,
        p_native: float = 0.5,
        max_downscale: float = 2.0,
        frame_short_range: Optional[Tuple[int, int]] = (288, 768),
    ):
        self.size = size
        self.p_native = p_native
        self.max_downscale = max_downscale
        self.frame_short_range = frame_short_range

    def _whole_frame(self, img: "Image.Image") -> "Image.Image":
        """Whole-frame view with the downscale factor equalised across classes."""
        import random

        if self.frame_short_range:
            lo, hi = self.frame_short_range
            w, h = img.size
            short = min(w, h)
            if short > lo:
                target = random.randint(lo, min(hi, short))
                if target < short:
                    scale = target / short
                    img = img.resize((max(1, round(w * scale)),
                                      max(1, round(h * scale))), Image.BICUBIC)
        return img.resize((self.size, self.size), Image.BICUBIC)

    def __call__(self, img: "Image.Image") -> "Image.Image":
        import random

        w, h = img.size
        short = min(w, h)

        # Too small for a native crop to mean anything → legacy behaviour.
        if short < self.size * 1.25:
            return img.resize((self.size, self.size), Image.BICUBIC)

        if random.random() >= self.p_native:
            return self._whole_frame(img)

        box = int(round(self.size * random.uniform(1.0, self.max_downscale)))
        box = min(box, short)
        x = random.randint(0, w - box)
        y = random.randint(0, h - box)
        crop = img.crop((x, y, x + box, y + box))
        if box != self.size:
            crop = crop.resize((self.size, self.size), Image.BICUBIC)
        return crop


class CenterNativeCrop:
    """Deterministic counterpart to :class:`NativeScaleCrop` for val/test.

    Centre crop at 1:1 pixel scale when the image is large enough, plain resize
    when it is not. Deterministic so validation loss is comparable across epochs.
    """

    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, img: "Image.Image") -> "Image.Image":
        w, h = img.size
        if min(w, h) < self.size * 1.25:
            return img.resize((self.size, self.size), Image.BICUBIC)
        left = (w - self.size) // 2
        top = (h - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int = 224, modern: bool = False):
    """
    Return augmentation pipeline for train / val+test splits.

    Train augmentations are intentionally strong because CIFAKE images are
    32×32 upsampled to 224×224 — extra augmentation helps prevent overfitting.

    ``modern=True`` selects the pipeline for ``datasets/prepared/modern_v2``, which differs
    in three ways, in descending order of how much they matter:

    * ``NativeScaleCrop`` stops throwing away the evidence. Resizing a megapixel
      generation to 224 destroys the high-frequency band the verdict depends on;
      see that class for the full argument.
    * ``RandomRecompress`` / ``RandomAspectPad`` break the format and
      orientation shortcuts described on those classes. Without them the model
      can separate the modern corpus without learning anything transferable.
    * The heavy photometric augmentation is dialled back. Aggressive
      ``ColorJitter``, blur and grayscale were tuned for 32x32 CIFAKE, where
      there is no fine texture left to protect. On 1024px+ images the signal
      that distinguishes GPT-Image output from a photograph *is* fine texture
      and colour response, and jittering it that hard erases the evidence.
    """
    from torchvision import transforms

    if split == "train" and modern:
        return transforms.Compose([
            RandomAspectPad(p=0.3),
            # Recompress AFTER the crop, not before. Both orders break the format
            # shortcut equally — the only requirement is that both classes pass
            # through the same encoder — but before the crop it re-encodes and
            # re-decodes at full source resolution, up to 60 MP, to produce a
            # 224x224 tensor. Measured at 61 ms/image on the hi-res reals, 52% of
            # their total preprocessing cost, which starved the GPU to 0%
            # utilisation. After the crop it is a 224x224 encode: the same
            # augmentation for ~1/100th of the work.
            NativeScaleCrop(img_size, p_native=0.5, max_downscale=2.0),
            RandomRecompress(p=1.0, quality=(40, 98)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        if modern:
            # Deterministic mirror of the native-crop branch above.
            #
            # A whole-frame resize here would make early stopping select the
            # checkpoint that best reads composition, which is the half of the
            # training distribution that does *not* generalise to a new
            # generator. Validating on the centre native crop selects for the
            # texture evidence instead. Small images have no native scale and
            # fall back to the plain resize.
            #
            # This is one view; the served path averages several. Headline
            # numbers come from src/evaluate.py and diag_modern_baseline.py,
            # which both go through predict_image and therefore through the
            # real TTA ensemble.
            #
            # The fixed-quality recompress is not augmentation — it is the same
            # leak-closing step the train branch applies, held still so val stays
            # reproducible. Without it the val path is the *only* place container
            # format reaches the model: datasets/prepared/modern_v2 stores original bytes,
            # so pristine PNG occurs almost exclusively in the fake class, and
            # early stopping would select whichever checkpoint best detects the
            # absence of 8x8 blocking. Measured with diag_shortcut_probe.py: ten
            # low-level statistics and a linear model separate the val classes at
            # 0.613 balanced accuracy with no recompress, against 0.537 on the
            # train pipeline that has it. Anything available to a linear probe on
            # ten features is available to the CNN far more cheaply than a
            # generator fingerprint, and it is exactly the cue that does not
            # survive contact with a real upload.
            #
            # q88 sits at the top of the train range's useful span: heavy enough
            # to plant a real blocking grid in both classes, light enough to
            # leave the high-frequency evidence the native crop exists to expose.
            #
            # Ordered crop-then-recompress to match the train branch. If val
            # recompressed the full frame and train recompressed the 224 crop,
            # the JPEG grid would land at a different pixel scale in each, and
            # early stopping would be selecting against a distribution the model
            # never trains on.
            return transforms.Compose([
                CenterNativeCrop(img_size),
                RandomRecompress(p=1.0, quality=(88, 88)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class CIFAKEDataset(Dataset):
    """
    Expects a directory tree:
        root/
          real/  *.jpg | *.png
          fake/  *.jpg | *.png

    Nested subdirectories under ``real/`` or ``fake/`` are also accepted, so
    prepared corpora may preserve generator-specific provenance folders.

    Labels: real → 0, fake → 1

    ``root`` may also be a list of such trees, which is how the modern corpus is
    trained alongside the older one. Combining at load time rather than copying
    files keeps each corpus independently inspectable — and re-collectable —
    instead of melting them into a third directory that has to be rebuilt from
    scratch whenever either input changes.
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(
        self,
        root: Path | str | Sequence[Path | str],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
        modern: bool = False,
    ) -> None:
        if isinstance(root, (str, Path)):
            self.roots = [Path(root)]
        else:
            self.roots = [Path(r) for r in root]
        self.root      = self.roots[0]          # kept for backward compatibility
        self.split     = split
        self.img_size  = img_size
        self.transform = transform or get_transforms(split, img_size, modern=modern)

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        class_map = {"real": 0, "fake": 1}
        per_root: Dict[str, int] = {}
        for root in self.roots:
            before = len(self.samples)
            for cls_name, label in class_map.items():
                cls_dir = root / cls_name
                if not cls_dir.exists():
                    logger.warning(f"Class directory not found: {cls_dir}")
                    continue
                for fpath in sorted(cls_dir.rglob("*")):
                    if fpath.is_file() and fpath.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((fpath, label))
            per_root[root.parent.name + "/" + root.name] = len(self.samples) - before

        if not self.samples:
            logger.warning(
                f"No images found in {[str(r) for r in self.roots]}. "
                "Did you run `python dataset/download_cifake.py --method kaggle`?"
            )
        else:
            n_real = sum(1 for _, lbl in self.samples if lbl == 0)
            n_fake = sum(1 for _, lbl in self.samples if lbl == 1)
            logger.info(
                f"[{self.split}] Loaded {len(self.samples)} images "
                f"(real={n_real}, fake={n_fake})"
            )
            if len(self.roots) > 1:
                for name, count in per_root.items():
                    logger.info(f"    {name}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", int]:
        """Decode and augment one sample, degrading rather than raising.

        A 20-epoch run over 34k megapixel images touches the transform pipeline
        about 700k times, so anything that can throw once will end the run — and
        it will do it 40 minutes in, not at startup. That happened for real:
        ``RandomAspectPad`` grew a 64 MP panorama past Pillow's decompression-bomb
        ceiling and killed training at epoch 2. The pad is bounded now, but the
        general shape of that bug (one pathological file, hours of GPU lost)
        is worth closing off structurally rather than one cause at a time.

        Two levels of degradation, in order of how much they preserve:

        1. Augmentation failed but the pixels are fine → plain resize. Keeps the
           real image and its real label, losing only the augmentation for this
           view.
        2. The file itself will not decode → move to another sample and return
           *its* image and *its* label. A consistent pair from elsewhere in the
           corpus is harmless; the previous behaviour returned a black frame
           still carrying the failed file's label, which teaches "black ⇒ this
           class" and is worse than dropping the sample.
        """
        for attempt in range(4):
            path, label = self.samples[idx]
            try:
                with Image.open(path) as handle:
                    img = handle.convert("RGB")
            except Exception as exc:                               # noqa: BLE001
                self._note_bad(path, f"decode failed: {exc}")
                idx = (idx + 1) % len(self.samples)
                continue

            try:
                return self.transform(img), label
            except Exception as exc:                               # noqa: BLE001
                self._note_bad(path, f"transform failed: {exc}")

            try:
                return self._plain(img), label
            except Exception as exc:                               # noqa: BLE001
                self._note_bad(path, f"fallback resize failed: {exc}")
                idx = (idx + 1) % len(self.samples)

        import torch
        logger.error("4 consecutive unusable samples near index %d — "
                     "returning a zero tensor to keep the run alive", idx)
        return torch.zeros(3, self.img_size, self.img_size), 0

    def _plain(self, img: "Image.Image") -> "torch.Tensor":
        """Un-augmented view: the transform pipeline's guaranteed-safe core."""
        from torchvision import transforms as T

        return T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])(img)

    def _note_bad(self, path, reason: str) -> None:
        """Log each bad file once per worker instead of once per epoch."""
        seen = getattr(self, "_bad_paths", None)
        if seen is None:
            seen = self._bad_paths = set()
        key = str(path)
        if key not in seen:
            seen.add(key)
            logger.warning("Unusable sample %s — %s", key, reason)

    @property
    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency class weights for imbalanced datasets."""
        import torch
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = 1.0 / (counts + 1e-8)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader Factory ───────────────────────────────────────────────────────
def get_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders from config.

    ``cfg.extra_data_dirs`` names further corpus roots to merge in, and
    ``cfg.modern_augment`` selects the shortcut-breaking train pipeline. Both
    default to the legacy behaviour so existing checkpoints stay reproducible.
    """
    import torch

    roots = [cfg.data_dir] + [Path(d) for d in getattr(cfg, "extra_data_dirs", ())]
    modern = bool(getattr(cfg, "modern_augment", False))

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dirs = [r / split for r in roots if (r / split).exists()]
        if not split_dirs:
            logger.warning(
                f"Split '{split}' missing under all of "
                f"{[str(r) for r in roots]} — skipping."
            )
            continue

        dataset = CIFAKEDataset(
            root=split_dirs,
            split=split,
            img_size=cfg.img_size,
            modern=modern,
        )

        if len(dataset) == 0:
            logger.warning(f"Empty dataset for split '{split}' — skipping.")
            continue

        shuffle     = split == "train"
        pin_memory  = torch.cuda.is_available()

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=split == "train",
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )
        logger.info(
            f"[{split}] DataLoader: {len(dataset)} samples, "
            f"{len(loaders[split])} batches"
        )

    return loaders


# ─── HOG Feature Extractor ────────────────────────────────────────────────────
def extract_hog_features(
    img_array: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) features."""
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


# ─── LBP Feature Extractor ────────────────────────────────────────────────────
def extract_lbp_features(
    img_array: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
    n_bins: int = 64,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract Local Binary Pattern (LBP) texture histogram."""
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)

    gray  = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lbp   = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ─── Combined Feature Extractor ───────────────────────────────────────────────
def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Combine HOG + LBP into a single feature vector."""
    hog_feat = extract_hog_features(img_array)
    lbp_feat = extract_lbp_features(img_array)
    return np.concatenate([hog_feat, lbp_feat])


# ─── Dataset Feature Extractor ────────────────────────────────────────────────
def build_feature_dataset(
    split_dir: Path,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk a split directory (real/ + fake/) and extract features for every image.
    Returns (X, y) numpy arrays ready for scikit-learn.
    """
    from tqdm import tqdm

    X_list, y_list = [], []
    class_map = {"real": 0, "fake": 1}

    for cls_name, label in class_map.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            logger.warning(f"[Feature Build] Missing: {cls_dir}")
            continue

        files = sorted(
            f for f in cls_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if max_samples_per_class:
            files = files[:max_samples_per_class]

        logger.info(f"  Extracting features from {cls_name} ({len(files)} images) ...")
        for fpath in tqdm(files, desc=f"{cls_name}", leave=False, ncols=80):
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                feat = extract_features(img)
                X_list.append(feat)
                y_list.append(label)
            except Exception as exc:
                logger.debug(f"  Skip {fpath.name}: {exc}")

    if not X_list:
        raise RuntimeError(
            f"No features extracted from {split_dir}. "
            "Check that the dataset is downloaded and organised correctly."
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─── Single Image Preprocessor ────────────────────────────────────────────────
def preprocess_single_image(
    image_source,
    img_size: int = 224,
    return_tensor: bool = True,
):
    """
    Preprocess a single image for CNN inference.

    Args:
        image_source : PIL.Image, numpy array, or file path string/Path.
        img_size     : Target size (default 224 for ResNet).
        return_tensor: If True, returns a torch.Tensor [1, 3, H, W].
                       If False, returns a numpy array [H, W, 3].
    """
    import torch

    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_source)}")

    transform = get_transforms("test", img_size)
    tensor    = transform(img)

    if return_tensor:
        return tensor.unsqueeze(0)  # [1, 3, H, W]
    else:
        return np.array(img.resize((img_size, img_size)))
