"""Probe for a content-genre confound between the real and fake pools.

Four sampled ``Scam-AI/gpt-image-2`` training images turned out to be a game UI
mockup, a marketing landing-page sheet, an anime character sheet and a stylised
product shot — none of them photographs. The real pool is camera photography by
construction (flickr-cc0, open-images, hires-original). If that pattern holds at
scale then a classifier can reach high accuracy by answering "is this a
photograph?" instead of "is this generated?", and it would then fail on exactly
the case that matters: photorealistic AI output.

This measures genre with cheap, generator-agnostic statistics rather than with
the model itself, which would be circular:

``flat_frac``    fraction of pixels sitting in a locally uniform 3x3 patch. Flat
                 colour fills are the signature of UI, vector art and graphics;
                 sensor noise makes them near-zero in real photography.
``uniq_colors``  distinct colours after 5-bit-per-channel quantisation, scaled by
                 pixel count. Illustrations use restricted palettes.
``axis_edges``   share of gradient energy on strictly horizontal or vertical
                 lines. Panels, frames and text boxes are axis-aligned; nature
                 is not.
``sat_mean``     mean HSV saturation. Graphics are more saturated than photos.

The headline number is the AUC of each feature at separating the *real* pool from
the *fake* pool. An AUC near 0.5 means the pools are genre-matched and the model
cannot be exploiting genre. An AUC near 1.0 means the label is partly readable
off genre alone, and the corpus — not the architecture — is the thing to fix.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OUT_PATH = Path("results/metrics/content_genre_probe.json")
MAX_SIDE = 512          # analysis resolution; genre statistics are scale-robust


def genre_features(img: Image.Image) -> dict:
    """Cheap statistics that separate photographs from graphics.

    ``flat_frac`` is computed on a *native-resolution* centre crop, not on the
    downscaled frame: bilinear downsampling averages neighbouring pixels and so
    destroys the sensor noise that makes photographs non-flat, which would make
    photos and graphics look alike on exactly the feature meant to tell them
    apart. The remaining three features are compositional and scale-robust, so
    they are taken on the downscaled whole frame where they belong.
    """
    img = img.convert("RGB")

    # --- flat_frac on a native-resolution crop ----------------------------
    w, h = img.size
    c = min(256, w, h)
    crop = img.crop(((w - c) // 2, (h - c) // 2,
                     (w - c) // 2 + c, (h - c) // 2 + c))
    cg = np.asarray(crop, dtype=np.float32).mean(axis=2)

    def box3(x):
        p = np.pad(x, 1, mode="edge")
        out = np.zeros_like(x)
        for dy in range(3):
            for dx in range(3):
                out += p[dy:dy + x.shape[0], dx:dx + x.shape[1]]
        return out / 9.0

    mu = box3(cg)
    var = np.maximum(box3(cg * cg) - mu * mu, 0.0)
    flat_frac = float((np.sqrt(var) < 0.8).mean())

    # --- compositional features on the downscaled whole frame -------------
    small = img.copy()
    small.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    gray = a.mean(axis=2)

    # --- uniq_colors: palette richness ------------------------------------
    q = (a / 8.0).astype(np.uint8)                     # 5 bits per channel
    packed = (q[..., 0].astype(np.uint32) << 10) | \
             (q[..., 1].astype(np.uint32) << 5) | q[..., 2]
    uniq = float(np.unique(packed).size) / packed.size

    # --- axis_edges: axis-aligned gradient energy -------------------------
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=0)    # per-column energy
    gy = np.abs(np.diff(gray, axis=0)).mean(axis=1)    # per-row energy
    # A panel border is a column/row whose energy far exceeds the local median.
    def spike_share(v):
        if v.size < 8:
            return 0.0
        med = np.median(v) + 1e-6
        return float((v > 4.0 * med).mean())
    axis_edges = 0.5 * (spike_share(gx) + spike_share(gy))

    # --- sat_mean ---------------------------------------------------------
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat_mean = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())

    return {"flat_frac": flat_frac, "uniq_colors": uniq,
            "axis_edges": axis_edges, "sat_mean": sat_mean}


def auc(pos: list, neg: list) -> float:
    """Rank AUC — probability a random pos scores above a random neg."""
    if not pos or not neg:
        return float("nan")
    allv = np.concatenate([np.asarray(pos), np.asarray(neg)])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[:len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((rp - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="corpus roots with manifest.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--score", action="store_true",
                    help="also score every image with the CNN and report "
                         "accuracy by photographic-ness tercile")
    ap.add_argument("--model", default="resnet18")
    args = ap.parse_args()

    predict_image = cfg = None
    if args.score:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.utils import Config, resolve_cnn_checkpoint
        from src.predict import predict_image as _pi
        predict_image = _pi
        cfg = resolve_cnn_checkpoint(Config())

    feats: dict = defaultdict(lambda: defaultdict(list))   # source -> feat -> vals
    src_label: dict = {}
    scored: list = []      # (label, source, uniq_colors, flat_frac, p_fake)

    for root in args.roots:
        base = Path(root)
        mf = base / "manifest.jsonl"
        if not mf.exists():
            print(f"  ! {mf} missing, skipping")
            continue
        df = pd.read_json(mf, lines=True)
        df = df[df.split == args.split]
        if not len(df):
            print(f"  ! {root}: no rows in split={args.split}")
            continue
        key = "generator" if "generator" in df.columns else "source_key"
        rows = (df.sort_values("sha256")
                  .groupby(key, group_keys=False)
                  .head(args.per_source))
        print(f"  {root}: {len(rows)} images across {rows[key].nunique()} sources")
        for r in rows.itertuples():
            src = f"{getattr(r, key)}"
            try:
                img = Image.open(base / r.path)
                f = genre_features(img)
            except Exception:
                continue
            src_label[src] = r.label
            for k, v in f.items():
                feats[src][k].append(v)
            if args.score:
                try:
                    res, _ = predict_image(img.convert("RGB"), cfg,
                                           model_type=args.model,
                                           with_gradcam=False)
                    scored.append((r.label, src, f["uniq_colors"],
                                   f["flat_frac"],
                                   float(res["probabilities"]["FAKE"]) / 100.0))
                except Exception:
                    pass

    if not feats:
        print("No data.")
        return

    FEATURES = ["flat_frac", "uniq_colors", "axis_edges", "sat_mean"]

    print()
    print("=" * 92)
    print(f"  CONTENT-GENRE STATISTICS BY SOURCE  (split={args.split})")
    print("  flat_frac / axis_edges high => graphics, UI, vector art")
    print("  flat_frac near 0 => camera photography (sensor noise fills every patch)")
    print("=" * 92)
    hdr = f"{'source':<26}{'label':<7}{'n':>5}  " + "".join(f"{f:>13}" for f in FEATURES)
    print(hdr)
    print("-" * len(hdr))
    for src in sorted(feats, key=lambda s: (src_label[s], s)):
        d = feats[src]
        n = len(d["flat_frac"])
        cells = "".join(f"{np.median(d[f]):>13.4f}" for f in FEATURES)
        print(f"{src:<26}{src_label[src]:<7}{n:>5}  {cells}")

    # ------------------------------------------------------------------ AUC
    pools: dict = {"real": defaultdict(list), "fake": defaultdict(list)}
    for src, d in feats.items():
        for f in FEATURES:
            pools[src_label[src]][f].extend(d[f])

    print()
    print("=" * 92)
    print("  AUC — genre feature alone separating FAKE from REAL")
    print("  0.50 = pools genre-matched, model cannot exploit genre")
    print("  0.80+ = the label is partly readable off content genre; the corpus")
    print("          is the thing to fix, not the architecture")
    print("=" * 92)
    aucs = {}
    for f in FEATURES:
        a = auc(pools["fake"][f], pools["real"][f])
        # Report the stronger direction; a feature at 0.2 is as exploitable as 0.8.
        aucs[f] = a
        flag = "  << EXPLOITABLE" if (a > 0.75 or a < 0.25) else ""
        print(f"    {f:<14} AUC(fake>real) = {a:5.3f}   |0.5-AUC| = "
              f"{abs(0.5 - a):5.3f}{flag}")

    n_real = len(pools['real']['flat_frac'])
    n_fake = len(pools['fake']['flat_frac'])
    print(f"\n  pools: real n={n_real}, fake n={n_fake}")

    # ---------------------------------------------- accuracy by genre tercile
    tercile_report = {}
    if args.score and scored:
        print()
        print("=" * 92)
        print("  ACCURACY BY PHOTOGRAPHIC-NESS TERCILE")
        print("  Terciles are cut on uniq_colors (palette richness): T1 = most")
        print("  graphic / restricted palette, T3 = most photographic. Cuts are")
        print("  taken within each class so the two rows are independent.")
        print()
        print("  The real row is the control. If fake accuracy collapses toward")
        print("  T3 while real accuracy holds, the model is reading genre rather")
        print("  than generator, and the corpus is what needs fixing. If both")
        print("  fall together, T3 is simply harder for everyone.")
        print("=" * 92)
        hdr2 = f"{'class':<8}{'n':>5}   " + \
               "".join(f"{t:>22}" for t in ("T1 graphic", "T2", "T3 photographic"))
        print(hdr2)
        print("-" * len(hdr2))
        for lab in ("fake", "real"):
            rows_l = [s for s in scored if s[0] == lab]
            if len(rows_l) < 6:
                continue
            rows_l.sort(key=lambda s: s[2])          # ascending uniq_colors
            n = len(rows_l)
            cuts = [rows_l[: n // 3], rows_l[n // 3: 2 * n // 3], rows_l[2 * n // 3:]]
            want_fake = lab == "fake"
            cells, per_t = "", []
            for t in cuts:
                acc = sum(1 for s in t if (s[4] >= 0.5) == want_fake) / len(t)
                mp = sum(s[4] for s in t) / len(t)
                cells += f"{acc * 100:>13.1f}% (p{mp * 100:>4.1f})"
                per_t.append({"acc": acc, "mean_p_fake": mp, "n": len(t)})
            print(f"{lab:<8}{n:>5}   {cells}")
            tercile_report[lab] = per_t
        print()
        print("  (p## = mean P(FAKE) inside that tercile)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "split": args.split, "roots": args.roots,
        "auc_fake_over_real": aucs,
        "n_real": n_real, "n_fake": n_fake,
        "accuracy_by_genre_tercile": tercile_report,
        "per_source_median": {
            s: {f: float(np.median(feats[s][f])) for f in FEATURES}
            for s in feats},
        "per_source_label": src_label,
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
