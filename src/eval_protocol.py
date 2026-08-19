#!/usr/bin/env python3
"""
TruthLens — src/eval_protocol.py
=================================
The evaluation protocol. One module that decides *which* images a number is
allowed to be computed over, so that "how good is the detector" has an answer
that survives contact with images the training corpus never saw.

Why this exists
---------------
The shipped ResNet18 scores 93.9% on ``datasets/prepared/modern_v2/test`` and 12.0% on
pristine gpt-image-2 from a *different* source repository — the same generator.
Both numbers were computed correctly. The first one is simply not evidence about
generalization, because the test split draws from the same uploads as train.

Worse, pooling hides it. The whole ``holdout`` split scores 87.1% balanced,
which looks healthy and contains the 12.0% inside it. The failure was invisible
until the sets were separated, so separation is the protocol's core rule.

The tiers
---------
Derived from the manifest at runtime rather than hardcoded, so that adding a
source in ``dataset/collect_modern_generators.py`` reclassifies the evaluation
automatically instead of requiring this file to be edited in step.

================  ============================================================
``T1_indist``     ``datasets/prepared/modern_v2/test``. Same source repositories as
                  train. Sanity check only — a high score here is necessary and
                  proves nothing. Reported, never headlined.
``T2_xsource``    Generator *was* trained on, source repository was not. The
                  cleanest measurement available: generator held constant,
                  provenance varied. This is where the current model collapses.
``T3_xgen``       Generator was never trained on, from a source never trained
                  on. What a genuinely new model release looks like.
``T4_wild``       ``datasets/evaluation/internet``. Independent origin, both classes pushed
                  through one thumbnailer so format and resolution carry no
                  class information. Closest thing to a real upload.
================  ============================================================

T2 and T3 borrow their real class from the unseen-source reals (currently the
caption-matched COCO photos shipped alongside the liars-dividend fakes). That is
recorded per tier in ``negatives`` so a shared negative pool is never mistaken
for four independent measurements.

The metrics, and why three of them
----------------------------------
``balanced_accuracy`` at the served threshold is what the product does today.
``auc`` is what the features could do given a correct threshold. These diverge
hard — the frozen-CLIP pilot reaches 0.923 AUC at 0.690 balanced accuracy — and
reporting either one alone hides a different problem: AUC alone hides a
miscalibrated product, accuracy alone hides usable signal behind a bad cutoff.
``recall_at_target_fpr`` is the operational number: of the fakes, how many are
caught when false alarms on real photos are held to 5%.

Usage
-----
    python src/eval_protocol.py --all
    python src/eval_protocol.py --tiers T2_xsource T4_wild --limit 100
    python src/eval_protocol.py --all --threshold-from T2_xsource
    python src/eval_protocol.py --all --tag baseline --checkpoint resnet18_modern.pth
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "results" / "metrics"
SCORE_DIR = OUT_DIR / "scores"

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Splits whose source repositories the model is allowed to have seen. Anything
# outside these is candidate held-out material.
FITTED_SPLITS = ("train", "val")

SERVED_THRESHOLD = 0.5   # argmax over two classes, i.e. what the app serves
TARGET_FPR = 0.05        # false alarms on real photos allowed at the operating point


# ─── Row builders ─────────────────────────────────────────────────────────────
def rows_from_manifest(corpus: Path, split: Optional[str] = None,
                       group_field: str = "source_key") -> List[dict]:
    """Rows from a ``manifest.jsonl``, carrying every field through.

    ``group`` is the field results are broken out by; the raw manifest record
    stays available under the other keys so tiering can look at ``source_key``
    and ``generator`` independently of how results are displayed.

    ``source_key`` and ``generator`` are defaulted rather than assumed present:
    not every corpus in this repo records provenance the same way, and tiering
    reads both on every row. A ``KeyError`` deep inside tier assignment is a
    worse failure than a row that declares its provenance unknown.
    """
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        return []
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if split and r.get("split") != split:
            continue
        path = corpus / r["path"]
        if not path.exists():
            continue
        out.append({"source_key": f"{corpus.name}/unknown", "generator": "?",
                    **r, "path": path, "group": r.get(group_field, "?")})
    return out


def rows_from_tree(root: Path) -> List[dict]:
    """Rows from a plain ``real/`` + ``fake/`` directory tree (no manifest)."""
    out = []
    for label in ("real", "fake"):
        d = root / label
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append({"path": p, "label": label, "group": label,
                            "source_key": f"tree/{root.name}", "generator": "?"})
    return out


def cap_per_group(rows: List[dict], limit: int) -> List[dict]:
    """Cap rows per (group, label). Sorted by filename so runs are comparable."""
    if not limit:
        return list(rows)
    kept, seen = [], collections.Counter()
    for r in sorted(rows, key=lambda x: x["path"].name):
        key = (r["group"], r["label"])
        if seen[key] >= limit:
            continue
        seen[key] += 1
        kept.append(r)
    return kept


# ─── Tiers ────────────────────────────────────────────────────────────────────
@dataclass
class Tier:
    key: str
    title: str
    why: str
    rows: List[dict] = field(default_factory=list)
    negatives: str = ""          # provenance note for the real class
    headline: bool = True        # T1 is excluded; it is not evidence

    @property
    def n_real(self) -> int:
        return sum(r["label"] == "real" for r in self.rows)

    @property
    def n_fake(self) -> int:
        return sum(r["label"] == "fake" for r in self.rows)


TIER_ORDER = ("T1_indist", "T2_xsource", "T3_xgen", "T4_wild")


def build_tiers(limit: int = 0, corpus: Optional[Path] = None,
                wild_corpus: Optional[Path] = None) -> Dict[str, Tier]:
    """Assign every available image to a tier, from the manifest's own fields.

    Nothing here names a specific dataset. A generator counts as held-out
    because no training row mentions it, not because someone remembered to add
    it to a list — which is the failure mode that let the first corpus ship with
    one source per generator and no way to notice.

    Both corpus roots are parameters so this is testable on a synthetic manifest;
    ``wild_corpus`` can be passed a nonexistent path to build the tiers that come
    from ``corpus`` alone.
    """
    corpus = corpus or (ROOT_DIR / "datasets" / "prepared" / "modern_v2")
    wild_corpus = wild_corpus if wild_corpus is not None else (ROOT_DIR / "datasets" / "evaluation" / "internet")
    man = rows_from_manifest(corpus, None)
    if not man:
        return {}

    fitted_sources = {r["source_key"] for r in man if r["split"] in FITTED_SPLITS}
    fitted_gens = {r["generator"] for r in man
                   if r["split"] in FITTED_SPLITS and r["label"] == "fake"}

    outside = [r for r in man if r["split"] not in FITTED_SPLITS
               and r["split"] != "test"]

    # Reals from a repository the model never trained on: the negative class for
    # every held-out tier. Pooled deliberately — there is currently exactly one
    # such source, and pretending otherwise would overstate the evidence.
    negatives = [r for r in outside
                 if r["label"] == "real" and r["source_key"] not in fitted_sources]
    neg_note = ", ".join(sorted({r["source_key"] for r in negatives})) or "(none)"

    fakes_out = [r for r in outside if r["label"] == "fake"]
    t2_fake = [r for r in fakes_out if r["generator"] in fitted_gens
               and r["source_key"] not in fitted_sources]
    # An unseen generator is evidence wherever it came from. When it arrives from
    # a repository that also contributed training rows, that is the *cleanest*
    # form of the question rather than a contaminated one: provenance, packaging
    # and rendition are held constant and the generator is the only variable. So
    # T3 is keyed on the generator alone.
    t3_fake = [r for r in fakes_out if r["generator"] not in fitted_gens]
    # A real leak is a row that is novel on neither axis — same repository *and*
    # same generator as training. Those match no tier above by construction, so
    # this warning is about fixing the corpus, not about what got dropped.
    leaked = [r for r in fakes_out if r["source_key"] in fitted_sources
              and r["generator"] in fitted_gens]
    if leaked:
        keys = sorted({r["source_key"] for r in leaked})
        print(f"  ! {len(leaked)} held-out fakes are novel on neither axis — "
              f"trained source AND trained generator ({', '.join(keys)}); "
              f"they match no tier, this is a leak to fix", file=sys.stderr)

    tiers: Dict[str, Tier] = {}

    t1_rows = rows_from_manifest(corpus, "test")
    if t1_rows:
        tiers["T1_indist"] = Tier(
            "T1_indist", "in-distribution (same sources as train)",
            "sanity only — cannot detect source overfitting",
            t1_rows, negatives="held-out images from trained real sources",
            headline=False)

    if t2_fake and negatives:
        gens = ", ".join(sorted({r["generator"] for r in t2_fake}))
        tiers["T2_xsource"] = Tier(
            "T2_xsource", "held-out SOURCE, trained generator",
            f"same generator ({gens}), unseen provenance",
            t2_fake + negatives, negatives=neg_note)

    if t3_fake and negatives:
        gens = ", ".join(sorted({r["generator"] for r in t3_fake}))
        why = f"never-trained generator ({gens})"
        # Say plainly whether provenance is also novel. Both cases are valid
        # evidence but they answer slightly different questions, and a reader
        # comparing this tier across corpus revisions needs to know which.
        shared = sorted({r["source_key"] for r in t3_fake
                         if r["source_key"] in fitted_sources})
        if shared:
            why += (f"; provenance shared with training ({', '.join(shared)}) "
                    f"so the generator is the only variable")
        tiers["T3_xgen"] = Tier(
            "T3_xgen", "held-out GENERATOR", why,
            t3_fake + negatives, negatives=neg_note)

    wild = rows_from_manifest(wild_corpus, None, "tag")
    if not wild:
        wild = rows_from_tree(wild_corpus)
    if wild:
        tiers["T4_wild"] = Tier(
            "T4_wild", "in the wild (independent origin)",
            "closest available proxy for a real upload",
            wild, negatives="internet reals")

    ordered = {k: tiers[k] for k in TIER_ORDER if k in tiers}
    if limit:
        for t in ordered.values():
            t.rows = cap_per_group(t.rows, limit)
    return ordered


# ─── Metrics ──────────────────────────────────────────────────────────────────
def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on the reals stays <= target.

    Lowest rather than any, because among thresholds that meet the false-alarm
    budget the lowest catches the most fakes. Returns ``inf`` when no threshold
    meets the budget (possible when many reals score above every fake).
    """
    from sklearn.metrics import roc_curve
    if not ((y == 0).any() and (y == 1).any()):
        return float("nan")
    fpr, _tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(fpr <= target_fpr)
    return float(thr[ok[-1]]) if len(ok) else float("inf")


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    real = y == 0
    fake = y == 1
    ra = float((pred[real] == 0).mean()) if real.any() else float("nan")
    fa = float((pred[fake] == 1).mean()) if fake.any() else float("nan")
    return {"threshold": round(float(thr), 4),
            "real_acc": round(ra, 4),
            "fake_recall": round(fa, 4),
            "fpr": round(1.0 - ra, 4) if real.any() else float("nan"),
            "balanced_acc": round((ra + fa) / 2, 4)}


def tier_metrics(rows: Sequence[dict], p: np.ndarray,
                 served_thr: float = SERVED_THRESHOLD,
                 target_fpr: float = TARGET_FPR,
                 applied_thr: Optional[float] = None) -> dict:
    """Every number the protocol reports for one tier.

    ``oracle`` is the threshold chosen on *this* tier's own reals — an upper
    bound on what calibration could buy, not a claim about the product. When
    ``applied_thr`` is given (a threshold fitted elsewhere), the honest
    out-of-sample operating point is reported alongside it.
    """
    from sklearn.metrics import roc_auc_score

    y = np.array([1 if r["label"] == "fake" else 0 for r in rows])
    p = np.asarray(p, dtype=float)

    out = {
        "n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "auc": (round(float(roc_auc_score(y, p)), 4)
                if (y == 0).any() and (y == 1).any() else None),
        "served": at_threshold(y, p, served_thr),
        "mean_p_fake": {
            "real": round(float(p[y == 0].mean()), 4) if (y == 0).any() else None,
            "fake": round(float(p[y == 1].mean()), 4) if (y == 1).any() else None,
        },
    }

    thr = threshold_at_fpr(y, p, target_fpr)
    out["oracle_at_fpr"] = {
        "target_fpr": target_fpr,
        **({} if not np.isfinite(thr) else at_threshold(y, p, thr)),
        "reachable": bool(np.isfinite(thr)),
        # 5% of 100 reals is 5 images: say so rather than imply three decimals
        # of precision the sample size cannot support.
        "fpr_resolution": (round(1.0 / max(int((y == 0).sum()), 1), 4)),
    }
    if applied_thr is not None and np.isfinite(applied_thr):
        out["applied"] = at_threshold(y, p, applied_thr)

    groups: Dict[str, dict] = {}
    for label in ("real", "fake"):
        for g in sorted({r["group"] for r in rows if r["label"] == label}):
            idx = [i for i, r in enumerate(rows)
                   if r["group"] == g and r["label"] == label]
            pg = p[idx]
            hit = (pg >= served_thr) if label == "fake" else (pg < served_thr)
            groups[f"{g}|{label}"] = {
                "n": len(idx),
                "accuracy": round(float(hit.mean()), 4),
                "mean_p_fake": round(float(pg.mean()), 4),
            }
    out["by_group"] = groups
    return out


# ─── Scorers ──────────────────────────────────────────────────────────────────
class ResnetTTAScorer:
    """P(FAKE) from the ResNet18 test-time-augmentation path the app serves.

    Calls ``tta_probs`` rather than ``predict_image``: identical probabilities,
    without recomputing FFT, ELA and prose explanations for every one of a few
    thousand images.
    """

    def __init__(self, cfg, model_type: str = "resnet18") -> None:
        from src.predict import CNNPredictor
        if model_type != "resnet18":
            raise ValueError(f"ResnetTTAScorer cannot serve model_type={model_type!r}")
        self.cfg = cfg
        self.predictor = CNNPredictor(cfg)
        self.name = f"resnet18-tta:{cfg.cnn_model_name}"
        self.meta = dict(self.predictor.checkpoint_meta)

    def p_fake(self, rows: Sequence[dict]) -> List[Optional[float]]:
        from PIL import Image
        from tqdm import tqdm
        out: List[Optional[float]] = []
        for r in tqdm(rows, desc="  scoring", ncols=72, unit="img", leave=False):
            try:
                with Image.open(r["path"]) as im:
                    img = im.convert("RGB")
                probs = self.predictor.tta_probs(img, img)
                out.append(float(probs[1]))          # class order is (REAL, FAKE)
            except Exception as exc:                 # noqa: BLE001
                print(f"  ! {r['path'].name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                out.append(None)
        return out


def _clip_scorer(cfg, **kw):
    """Lazy hook for the frozen-CLIP head.

    Imported inside the call because :mod:`src.score_clip` pulls in open_clip and
    builds a ViT-L/14, which is a few seconds and ~1.7GB — an unacceptable cost
    for ``--model resnet18``, and for ``--reuse-scores``, which constructs no
    scorer at all.
    """
    from src.score_clip import ClipLinearScorer
    return ClipLinearScorer(cfg, **kw)


_clip_scorer.manages_own_checkpoint = True

SCORERS: Dict[str, Callable[..., object]] = {
    "resnet18": ResnetTTAScorer,
    "clip": _clip_scorer,
}


# ─── Score cache ──────────────────────────────────────────────────────────────
def score_path(tag: str) -> Path:
    return SCORE_DIR / f"{tag}.jsonl"


def write_scores(tag: str, records: List[dict]) -> Path:
    """Persist per-image scores so calibration never needs a re-score.

    Fitting a threshold, a Platt scaler or a fusion weight all consume exactly
    these columns. Keeping them means the expensive pass happens once per
    checkpoint instead of once per experiment.
    """
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    dest = score_path(tag)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return dest


def read_scores(tag: str) -> Dict[str, Dict[str, float]]:
    """{tier: {relative_path: p_fake}} from a previous run, or {}."""
    src = score_path(tag)
    if not src.exists():
        return {}
    out: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["tier"]][r["path"]] = r["p_fake"]
    return dict(out)


# ─── Reporting ────────────────────────────────────────────────────────────────
def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def print_tier(tier: Tier, m: dict) -> None:
    flag = "" if tier.headline else "   [not headline]"
    print(f"\n  {tier.key}  {tier.title}{flag}")
    print(f"    {tier.why}")
    print(f"    reals: {tier.negatives}")
    print(f"    {'group':<26}{'label':<7}{'n':>5}{'acc':>9}{'mean P(FAKE)':>15}")
    print("    " + "-" * 62)
    for key, v in m["by_group"].items():
        group, label = key.rsplit("|", 1)
        note = ""
        if label == "fake" and v["accuracy"] < 0.5:
            note = "  << MISSED"
        elif label == "real" and v["accuracy"] < 0.7:
            note = "  << FALSE ALARMS"
        print(f"    {group:<26}{label:<7}{v['n']:>5}{_pct(v['accuracy']):>9}"
              f"{_pct(v['mean_p_fake']):>15}{note}")
    s, o = m["served"], m["oracle_at_fpr"]
    print("    " + "-" * 62)
    print(f"    served  @{s['threshold']:<5}  balanced {_pct(s['balanced_acc']):>7}"
          f"   real {_pct(s['real_acc']):>7}   fake-recall {_pct(s['fake_recall']):>7}")
    if o.get("reachable"):
        print(f"    oracle  @{o['threshold']:<5}  fake-recall {_pct(o['fake_recall']):>7}"
              f" at {_pct(o['target_fpr'])} FPR "
              f"(±{_pct(o['fpr_resolution'])} resolution, n_real={m['n_real']})")
    else:
        print(f"    oracle          no threshold reaches {_pct(o['target_fpr'])} FPR")
    if "applied" in m:
        a = m["applied"]
        print(f"    applied @{a['threshold']:<5}  balanced {_pct(a['balanced_acc']):>7}"
              f"   real {_pct(a['real_acc']):>7}   fake-recall {_pct(a['fake_recall']):>7}"
              f"   (threshold fitted elsewhere)")
    print(f"    AUC {m['auc']}")


def print_board(tiers: Dict[str, Tier], board: Dict[str, dict]) -> None:
    print("\n" + "=" * 88)
    print("  PROTOCOL BOARD — read T2/T3/T4. T1 shares sources with train.")
    print("=" * 88)
    print(f"  {'tier':<13}{'n':>6}{'AUC':>8}{'balanced':>11}{'real':>9}"
          f"{'fake':>9}{'recall@5%FPR':>15}")
    print("  " + "-" * 74)
    for key, t in tiers.items():
        m = board.get(key)
        if not m:
            continue
        s, o = m["served"], m["oracle_at_fpr"]
        rec = _pct(o.get("fake_recall")) if o.get("reachable") else "n/a"
        mark = " " if t.headline else "*"
        print(f" {mark}{key:<13}{m['n']:>6}{(m['auc'] if m['auc'] is not None else 0):>8.3f}"
              f"{_pct(s['balanced_acc']):>11}{_pct(s['real_acc']):>9}"
              f"{_pct(s['fake_recall']):>9}{rec:>15}")
    print("  " + "-" * 74)
    print("  * in-distribution — not evidence of generalization")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="*", default=None, choices=list(TIER_ORDER),
                    help="default: every tier that has data")
    ap.add_argument("--all", action="store_true", help="explicit form of the default")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per group+label (0 = no cap)")
    ap.add_argument("--model", default="resnet18", choices=sorted(SCORERS))
    ap.add_argument("--checkpoint", default=None,
                    help="filename under models/, a repo-relative path, or absolute")
    ap.add_argument("--tag", default=None,
                    help="output label (default: derived from the checkpoint)")
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    ap.add_argument("--threshold-from", default=None, choices=list(TIER_ORDER),
                    help="fit the operating point on this tier, apply it to the rest")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="re-report from a cached score file instead of re-scoring")
    args = ap.parse_args()

    from src.utils import Config, resolve_cnn_checkpoint, set_cnn_checkpoint

    cfg = Config(modern_augment=True)
    # Some scorers carry their own weights. Resolving a CNN checkpoint for them
    # would either fail or, worse, succeed — labelling the run after a ``.pth``
    # that had nothing to do with the numbers.
    own = getattr(SCORERS[args.model], "manages_own_checkpoint", False)
    if args.checkpoint and not own:
        set_cnn_checkpoint(cfg, args.checkpoint)
    elif not own:
        resolve_cnn_checkpoint(cfg)
    if own:
        tag = args.tag or args.model
        served = args.model
    else:
        tag = args.tag or f"{args.model}_{Path(cfg.cnn_model_name).stem}"
        served = cfg.cnn_model_name

    print("=" * 88)
    print(f"  TruthLens evaluation protocol — {args.model} / {served}")
    print("=" * 88)

    tiers = build_tiers(limit=args.limit)
    if args.tiers:
        tiers = {k: v for k, v in tiers.items() if k in args.tiers}
    if not tiers:
        sys.exit("No evaluation tiers have data. Build a corpus first:\n"
                 "  python dataset/collect_modern_generators.py\n"
                 "  python dataset/build_internet_testset.py")

    for k, t in tiers.items():
        print(f"  {k:<13} {t.n_real:>5} real  {t.n_fake:>5} fake   {t.title}")

    cached = read_scores(tag) if args.reuse_scores else {}
    if cached:
        print(f"\n  reusing cached scores from {score_path(tag).relative_to(ROOT_DIR)}")
        scorer = None
    else:
        scorer = SCORERS[args.model](cfg)

    # ── score every tier, keeping the per-image rows for the cache ─────────────
    scored: Dict[str, tuple] = {}
    records: List[dict] = []
    for key, tier in tiers.items():
        hits = cached.get(key, {})
        if hits:
            pairs = [(r, hits.get(str(r["path"].relative_to(ROOT_DIR))))
                     for r in tier.rows]
        else:
            print(f"\n  [{key}] scoring {len(tier.rows)} images")
            pairs = list(zip(tier.rows, scorer.p_fake(tier.rows)))
        keep = [(r, p) for r, p in pairs if p is not None]
        if len(keep) < len(pairs):
            print(f"  [{key}] {len(pairs) - len(keep)} images unscored, dropped")
        if not keep:
            print(f"  [{key}] nothing scored — check the paths")
            continue
        rows = [r for r, _ in keep]
        p = np.array([v for _, v in keep], dtype=float)
        scored[key] = (rows, p)
        for r, v in keep:
            records.append({
                "tier": key,
                "path": str(r["path"].relative_to(ROOT_DIR)),
                "label": r["label"],
                "group": r["group"],
                "source_key": r.get("source_key", "?"),
                "generator": r.get("generator", "?"),
                "p_fake": round(float(v), 6),
            })

    # ── an operating point fitted on one tier, honestly applied to the others ──
    applied: Dict[str, Optional[float]] = {k: None for k in scored}
    if args.threshold_from and args.threshold_from in scored:
        rows, p = scored[args.threshold_from]
        y = np.array([1 if r["label"] == "fake" else 0 for r in rows])
        thr = threshold_at_fpr(y, p, args.target_fpr)
        print(f"\n  operating point fitted on {args.threshold_from} "
              f"at {args.target_fpr:.0%} FPR: threshold {thr:.4f}")
        for k in scored:
            if k != args.threshold_from:
                applied[k] = thr

    board: Dict[str, dict] = {}
    for key, (rows, p) in scored.items():
        board[key] = tier_metrics(rows, p, target_fpr=args.target_fpr,
                                  applied_thr=applied.get(key))
        print_tier(tiers[key], board[key])

    print_board({k: v for k, v in tiers.items() if k in board}, board)

    payload = {
        "model": args.model,
        "checkpoint": served,
        "scorer": getattr(scorer, "name", f"cached:{tag}"),
        "checkpoint_meta": getattr(scorer, "meta", {}),
        "limit_per_group": args.limit,
        "target_fpr": args.target_fpr,
        "threshold_from": args.threshold_from,
        "tiers": {k: {"title": tiers[k].title, "why": tiers[k].why,
                      "negatives": tiers[k].negatives,
                      "headline": tiers[k].headline, **v}
                  for k, v in board.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"protocol_board_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT_DIR)}")
    if records and not cached:
        print(f"Wrote {write_scores(tag, records).relative_to(ROOT_DIR)}  "
              f"({len(records)} per-image scores, reusable with --reuse-scores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
