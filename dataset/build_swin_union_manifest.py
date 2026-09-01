#!/usr/bin/env python3
"""
TruthLens — Unified SwinV2 Training Manifest Builder
=====================================================
Flattens the prepared training corpora into one
``manifest.jsonl`` with a common schema, computes sha256 for the corpora that
did not record it, and then deduplicates **by sha256 across splits** so no image
can appear in train and in val/test under different paths.

Why this exists
---------------
The SwinV2 training runs feed on the *union* of the prepared corpora
(including ``corpus``, ``modern_v2``, ``legacy_archive``, ``gpt_nano``, and the
new Grok/Aurora set). Each corpus is internally split, but the
union is what the model actually trains on, so the union is what has to be
checked for leakage — the exact failure mode ``diag_corpus_audit.py`` describes
for format/resolution, applied here to content identity.

The output ``sha256`` always describes the bytes that training opens. Source
manifests sometimes hash an image before JPEG conversion, so their value is
kept separately as ``source_sha256``. Computed file hashes are cached in a
sidecar JSON, making rebuilds cheap while still invalidating changed files.

Dedup policy (priority holdout > test > val > train)
-----------------------------------------------------
Rows are grouped by sha256. For a hash that appears in several splits, the
copy in the most protected evaluation split is kept and the others dropped.
This prevents a duplicate train row from silently pulling a designated test or
holdout identity into training. The number of dropped rows is reported per
split so the cost of leak protection is on the record.

Output schema
-------------
``datasets/prepared/swin_union/manifest.jsonl`` — one JSON object per row:

    path, split, label, source, generator, architecture,
    sha256, width, height, corpus

``path`` is absolute (the union spans several corpus roots; relative paths would be
ambiguous). All other columns carry through from the source manifest where
present and default to ``"?"`` where not.

Usage
-----
    .venv/bin/python dataset/build_swin_union_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
OUT_DIR = DATASETS_DIR / "prepared" / "swin_union"

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPLIT_PRIORITY = {"holdout": 0, "test": 1, "val": 2, "train": 3}


def sha256_and_size(path: Path) -> tuple:
    """sha256 hex digest and (width, height) for one image file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001
        w = h = None
    return digest.hexdigest(), w, h


# ─── Per-corpus row extractors ────────────────────────────────────────────────
def rows_corpus_style(corpus: Path, name: str,
                      hash_cache: Dict[str, dict]) -> List[dict]:
    """corpus / highres_corpus / modern_corpus: identical csv schema."""
    manifest = corpus / "manifest.csv"
    if not manifest.exists():
        return []
    out: List[dict] = []
    with manifest.open() as fh:
        for r in csv.DictReader(fh):
            path = corpus / r["path"]
            if not path.exists():
                continue
            split = r["path"].split("/")[0]
            if split not in SPLIT_PRIORITY:
                continue
            info = _cached_hash(path, hash_cache)
            out.append({
                "path": str(path), "split": split,
                "label": r["label"], "source": r.get("source", "?"),
                "generator": r.get("generator", "?"),
                "architecture": r.get("architecture", "?"),
                "sha256": info["sha256"],
                "source_sha256": r.get("sha256", ""),
                "width": info["width"], "height": info["height"],
                "corpus": name,
            })
    return out


def rows_modern_v2(corpus: Path, hash_cache: Dict[str, dict]) -> List[dict]:
    """modern_v2: jsonl manifest with source_key / generator / sha256."""
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        return []
    out: List[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = corpus / r["path"]
        if not path.exists():
            continue
        split = r.get("split", "?")
        if split not in SPLIT_PRIORITY:
            continue
        info = _cached_hash(path, hash_cache)
        out.append({
            "path": str(path), "split": split,
            "label": r.get("label", "?"),
            "source": r.get("source_key", "?"),
            "generator": r.get("generator", "?"),
            "architecture": r.get("architecture", "?"),
            "sha256": info["sha256"],
            "source_sha256": r.get("sha256", ""),
            "width": info["width"], "height": info["height"],
            "corpus": "modern_v2",
        })
    return out


def rows_gpt_nano(corpus: Path, hash_cache: Dict[str, dict]) -> List[dict]:
    """gpt_nano: csv manifest, no sha256 — computed and cached."""
    manifest = corpus / "manifest.csv"
    if not manifest.exists():
        return []
    out: List[dict] = []
    with manifest.open() as fh:
        for r in csv.DictReader(fh):
            path = corpus / r["path"]
            if not path.exists():
                continue
            split = r.get("split", "?")
            if split not in SPLIT_PRIORITY:
                continue
            info = _cached_hash(path, hash_cache)
            out.append({
                "path": str(path), "split": split,
                "label": r.get("label", "?"),
                "source": r.get("generator", "?"),
                "generator": r.get("generator", "?"),
                "architecture": "?",
                "sha256": info["sha256"],
                "width": info["width"], "height": info["height"],
                "corpus": "gpt_nano",
            })
    return out


def rows_legacy_archive(corpus: Path, hash_cache: Dict[str, dict]) -> List[dict]:
    """legacy_archive: csv manifest, no sha256 — computed and cached."""
    manifest = corpus / "manifest.csv"
    if not manifest.exists():
        return []
    out: List[dict] = []
    with manifest.open() as fh:
        for r in csv.DictReader(fh):
            path = corpus / r["path"]
            if not path.exists():
                continue
            split = r.get("split", "?")
            if split not in SPLIT_PRIORITY:
                continue
            info = _cached_hash(path, hash_cache)
            out.append({
                "path": str(path), "split": split,
                "label": r.get("label", "?"),
                "source": r.get("source_archive", "?"),
                "generator": "?",
                "architecture": "?",
                "sha256": info["sha256"],
                "width": info["width"], "height": info["height"],
                "corpus": "legacy_archive",
            })
    return out


def rows_grok_aurora(corpus: Path, hash_cache: Dict[str, dict]) -> List[dict]:
    """Curated Grok/Aurora images extracted from the local preference set.

    These images are additions to training, not a new random validation split;
    the existing source-aware union validation sets remain untouched. The
    extractor records the pre-conversion source hash, so the stored JPEG is
    hashed again here for consistent deduplication.
    """
    manifest = corpus / "manifest_grok.jsonl"
    if not manifest.exists():
        return []
    out: List[dict] = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            path = Path(r["path"])
            if not path.is_absolute():
                path = corpus / path
            if not path.exists() or r.get("label") != "fake":
                continue
            info = _cached_hash(path, hash_cache)
            out.append({
                "path": str(path.resolve()), "split": "train",
                "label": "fake", "source": "xai_aurora_preferences",
                "generator": r.get("generator", "grok"),
                "architecture": r.get("origin", "aurora-20-1-25"),
                "sha256": info["sha256"],
                "source_sha256": r.get("sha256", ""),
                "width": info["width"], "height": info["height"],
                "corpus": "grok_aurora",
            })
    return out


def _cached_hash(path: Path, cache: Dict[str, dict]) -> dict:
    key = str(path)
    hit = cache.get(key)
    stat = path.stat()
    changed = hit is not None and (
        hit.get("size") not in (None, stat.st_size)
        or hit.get("mtime_ns") not in (None, stat.st_mtime_ns)
    )
    if hit is None or changed:
        digest, w, h = sha256_and_size(path)
        hit = {"sha256": digest, "width": w, "height": h,
               "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cache[key] = hit
    elif "size" not in hit or "mtime_ns" not in hit:
        # Upgrade old cache records. Their digest was already computed from the
        # stored bytes; recording file identity makes later edits invalidate it.
        hit = {**hit, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cache[key] = hit
    return hit


def load_hash_cache(path: Path) -> Dict[str, dict]:
    if path.exists():
        return {k: v for k, v in json.loads(path.read_text()).items()}
    return {}


def save_hash_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1))


def dedup(rows: List[dict]) -> tuple[List[dict], dict]:
    """Drop duplicate hashes; preserve the most protected evaluation copy."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    missing = 0
    for r in rows:
        h = r["sha256"]
        if not h:
            missing += 1
            h = f"NO_SHA:{r['path']}"
        groups[h].append(r)

    kept: List[dict] = []
    dropped = Counter()
    conflicting_labels = []
    for h, members in groups.items():
        labels = {r["label"] for r in members}
        if len(labels) > 1:
            conflicting_labels.append({
                "sha256": h, "labels": sorted(labels),
                "paths": [r["path"] for r in members],
            })
        best = min(members, key=lambda r: SPLIT_PRIORITY[r["split"]])
        kept.append(best)
        for r in members:
            if r is not best:
                dropped[f"{r['split']} (dup of {best['split']})"] += 1
    return kept, {"dropped": dict(dropped), "missing_sha": missing,
                  "groups": len(groups),
                  "conflicting_labels": conflicting_labels}


def report_leaks(rows: List[dict]) -> List[dict]:
    """Any remaining hash in both train and a non-train split is a leak."""
    train_hashes = {r["sha256"] for r in rows if r["split"] == "train"}
    leaks = []
    for r in rows:
        if r["split"] != "train" and r["sha256"] in train_hashes:
            leaks.append(r)
    return leaks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--hash-cache", type=Path,
                    default=DATASETS_DIR / "prepared" / ".sha256_cache.json")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the hash cache and recompute all missing hashes")
    args = ap.parse_args()

    prepared = DATASETS_DIR / "prepared"
    all_rows: List[dict] = []
    cache = {} if args.no_cache else load_hash_cache(args.hash_cache)

    extractors = [
        ("corpus", lambda c: rows_corpus_style(c, "corpus", cache)),
        ("highres_corpus", lambda c: rows_corpus_style(c, "highres_corpus", cache)),
        ("modern_corpus", lambda c: rows_corpus_style(c, "modern_corpus", cache)),
        ("modern_v2", lambda c: rows_modern_v2(c, cache)),
        ("gpt_nano", lambda c: rows_gpt_nano(c, cache)),
        ("legacy_archive", lambda c: rows_legacy_archive(c, cache)),
        ("hard_styles", lambda c: rows_grok_aurora(c, cache)),
    ]
    for name, fn in extractors:
        rows = fn(prepared / name)
        print(f"  {name:<16} {len(rows):>7} rows")
        all_rows.extend(rows)

    save_hash_cache(args.hash_cache, cache)

    kept, stats = dedup(all_rows)
    print(f"\n  total rows read     : {len(all_rows)}")
    print(f"  unique sha256 groups: {stats['groups']}")
    print(f"  rows without sha256  : {stats['missing_sha']}")
    print(f"  conflicting labels   : {len(stats['conflicting_labels'])}")
    for why, n in sorted(stats["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:>6}  {why}")
    print(f"  rows kept           : {len(kept)}")

    if stats["conflicting_labels"]:
        for conflict in stats["conflicting_labels"][:10]:
            print(f"    ! {conflict['sha256'][:12]}… labels={conflict['labels']}",
                  file=sys.stderr)
        print("  ERROR: identical bytes have conflicting labels.", file=sys.stderr)
        return 1

    leaks = report_leaks(kept)
    print(f"\n  leaks after dedup    : {len(leaks)}")
    for r in leaks[:10]:
        print(f"    ! {r['path']}  (sha {r['sha256'][:12]}… split={r['split']})")
    if leaks:
        print("  ERROR: a hash survives in both train and a non-train split.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "manifest.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for r in sorted(kept, key=lambda r: (r["split"], r["label"], r["path"])):
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {dest} ({len(kept)} rows)")

    # summary block for quick inspection
    by_split = Counter(r["split"] for r in kept)
    by_label = Counter(r["label"] for r in kept)
    by_corpus = Counter(r["corpus"] for r in kept)
    print("\n  by split :", dict(by_split))
    print("  by label :", dict(by_label))
    print("  by corpus:", dict(by_corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
